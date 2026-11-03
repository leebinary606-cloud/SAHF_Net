"""Run a single-season or dual-season PV segmentation experiment with integrated analysis.

This script trains on PreparedDataset train patches, selects checkpoints on
complete validation scenes, and immediately writes per-scene, per-province,
per-scale, and paired comparison reports at completion. It never reads Test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from torch.utils.data import DataLoader

from config import PROJECT_DIR, load_config
from data import PAIR_MODES, EpochShuffleSampler, PairedSentinelDataset, manifest_pos_weight
from models import (
    AutumnOnlyNet,
    SeasonalAuxiliaryControlNet,
    SeasonalConcatNet,
    SpringOnlyNet,
)
from training import MaskedBCEDiceLoss, Trainer, build_optimizer, build_scheduler, set_seed
from training.engine import load_resume_checkpoint, worker_seed

MODEL = "spring_only"
CONFIG = "configs/experiment.yaml"
PAIR_MODE = "paired"
SEED = None
RUN_ID = None
COMPARE_TO = None
RESUME = False

RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
MODEL_CLASSES = {
    "spring_only": SpringOnlyNet,
    "autumn_only": AutumnOnlyNet,
    "concat": SeasonalConcatNet,
    "aux_only": SeasonalAuxiliaryControlNet,
}
MODEL_VARIANTS = tuple(MODEL_CLASSES)


def validate_run_id(value: str) -> str:
    if RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("run_id may only contain letters, digits, '-' and '_'")
    return value


def selected_training_config(training: dict, model_variant: str, seed_override: int | None = None) -> dict:
    """Resolve objective weights and optional seed override without mutating loaded YAML."""
    selected = dict(training)
    if seed_override is not None:
        selected["seed"] = int(seed_override)
    if model_variant == "aux_only":
        selected["auxiliary_loss_weight"] = float(training["auxiliary_loss_weight"])
    else:
        selected["auxiliary_loss_weight"] = 0.0
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_signature(config_path: Path, patch_manifest: Path, scene_manifest: Path,
                   normalization_file: Path, model_variant: str, pair_mode: str,
                   seed: int) -> str:
    source_files = (
        PROJECT_DIR / "config.py",
        PROJECT_DIR / "data" / "dataset.py",
        PROJECT_DIR / "data" / "sampler.py",
        PROJECT_DIR / "models" / "backbone.py",
        PROJECT_DIR / "models" / "fusion_model.py",
        PROJECT_DIR / "training" / "loss.py",
        PROJECT_DIR / "training" / "engine.py",
        PROJECT_DIR / "main.py",
    )
    digest = hashlib.sha256()
    digest.update(f"model_variant={model_variant}".encode("utf-8"))
    digest.update(f"pair_mode={pair_mode}".encode("utf-8"))
    digest.update(f"seed={seed}".encode("utf-8"))
    for path in (config_path, patch_manifest, scene_manifest, normalization_file, *source_files):
        digest.update(str(path.relative_to(PROJECT_DIR.parent)).encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _loader(dataset, batch_size: int, workers: int, shuffle: bool,
            pin_memory: bool, persistent_workers: bool, prefetch_factor: int,
            seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    sampler = EpochShuffleSampler(len(dataset), seed) if shuffle else None
    options = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and workers > 0,
        worker_init_fn=worker_seed,
        generator=generator,
    )
    if workers > 0:
        options["prefetch_factor"] = prefetch_factor
    return DataLoader(**options)


def main(progress: dict) -> None:
    parser = argparse.ArgumentParser(description="Train a seasonal PV segmentation experiment with integrated analysis")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model", choices=MODEL_VARIANTS, required=True)
    parser.add_argument(
        "--pair-mode", choices=PAIR_MODES, default="paired",
        help="paired, duplicate Spring as Autumn, or deterministic cross-scene Autumn",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override training seed")
    parser.add_argument("--run-id", "--run-name", dest="run_id", default=None)
    parser.add_argument(
        "--compare-to", default=None,
        help="Optional reference run_id or directory for paired validation comparison",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(configured_args() if len(sys.argv) == 1 else None)
    progress.update(model=args.model, pair_mode=args.pair_mode, resume=args.resume)
    if args.pair_mode != "paired" and args.model not in {"concat", "aux_only"}:
        raise ValueError("Counterfactual pair modes require a dual-season model (--model concat or aux_only)")

    cfg = load_config(args.config)
    training = selected_training_config(cfg.training, args.model, seed_override=args.seed)
    seed = int(training["seed"])
    set_seed(seed)

    if args.run_id is None:
        args.run_id = {
            "spring_only": f"spv2_spring_only_s{seed}",
            "autumn_only": f"spv2_autumn_only_s{seed}",
            "concat": f"spv2_concat_{args.pair_mode}_s{seed}",
            "aux_only": f"spv2_aux_only_{args.pair_mode}_s{seed}",
        }[args.model]
    run_id = validate_run_id(args.run_id)
    progress.update(run_id=run_id, seed=seed)

    model_config = cfg.model
    model_class = MODEL_CLASSES[args.model]
    configured_architecture = model_config["architectures"][args.model]
    if configured_architecture != model_class.architecture_version:
        raise RuntimeError(
            f"Configured architecture {configured_architecture} != "
            f"implementation {model_class.architecture_version}"
        )

    device = torch.device(str(training["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured device='cuda' is required, but CUDA is unavailable")

    output = cfg.run_root / run_id
    last_checkpoint = output / "checkpoints" / "last.pt"
    if args.resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint missing: {last_checkpoint}")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Run directory already contains files and will not be overwritten: {output}. "
            "Use a new --run-id or --resume."
        )
    output.mkdir(parents=True, exist_ok=True)

    signature = _run_signature(
        cfg.source, cfg.patch_manifest, cfg.scene_manifest, cfg.normalization_file,
        args.model, args.pair_mode, seed,
    )
    progress["run_signature"] = signature
    if output.is_relative_to(PROJECT_DIR):
        progress["output"] = output.relative_to(PROJECT_DIR).as_posix()
    resume = load_resume_checkpoint(last_checkpoint) if args.resume else None
    if resume is not None and resume.get("run_signature") != signature:
        raise RuntimeError("Checkpoint run signature differs from current code/config/data")
    resolved_config = output / "resolved_experiment.yaml"
    if not resolved_config.exists():
        shutil.copy2(cfg.source, resolved_config)
    identity_path = output / "run_identity.json"
    identity = {
        "run_id": run_id,
        "model_variant": args.model,
        "architecture_version": model_class.architecture_version,
        "pair_mode": args.pair_mode,
        "seed": seed,
        "run_signature": signature,
        "training_objective": {
            "auxiliary_loss_weight": float(training["auxiliary_loss_weight"]),
        },
    }
    if identity_path.exists():
        observed_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if observed_identity != identity:
            raise RuntimeError("Run identity differs from the requested model variant")
    else:
        identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")

    train_dataset = PairedSentinelDataset(
        cfg.patch_manifest, cfg.workspace_root, cfg.normalization_file,
        split="train", kind="patch", size=int(cfg.data["patch_size"]),
        augment=bool(cfg.data["geometric_augmentation"]),
        augmentation_seed=seed,
        pair_mode=args.pair_mode,
    )
    validation_dataset = PairedSentinelDataset(
        cfg.scene_manifest, cfg.workspace_root, cfg.normalization_file,
        split="val", kind="scene", size=int(cfg.data["scene_size"]), augment=False,
        pair_mode=args.pair_mode,
    )
    pos_weight, class_statistics = manifest_pos_weight(
        train_dataset.frame, training["pos_weight_power"], training["pos_weight_max"]
    )
    statistics_path = output / "class_statistics.json"
    if not statistics_path.exists():
        statistics_path.write_text(json.dumps(class_statistics, indent=2), encoding="utf-8")

    workers = int(training["num_workers"])
    train_loader = _loader(
        train_dataset, int(training["batch_size"]), workers, True,
        bool(training["pin_memory"]), bool(training["persistent_workers"]),
        int(training["prefetch_factor"]), seed,
    )
    validation_loader = _loader(
        validation_dataset, int(training["validation_batch_size"]), workers, False,
        bool(training["pin_memory"]), bool(training["persistent_workers"]),
        int(training["prefetch_factor"]), seed + 1,
    )

    model_kwargs = {
        "encoder_channels": tuple(model_config["encoder_channels"]),
        "decoder_channels": int(model_config["decoder_channels"]),
        "drop_path_rate": float(model_config["drop_path_rate"]),
        "fusion_dropout": float(model_config["fusion_dropout"]),
    }
    model = model_class(cfg.pretrained_dir, **model_kwargs).to(device)
    criterion = MaskedBCEDiceLoss(
        pos_weight, training["bce_weight"], training["dice_weight"]
    ).to(device)
    optimizer = build_optimizer(model, training)
    scheduler = build_scheduler(optimizer, training)
    trainer = Trainer(model, criterion, optimizer, scheduler, device, output, training, signature)
    summary = trainer.fit(
        train_loader,
        validation_loader,
        int(training["epochs"]),
        resume=resume,
        compare_to=args.compare_to,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    progress["best_validation_micro_iou"] = summary["best_validation_micro_iou"]


def configured_args() -> list[str]:
    args = [
        "--model", MODEL,
        "--config", CONFIG,
        "--pair-mode", PAIR_MODE,
    ]
    for flag, value in (
        ("--seed", SEED),
        ("--run-id", RUN_ID),
        ("--compare-to", COMPARE_TO),
    ):
        if value is not None:
            args.extend((flag, str(value)))
    if RESUME:
        args.append("--resume")
    return args


def run_with_progress() -> None:
    started = datetime.now(timezone.utc)
    directory = PROJECT_DIR / "progress" / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    record = directory / f"{started:%Y%m%dT%H%M%S}-{uuid4().hex[:12]}.json"
    progress = {"started_utc": started.isoformat(), "status": "running"}

    def save() -> None:
        temporary = record.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        temporary.replace(record)

    save()
    try:
        main(progress)
    except SystemExit as error:
        progress["status"] = "help" if error.code == 0 else "invalid_arguments"
        raise
    except KeyboardInterrupt:
        progress["status"] = "interrupted"
        raise
    except Exception as error:
        progress.update(status="failed", error_type=type(error).__name__)
        raise
    else:
        progress["status"] = "completed"
    finally:
        progress["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    run_with_progress()
