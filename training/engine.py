"""Training and integrated post-run diagnostic engine for the PV benchmark."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from affine import Affine
from pyproj import CRS
from tqdm import tqdm


HISTORY_FIELDS = (
    "epoch", "train_loss", "val_loss", "val_micro_iou", "val_precision", "val_recall",
    "spring_weight_mean", "lr_encoder", "lr_new_modules", "epoch_seconds",
)

AUXILIARY_DIAGNOSTIC_FIELDS = ("epoch", "auxiliary_loss_mean")

HEADROOM_SCENE_FIELDS = (
    "scene_id", "valid_pixels", "base_error_pixels",
    "spring_autumn_disagreement_pixels", "spring_autumn_disagreement_fraction",
    "either_season_rescue_pixels", "either_season_rescue_fraction_of_base_errors",
    "spring_unique_rescue_pixels", "autumn_unique_rescue_pixels",
    "base_iou", "spring_iou", "autumn_iou", "final_iou", "oracle_iou",
)

SCENE_ANALYSIS_FIELDS = (
    "scene_id", "province", "spatial_group_id", "scale_bin",
    "valid_mask_sha256", "target_sha256", "threshold", "precision",
    "valid_pixels", "positive_pixels", "positive_ratio",
    "tp", "fp", "fn", "tn",
    "scene_loss", "scene_iou", "scene_f1", "scene_precision", "scene_recall",
    "reference_area_m2", "predicted_area_m2", "area_error_m2",
    "abs_area_error_m2", "false_positive_area_m2",
)

PROVINCE_ANALYSIS_FIELDS = (
    "province", "scene_count", "positive_scene_count",
    "valid_pixels", "positive_pixels", "positive_ratio",
    "tp", "fp", "fn",
    "micro_iou", "macro_positive_scene_iou", "median_scene_iou",
    "micro_f1", "micro_precision", "micro_recall",
    "reference_area_ha", "predicted_area_ha", "mae_area_ha", "rel_area_bias_pct",
)

SCALE_ANALYSIS_FIELDS = (
    "scale_bin", "scene_count", "positive_scene_count",
    "valid_pixels", "positive_pixels", "positive_ratio",
    "tp", "fp", "fn",
    "micro_iou", "macro_positive_scene_iou", "median_scene_iou",
    "micro_f1", "micro_precision", "micro_recall",
    "reference_area_ha", "predicted_area_ha", "mae_area_ha", "rel_area_bias_pct",
)

SCALE_BINS = ("<5%", "5%-15%", ">=15%")


def _scale_bin(positive_ratio: float) -> str:
    if positive_ratio < 0.05:
        return "<5%"
    if positive_ratio < 0.15:
        return "5%-15%"
    return ">=15%"


@lru_cache(maxsize=1024)
def geodesic_row_areas(crs_text: str, transform_tuple: tuple[float, ...], height: int) -> np.ndarray:
    """Compute exact WGS84/projected pixel areas (m^2) per row for north-up rasters."""
    crs = CRS.from_user_input(crs_text)
    transform = Affine(*transform_tuple[:6])
    if transform.b != 0 or transform.d != 0:
        raise ValueError("Row area weights require a north-up raster")
    if crs.is_projected:
        factor = crs.axis_info[0].unit_conversion_factor
        return np.full(height, abs(transform.a * transform.e) * (factor ** 2), dtype=np.float64)
    geod = crs.get_geod()
    values = []
    for row in range(height):
        points = [transform * point for point in ((0, row), (1, row), (1, row + 1), (0, row + 1))]
        lons, lats = zip(*points)
        values.append(abs(geod.polygon_area_perimeter(lons, lats)[0]))
    return np.asarray(values, dtype=np.float64)


def mask_geodesic_area(mask: np.ndarray, row_weights: np.ndarray) -> float:
    return float((np.asarray(mask, dtype=bool).sum(axis=1) * row_weights).sum())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything = set_seed


def worker_seed(worker_id: int) -> None:
    _ = worker_id
    value = torch.initial_seed() % (2 ** 32)
    random.seed(value)
    np.random.seed(value)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    encoder = list(model.shared_encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder}
    new_modules = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    return torch.optim.AdamW([
        {"params": encoder, "lr": float(config["encoder_learning_rate"]), "name": "encoder"},
        {"params": new_modules, "lr": float(config["new_module_learning_rate"]), "name": "new_modules"},
    ], weight_decay=float(config["weight_decay"]))


class WarmupPlateauScheduler:
    """Linear warmup followed by validation-loss-driven LR reductions."""

    def __init__(self, optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> None:
        if config["scheduler"] != "warmup_reduce_on_plateau":
            raise ValueError(f"Unsupported scheduler: {config['scheduler']}")
        if config["plateau_monitor"] != "val_loss":
            raise ValueError("warmup_reduce_on_plateau must monitor val_loss")
        self.optimizer = optimizer
        self.warmup_epochs = int(config["warmup_epochs"])
        if self.warmup_epochs < 1:
            raise ValueError("warmup_epochs must be at least 1")
        self.completed_epochs = 0
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        minimum_lrs = [
            float(config["min_encoder_learning_rate"]),
            float(config["min_new_module_learning_rate"]),
        ]
        if len(minimum_lrs) != len(optimizer.param_groups):
            raise ValueError("Configured minimum learning rates must match optimizer groups")
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config["plateau_factor"]),
            patience=int(config["plateau_patience"]),
            threshold=float(config["plateau_threshold"]),
            threshold_mode="rel",
            min_lr=minimum_lrs,
        )
        self._set_scale(1.0 / self.warmup_epochs)

    def _set_scale(self, scale: float) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * float(scale)

    def step(self, validation_loss: float) -> None:
        self.completed_epochs += 1
        if self.completed_epochs < self.warmup_epochs:
            self._set_scale((self.completed_epochs + 1) / self.warmup_epochs)
            return
        if self.completed_epochs == self.warmup_epochs:
            self._set_scale(1.0)
        self.plateau.step(float(validation_loss))

    def state_dict(self) -> dict[str, Any]:
        return {
            "completed_epochs": self.completed_epochs,
            "base_lrs": self.base_lrs,
            "plateau": self.plateau.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.completed_epochs = int(state["completed_epochs"])
        self.base_lrs = [float(value) for value in state["base_lrs"]]
        self.plateau.load_state_dict(state["plateau"])


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> WarmupPlateauScheduler:
    return WarmupPlateauScheduler(optimizer, config)


def micro_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    return {
        "val_micro_iou": tp / max(1, tp + fp + fn),
        "val_precision": tp / max(1, tp + fp),
        "val_recall": tp / max(1, tp + fn),
    }


def _extended_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    iou = tp / max(1, tp + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * tp) / max(1, 2 * tp + fp + fn)
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def _confusion(prediction: torch.Tensor, truth: torch.Tensor,
               valid: torch.Tensor) -> dict[str, int]:
    return {
        "tp": int((prediction & truth & valid).sum().cpu()),
        "fp": int((prediction & ~truth & valid).sum().cpu()),
        "fn": int((~prediction & truth & valid).sum().cpu()),
    }


def fusion_headroom_statistics(base_logits: torch.Tensor, spring_logits: torch.Tensor,
                               autumn_logits: torch.Tensor, final_logits: torch.Tensor,
                               target: torch.Tensor, valid_mask: torch.Tensor,
                               threshold: float = 0.5) -> dict[str, Any]:
    """Measure label-oracle seasonal complementarity at a fixed threshold."""
    tensors = (base_logits, spring_logits, autumn_logits, final_logits, target)
    if any(tensor.shape != target.shape for tensor in tensors):
        raise ValueError("Headroom logits, target, and mask must share shape")
    truth = target >= 0.5
    valid = valid_mask.bool().expand_as(truth)
    predictions = {
        "base": torch.sigmoid(base_logits.float()) >= threshold,
        "spring": torch.sigmoid(spring_logits.float()) >= threshold,
        "autumn": torch.sigmoid(autumn_logits.float()) >= threshold,
        "final": torch.sigmoid(final_logits.float()) >= threshold,
    }
    correct = {name: prediction.eq(truth) for name, prediction in predictions.items()}
    any_candidate_correct = correct["base"] | correct["spring"] | correct["autumn"]
    oracle = torch.where(any_candidate_correct, truth, predictions["base"])
    confusion = {
        **{name: _confusion(prediction, truth, valid)
           for name, prediction in predictions.items()},
        "oracle": _confusion(oracle, truth, valid),
    }
    base_wrong = valid & ~correct["base"]
    spring_correct = correct["spring"]
    autumn_correct = correct["autumn"]
    return {
        "confusion": confusion,
        "valid_pixels": int(valid.sum().cpu()),
        "base_error_pixels": int(base_wrong.sum().cpu()),
        "spring_autumn_disagreement_pixels": int(
            (valid & predictions["spring"].ne(predictions["autumn"])).sum().cpu()
        ),
        "either_season_rescue_pixels": int(
            (base_wrong & (spring_correct | autumn_correct)).sum().cpu()
        ),
        "spring_unique_rescue_pixels": int(
            (base_wrong & spring_correct & ~autumn_correct).sum().cpu()
        ),
        "autumn_unique_rescue_pixels": int(
            (base_wrong & autumn_correct & ~spring_correct).sum().cpu()
        ),
    }


def generate_loss_curve(history_path: Path, output_path: Path, best_epoch: int,
                        run_name: str) -> None:
    """Plot train/validation loss from the canonical CSV history."""
    if not history_path.is_file():
        raise FileNotFoundError(f"Training history is missing: {history_path}")
    with history_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Training history contains no completed epochs: {history_path}")

    epochs = [int(row["epoch"]) for row in rows]
    train_losses = [float(row["train_loss"]) for row in rows]
    validation_losses = [float(row["val_loss"]) for row in rows]
    if best_epoch not in epochs:
        raise ValueError(f"Best epoch {best_epoch} is absent from {history_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    try:
        axis.plot(epochs, train_losses, label="Train Loss", linewidth=2.0)
        axis.plot(epochs, validation_losses, label="Validation Loss", linewidth=2.0)
        axis.axvline(best_epoch, color="black", linestyle="--", linewidth=1.2,
                     label=f"Best IoU Epoch ({best_epoch})")
        best_index = epochs.index(best_epoch)
        axis.scatter([best_epoch], [validation_losses[best_index]], color="black", s=35, zorder=4)
        axis.set_title(f"Training and Validation Loss - {run_name}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("BCE + Positive-Aware Dice Loss")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
        figure.savefig(temporary, bbox_inches="tight")
        os.replace(temporary, output_path)
    finally:
        plt.close(figure)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(rows: list[dict[str, Any]], fieldnames: tuple[str, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _summarize_group_rows(group_key: str, group_value: str,
                          rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_pixels = sum(int(r["valid_pixels"]) for r in rows)
    positive_pixels = sum(int(r["positive_pixels"]) for r in rows)
    tp = sum(int(r["tp"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)
    metrics = _extended_metrics(tp, fp, fn)
    pos_ious = [float(r["scene_iou"]) for r in rows if int(r["positive_pixels"]) > 0]
    macro_iou = float(np.mean(pos_ious)) if pos_ious else float("nan")
    median_iou = float(np.median(pos_ious)) if pos_ious else float("nan")
    ref_area_ha = sum(float(r["reference_area_m2"]) for r in rows) / 10000.0
    pred_area_ha = sum(float(r["predicted_area_m2"]) for r in rows) / 10000.0
    mae_area_ha = (
        float(np.mean([float(r["abs_area_error_m2"]) for r in rows])) / 10000.0
        if rows else 0.0
    )
    rel_bias_pct = (
        100.0 * (pred_area_ha - ref_area_ha) / ref_area_ha
        if ref_area_ha > 0 else 0.0
    )
    return {
        group_key: group_value,
        "scene_count": len(rows),
        "positive_scene_count": len(pos_ious),
        "valid_pixels": valid_pixels,
        "positive_pixels": positive_pixels,
        "positive_ratio": positive_pixels / max(1, valid_pixels),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_iou": metrics["iou"],
        "macro_positive_scene_iou": macro_iou,
        "median_scene_iou": median_iou,
        "micro_f1": metrics["f1"],
        "micro_precision": metrics["precision"],
        "micro_recall": metrics["recall"],
        "reference_area_ha": ref_area_ha,
        "predicted_area_ha": pred_area_ha,
        "mae_area_ha": mae_area_ha,
        "rel_area_bias_pct": rel_bias_pct,
    }


def compare_scene_tables(candidate_rows: list[dict[str, Any]],
                         reference_rows: list[dict[str, Any]],
                         candidate_name: str,
                         reference_name: str,
                         bootstrap_samples: int = 5000,
                         seed: int = 42) -> dict[str, Any]:
    """Compute paired scene/province/scale delta IoU and spatial-block bootstrap CI."""
    ref_by_id = {str(r["scene_id"]): r for r in reference_rows}
    cand_by_id = {str(r["scene_id"]): r for r in candidate_rows}
    if not candidate_rows or not reference_rows:
        raise ValueError("Comparison requires nonempty scene tables")
    if len(cand_by_id) != len(candidate_rows) or len(ref_by_id) != len(reference_rows):
        raise ValueError("Comparison scene identifiers must be unique")
    if set(cand_by_id) != set(ref_by_id):
        raise ValueError("Candidate and reference per-scene tables do not share the same scene_ids")

    ordered_ids = sorted(cand_by_id)
    cand = [cand_by_id[sid] for sid in ordered_ids]
    ref = [ref_by_id[sid] for sid in ordered_ids]
    for c, r in zip(cand, ref):
        for key in ("valid_mask_sha256", "target_sha256", "threshold", "precision", "spatial_group_id", "province", "scale_bin"):
            if key not in c or key not in r:
                raise ValueError(f"Comparison requires {key}; regenerate historical tables with the recorded model")
            if str(c[key]) != str(r[key]):
                raise ValueError(f"Comparison protocol differs for {c['scene_id']}: {key}")

    def micro_iou_from_subset(rows: list[dict[str, Any]]) -> float:
        tp = sum(int(r["tp"]) for r in rows)
        fp = sum(int(r["fp"]) for r in rows)
        fn = sum(int(r["fn"]) for r in rows)
        return tp / max(1, tp + fp + fn)

    def paired_summary(c_rows: list[dict[str, Any]], r_rows: list[dict[str, Any]]) -> dict[str, Any]:
        c_micro = micro_iou_from_subset(c_rows)
        r_micro = micro_iou_from_subset(r_rows)
        deltas = [float(c["scene_iou"]) - float(r["scene_iou"]) for c, r in zip(c_rows, r_rows)]
        wins = sum(1 for d in deltas if d > 1e-9)
        losses = sum(1 for d in deltas if d < -1e-9)
        ties = len(deltas) - wins - losses
        return {
            "scene_count": len(c_rows),
            "candidate_micro_iou": c_micro,
            "reference_micro_iou": r_micro,
            "delta_micro_iou": c_micro - r_micro,
            "delta_micro_iou_pp": 100.0 * (c_micro - r_micro),
            "delta_macro_scene_iou": float(np.mean(deltas)) if deltas else 0.0,
            "delta_median_scene_iou": float(np.median(deltas)) if deltas else 0.0,
            "scene_wins": wins,
            "scene_losses": losses,
            "scene_ties": ties,
        }

    overall = paired_summary(cand, ref)

    # Spatial-block bootstrap grouped by spatial_group_id
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(cand):
        gid = str(row["spatial_group_id"])
        groups.setdefault(gid, []).append(idx)
    group_keys = sorted(groups)
    group_Count = len(group_keys)

    cand_tp = np.array([sum(int(cand[i]["tp"]) for i in groups[g]) for g in group_keys], dtype=np.float64)
    cand_den = np.array([
        sum(int(cand[i]["tp"]) + int(cand[i]["fp"]) + int(cand[i]["fn"]) for i in groups[g])
        for g in group_keys
    ], dtype=np.float64)
    ref_tp = np.array([sum(int(ref[i]["tp"]) for i in groups[g]) for g in group_keys], dtype=np.float64)
    ref_den = np.array([
        sum(int(ref[i]["tp"]) + int(ref[i]["fp"]) + int(ref[i]["fn"]) for i in groups[g])
        for g in group_keys
    ], dtype=np.float64)

    rng = np.random.default_rng(seed)
    boot_deltas = np.empty(bootstrap_samples, dtype=np.float64)
    for b in range(bootstrap_samples):
        sampled = rng.integers(0, group_Count, size=group_Count)
        c_iou = cand_tp[sampled].sum() / max(1.0, cand_den[sampled].sum())
        r_iou = ref_tp[sampled].sum() / max(1.0, ref_den[sampled].sum())
        boot_deltas[b] = c_iou - r_iou

    overall["spatial_block_count"] = group_Count
    overall["bootstrap_replicates"] = bootstrap_samples
    overall["delta_micro_iou_ci95"] = [
        float(np.percentile(boot_deltas, 2.5)),
        float(np.percentile(boot_deltas, 97.5)),
    ]
    overall["delta_micro_iou_pp_ci95"] = [
        100.0 * float(np.percentile(boot_deltas, 2.5)),
        100.0 * float(np.percentile(boot_deltas, 97.5)),
    ]

    by_province = {}
    provinces = sorted({str(r["province"]) for r in cand})
    for prov in provinces:
        c_sub = [r for r in cand if str(r["province"]) == prov]
        r_sub = [ref_by_id[str(r["scene_id"])] for r in c_sub]
        by_province[prov] = paired_summary(c_sub, r_sub)

    by_scale = {}
    for sbin in SCALE_BINS:
        c_sub = [r for r in cand if str(r["scale_bin"]) == sbin]
        if c_sub:
            r_sub = [ref_by_id[str(r["scene_id"])] for r in c_sub]
            by_scale[sbin] = paired_summary(c_sub, r_sub)

    return {
        "candidate_run": candidate_name,
        "reference_run": reference_name,
        "overall": overall,
        "by_province": by_province,
        "by_scale": by_scale,
    }


def _format_grid_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Format tabular metrics using the exact grid style of Tool/Logger.py."""
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(h)), *(len(r[i]) for r in str_rows)) + 2
        for i, h in enumerate(headers)
    ]
    border = "+" + "+".join("-" * w for w in widths) + "+"
    header_sep = "+" + "+".join("=" * w for w in widths) + "+"
    header_line = "|" + "|".join(f" {str(h):>{w - 2}} " for h, w in zip(headers, widths)) + "|"
    body_lines = []
    for r in str_rows:
        body_lines.append(
            "|" + "|".join(f" {val:>{w - 2}} " for val, w in zip(r, widths)) + "|"
        )
        body_lines.append(border)
    return "\n".join([border, header_line, header_sep, *body_lines])


class Trainer:
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer, scheduler: Any, device: torch.device,
                 output: Path, config: dict[str, Any], run_signature: str) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output = output
        self.config = config
        self.run_signature = run_signature
        self.threshold = float(config["validation_threshold"])
        self.clip = float(config["gradient_clip_norm"])
        self.patience = int(config["early_stopping_patience"])
        self.auxiliary_loss_weight = float(config["auxiliary_loss_weight"])
        self.amp = bool(config["amp"] and device.type == "cuda")
        requested_dtype = str(config["amp_dtype"]).lower()
        if requested_dtype not in {"float16", "bfloat16"}:
            raise ValueError(f"Unsupported AMP dtype: {requested_dtype}")
        self.amp_dtype = torch.bfloat16 if requested_dtype == "bfloat16" else torch.float16
        if self.amp and self.amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Configured BF16 AMP is not supported by this CUDA device")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp and self.amp_dtype == torch.float16)
        self.logger = self._logger()

    def _logger(self) -> logging.Logger:
        logger = logging.getLogger(self.output.name)
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(self.output / "train.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Record a summary metric dictionary in grid table format."""
        headers = list(metrics.keys())
        data = [list(metrics.values())]
        table = _format_grid_table(headers, data)
        self.logger.info("\n" + table + "\n")

    def close(self) -> None:
        """Release file handlers explicitly (important on Windows)."""
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp)

    def _move(self, batch: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        return tuple(batch[key].to(self.device, non_blocking=True) for key in (
            "spring", "autumn", "guidance", "mask", "valid_mask"
        ))

    def train_epoch(self, loader, epoch: int, epochs: int) -> tuple[
        float, float, list[dict[str, Any]]
    ]:
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        auxiliary_loss_sum = 0.0
        auxiliary_loss_count = 0.0
        num_batch = len(loader)
        pbar = tqdm(loader, ncols=150)
        for batch_index, batch in enumerate(pbar, start=1):
            spring, autumn, guidance, target, valid = self._move(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                logits, details = self.model(spring, autumn, guidance, return_details=True)
                main_loss = self.criterion(logits, target, valid)
                objective = main_loss
                auxiliary = details.get("auxiliary")
                auxiliary_loss_detached = None
                if auxiliary is not None:
                    if self.auxiliary_loss_weight <= 0.0:
                        raise ValueError(
                            "Model returned auxiliary predictions but auxiliary_loss_weight is disabled"
                        )
                    spring_auxiliary_loss = self.criterion(
                        auxiliary["spring_logits"], target, valid
                    )
                    autumn_auxiliary_loss = self.criterion(
                        auxiliary["autumn_logits"], target, valid
                    )
                    auxiliary_loss = 0.5 * (
                        spring_auxiliary_loss + autumn_auxiliary_loss
                    )
                    objective = main_loss + self.auxiliary_loss_weight * auxiliary_loss
                    auxiliary_loss_detached = auxiliary_loss.detach()
            if self.scaler.is_enabled():
                self.scaler.scale(objective).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                objective.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            if not torch.isfinite(gradient_norm):
                self.optimizer.zero_grad(set_to_none=True)
                diagnostic = {
                    "event": "nonfinite_gradient", "epoch": epoch, "batch": batch_index,
                    "sample_ids": [str(value) for value in batch["sample_id"]],
                    "gradient_norm": str(gradient_norm.detach().cpu()),
                    "action": "optimizer and scheduler were not advanced",
                }
                _atomic_json(diagnostic, self.output / "nonfinite_gradient.json")
                raise FloatingPointError(
                    f"Non-finite gradient at epoch={epoch}, batch={batch_index}; "
                    f"diagnostic={self.output / 'nonfinite_gradient.json'}"
                )
            if self.scaler.is_enabled():
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scaler.get_scale() < scale_before:
                    raise FloatingPointError("FP16 GradScaler skipped an update; incomplete epoch rejected")
            else:
                self.optimizer.step()
            value = float(main_loss.detach().item())
            total_loss += value
            batch_count += 1
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            pbar.set_description_str(
                f"Epoch[{epoch}/{epochs}], Batch[{batch_index}/{num_batch}]",
                refresh=False,
            )
            if auxiliary_loss_detached is not None:
                aux_val = float(auxiliary_loss_detached.item())
                auxiliary_loss_sum += aux_val
                auxiliary_loss_count += 1.0
                pbar.set_postfix_str(
                    f"loss: {value:.4f}, aux: {aux_val:.4f}, lr: {current_lr:.6f}"
                )
            else:
                pbar.set_postfix_str(f"loss: {value:.4f}, lr: {current_lr:.6f}")
        if batch_count == 0:
            raise RuntimeError("Empty training loader")
        auxiliary_diagnostics = []
        if auxiliary_loss_count > 0:
            auxiliary_diagnostics.append({
                "epoch": epoch,
                "auxiliary_loss_mean": auxiliary_loss_sum / auxiliary_loss_count,
            })
        return total_loss / batch_count, float("nan"), auxiliary_diagnostics

    @torch.no_grad()
    def validate(self, loader, epoch: int, epochs: int) -> dict[str, float]:
        _ = (epoch, epochs)
        self.model.eval()
        scene_loss_sum = 0.0
        scene_count = 0
        tp = fp = fn = 0
        pbar = tqdm(loader, desc="Valid", ncols=150)
        for batch in pbar:
            spring, autumn, guidance, target, valid = self._move(batch)
            with self._autocast():
                logits = self.model(spring, autumn, guidance)
            batch_losses = torch.stack([
                self.criterion(
                    logits[index:index + 1], target[index:index + 1], valid[index:index + 1]
                ).detach()
                for index in range(logits.shape[0])
            ])
            prediction = torch.sigmoid(logits.float()) >= self.threshold
            truth = target >= 0.5
            valid_bool = valid.bool().expand_as(truth)
            batch_counts = torch.stack((
                (prediction & truth & valid_bool).sum(),
                (prediction & ~truth & valid_bool).sum(),
                (~prediction & truth & valid_bool).sum(),
            )).cpu().tolist()
            batch_loss_sum = float(batch_losses.sum().item())
            batch_size = int(logits.shape[0])
            scene_loss_sum += batch_loss_sum
            scene_count += batch_size
            tp += int(batch_counts[0])
            fp += int(batch_counts[1])
            fn += int(batch_counts[2])
            batch_loss = batch_loss_sum / max(1, batch_size)
            running = micro_metrics(tp, fp, fn)["val_micro_iou"]
            pbar.set_postfix_str(f"loss: {batch_loss:.4f}, iou: {running:.4f}")
        if scene_count != len(loader.dataset):
            raise RuntimeError(f"Validation scene count mismatch: {scene_count} != {len(loader.dataset)}")
        return {"val_loss": scene_loss_sum / scene_count, **micro_metrics(tp, fp, fn)}

    @torch.no_grad()
    def calibrate_validation_threshold(self, loader) -> dict[str, Any]:
        """Select one threshold on Validation after the best epoch is fixed."""
        minimum = float(self.config["calibration_threshold_min"])
        maximum = float(self.config["calibration_threshold_max"])
        step = float(self.config["calibration_threshold_step"])
        if not (0.0 < minimum <= maximum < 1.0 and step > 0.0):
            raise ValueError("Invalid validation threshold calibration range")
        count = int(round((maximum - minimum) / step)) + 1
        thresholds = [minimum + index * step for index in range(count)]
        if thresholds[-1] > maximum + 1e-9:
            thresholds.pop()
        confusion = [{"tp": 0, "fp": 0, "fn": 0} for _ in thresholds]

        self.model.eval()
        progress = tqdm(loader, desc="Calibrate", ncols=150)
        for batch in progress:
            spring, autumn, guidance, target, valid = self._move(batch)
            with self._autocast():
                logits = self.model(spring, autumn, guidance)
            probability = torch.sigmoid(logits.float())
            truth = target >= 0.5
            valid_bool = valid.bool().expand_as(truth)
            for index, threshold in enumerate(thresholds):
                prediction = probability >= threshold
                confusion[index]["tp"] += int((prediction & truth & valid_bool).sum().cpu())
                confusion[index]["fp"] += int((prediction & ~truth & valid_bool).sum().cpu())
                confusion[index]["fn"] += int((~prediction & truth & valid_bool).sum().cpu())

        rows = []
        for threshold, counts in zip(thresholds, confusion):
            metrics = micro_metrics(counts["tp"], counts["fp"], counts["fn"])
            rows.append({"threshold": threshold, **counts, **metrics})
        best = max(
            rows,
            key=lambda row: (row["val_micro_iou"], -abs(row["threshold"] - self.threshold)),
        )
        diagnostic_dir = self.output / "diagnostics"
        _atomic_csv(rows, tuple(rows[0]), diagnostic_dir / "threshold_calibration.csv")
        result = {
            "selection_dataset": "validation",
            "checkpoint": "best.pt selected by validation micro IoU at fixed threshold 0.5",
            "threshold": best["threshold"],
            "val_micro_iou": best["val_micro_iou"],
            "val_precision": best["val_precision"],
            "val_recall": best["val_recall"],
            "test_accessed": False,
        }
        _atomic_json(result, diagnostic_dir / "threshold_calibration.json")
        return result

    @torch.no_grad()
    def audit_validation_fusion_headroom(self, loader) -> dict[str, Any]:
        """Audit seasonal complementarity on the already-fixed best checkpoint."""
        if not hasattr(self.model, "validation_headroom_outputs"):
            result = {
                "available": False,
                "selection_dataset": "validation",
                "reason": "model does not expose fixed seasonal candidates",
                "test_accessed": False,
            }
            _atomic_json(result, self.output / "diagnostics" / "fusion_headroom.json")
            return result

        totals = {
            "valid_pixels": 0, "base_error_pixels": 0,
            "spring_autumn_disagreement_pixels": 0,
            "either_season_rescue_pixels": 0,
            "spring_unique_rescue_pixels": 0,
            "autumn_unique_rescue_pixels": 0,
        }
        names = ("base", "spring", "autumn", "final", "oracle")
        confusion = {name: {"tp": 0, "fp": 0, "fn": 0} for name in names}
        scene_rows: list[dict[str, Any]] = []
        self.model.eval()
        progress = tqdm(loader, desc="Headroom", ncols=150)
        for batch in progress:
            spring, autumn, guidance, target, valid = self._move(batch)
            with self._autocast():
                outputs = self.model.validation_headroom_outputs(
                    spring, autumn, guidance
                )
            required = {
                "base_logits", "spring_logits", "autumn_logits", "final_logits"
            }
            if set(outputs) != required:
                raise ValueError(
                    f"Headroom output schema mismatch: {sorted(outputs)} != {sorted(required)}"
                )
            sample_ids = [str(value) for value in batch["sample_id"]]
            for index, scene_id in enumerate(sample_ids):
                statistics = fusion_headroom_statistics(
                    outputs["base_logits"][index:index + 1],
                    outputs["spring_logits"][index:index + 1],
                    outputs["autumn_logits"][index:index + 1],
                    outputs["final_logits"][index:index + 1],
                    target[index:index + 1], valid[index:index + 1], self.threshold,
                )
                for key in totals:
                    totals[key] += statistics[key]
                for name in names:
                    for key in ("tp", "fp", "fn"):
                        confusion[name][key] += statistics["confusion"][name][key]
                base_errors = statistics["base_error_pixels"]
                valid_pixels = statistics["valid_pixels"]
                scene_metrics = {
                    name: micro_metrics(**statistics["confusion"][name])["val_micro_iou"]
                    for name in names
                }
                scene_rows.append({
                    "scene_id": scene_id,
                    "valid_pixels": valid_pixels,
                    "base_error_pixels": base_errors,
                    "spring_autumn_disagreement_pixels": statistics[
                        "spring_autumn_disagreement_pixels"
                    ],
                    "spring_autumn_disagreement_fraction": (
                        statistics["spring_autumn_disagreement_pixels"]
                        / max(1, valid_pixels)
                    ),
                    "either_season_rescue_pixels": statistics[
                        "either_season_rescue_pixels"
                    ],
                    "either_season_rescue_fraction_of_base_errors": (
                        statistics["either_season_rescue_pixels"]
                        / max(1, base_errors)
                    ),
                    "spring_unique_rescue_pixels": statistics[
                        "spring_unique_rescue_pixels"
                    ],
                    "autumn_unique_rescue_pixels": statistics[
                        "autumn_unique_rescue_pixels"
                    ],
                    "base_iou": scene_metrics["base"],
                    "spring_iou": scene_metrics["spring"],
                    "autumn_iou": scene_metrics["autumn"],
                    "final_iou": scene_metrics["final"],
                    "oracle_iou": scene_metrics["oracle"],
                })

        diagnostic_dir = self.output / "diagnostics"
        _atomic_csv(scene_rows, HEADROOM_SCENE_FIELDS, diagnostic_dir / "fusion_headroom_per_scene.csv")
        candidates = {
            name: {**counts, **micro_metrics(**counts)}
            for name, counts in confusion.items()
        }
        result = {
            "available": True,
            "selection_dataset": "validation",
            "checkpoint": "fixed best.pt selected before this audit",
            "threshold": self.threshold,
            "diagnostic_only": True,
            "oracle_definition": (
                "label-aware per-pixel choice among base, Spring, and Autumn; "
                "optimistic upper bound, not a deployable model"
            ),
            "valid_pixel_count": totals["valid_pixels"],
            "base_error_pixel_count": totals["base_error_pixels"],
            "spring_autumn_disagreement_fraction": (
                totals["spring_autumn_disagreement_pixels"]
                / max(1, totals["valid_pixels"])
            ),
            "either_season_rescue_fraction_of_base_errors": (
                totals["either_season_rescue_pixels"]
                / max(1, totals["base_error_pixels"])
            ),
            "spring_unique_rescue_fraction_of_base_errors": (
                totals["spring_unique_rescue_pixels"]
                / max(1, totals["base_error_pixels"])
            ),
            "autumn_unique_rescue_fraction_of_base_errors": (
                totals["autumn_unique_rescue_pixels"]
                / max(1, totals["base_error_pixels"])
            ),
            "candidates": candidates,
            "test_accessed": False,
        }
        _atomic_json(result, diagnostic_dir / "fusion_headroom.json")
        return result

    def _scene_metadata_lookup(self, dataset) -> dict[str, dict[str, Any]]:
        frame = dataset.frame
        if frame.scene_id.duplicated().any():
            raise ValueError("Scene metadata contains duplicate identifiers")
        lookup: dict[str, dict[str, Any]] = {}
        for _, row in frame.iterrows():
            transform_tuple = tuple(float(row["transform_" + key]) for key in "abcdef")
            lookup[str(row["scene_id"])] = {
                "province": str(row["province"]),
                "spatial_group_id": str(row["spatial_group_id"]),
                "row_weights": geodesic_row_areas(str(row["crs"]), transform_tuple, int(row["height"])),
            }
        return lookup

    @torch.no_grad()
    def analyze_validation_results(
        self,
        loader,
        compare_to: Path | str | None = None,
    ) -> dict[str, Any]:
        """Generate per-scene, per-province, per-scale, and paired comparison reports."""
        self.model.eval()
        metadata = self._scene_metadata_lookup(loader.dataset)
        scene_rows: list[dict[str, Any]] = []
        total_forward_seconds = 0.0
        total_scenes = 0

        progress = tqdm(loader, desc="Test", ncols=150)
        for batch in progress:
            spring, autumn, guidance, target, valid = self._move(batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            with self._autocast():
                logits = self.model(spring, autumn, guidance)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            total_forward_seconds += time.perf_counter() - t0
            total_scenes += int(logits.shape[0])

            probabilities = torch.sigmoid(logits.float())
            predictions = probabilities >= self.threshold
            truths = target >= 0.5
            valid_bools = valid.bool().expand_as(truths)
            sample_ids = [str(v) for v in batch["sample_id"]]
            batch_loss_sum = 0.0

            for idx, scene_id in enumerate(sample_ids):
                scene_loss = float(self.criterion(
                    logits[idx:idx + 1], target[idx:idx + 1], valid[idx:idx + 1]
                ).detach().cpu())
                batch_loss_sum += scene_loss
                pred_np = predictions[idx, 0].cpu().numpy()
                truth_np = truths[idx, 0].cpu().numpy()
                valid_np = valid_bools[idx, 0].cpu().numpy()

                pred_valid = pred_np & valid_np
                truth_valid = truth_np & valid_np
                tp = int((pred_valid & truth_valid).sum())
                fp = int((pred_valid & ~truth_valid).sum())
                fn = int((~pred_valid & truth_valid).sum())
                tn = int((~pred_valid & ~truth_valid & valid_np).sum())
                valid_pixels = int(valid_np.sum())
                positive_pixels = int(truth_valid.sum())
                positive_ratio = positive_pixels / max(1, valid_pixels)
                metrics = _extended_metrics(tp, fp, fn)

                meta = metadata[scene_id]
                province = meta["province"]
                spatial_group_id = meta["spatial_group_id"]
                row_weights = meta["row_weights"]

                ref_area_m2 = mask_geodesic_area(truth_valid, row_weights)
                pred_area_m2 = mask_geodesic_area(pred_valid, row_weights)
                fp_area_m2 = mask_geodesic_area(pred_valid & ~truth_valid, row_weights)
                area_err_m2 = pred_area_m2 - ref_area_m2

                scene_rows.append({
                    "scene_id": scene_id,
                    "province": province,
                    "spatial_group_id": spatial_group_id,
                    "scale_bin": _scale_bin(positive_ratio),
                    "valid_mask_sha256": hashlib.sha256(valid_np.tobytes()).hexdigest(),
                    "target_sha256": hashlib.sha256(truth_valid.tobytes()).hexdigest(),
                    "threshold": self.threshold,
                    "precision": str(self.amp_dtype) if self.amp else "torch.float32",
                    "valid_pixels": valid_pixels,
                    "positive_pixels": positive_pixels,
                    "positive_ratio": positive_ratio,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "scene_loss": scene_loss,
                    "scene_iou": metrics["iou"],
                    "scene_f1": metrics["f1"],
                    "scene_precision": metrics["precision"],
                    "scene_recall": metrics["recall"],
                    "reference_area_m2": ref_area_m2,
                    "predicted_area_m2": pred_area_m2,
                    "area_error_m2": area_err_m2,
                    "abs_area_error_m2": abs(area_err_m2),
                    "false_positive_area_m2": fp_area_m2,
                })
            progress.set_postfix_str(f"loss: {batch_loss_sum / max(1, len(sample_ids)):.4f}")

        scene_rows.sort(key=lambda r: str(r["scene_id"]))
        diagnostic_dir = self.output / "diagnostics"
        _atomic_csv(scene_rows, SCENE_ANALYSIS_FIELDS, diagnostic_dir / "val_per_scene.csv")

        provinces = sorted({str(r["province"]) for r in scene_rows})
        province_rows = [
            _summarize_group_rows("province", prov, [r for r in scene_rows if str(r["province"]) == prov])
            for prov in provinces
        ]
        _atomic_csv(province_rows, PROVINCE_ANALYSIS_FIELDS, diagnostic_dir / "val_per_province.csv")

        scale_rows = [
            _summarize_group_rows("scale_bin", sbin, [r for r in scene_rows if str(r["scale_bin"]) == sbin])
            for sbin in SCALE_BINS
            if any(str(r["scale_bin"]) == sbin for r in scene_rows)
        ]
        _atomic_csv(scale_rows, SCALE_ANALYSIS_FIELDS, diagnostic_dir / "val_by_scale.csv")

        overall = _summarize_group_rows("split", "val", scene_rows)
        overall["mean_scene_loss"] = float(np.mean([float(r["scene_loss"]) for r in scene_rows]))
        overall["inference_ms_per_scene"] = 1000.0 * total_forward_seconds / max(1, total_scenes)
        overall["total_parameters"] = int(sum(p.numel() for p in self.model.parameters()))
        overall["trainable_parameters"] = int(
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        )

        hardest_scenes = sorted(scene_rows, key=lambda r: float(r["scene_iou"]))[:10]

        comparison_report = None
        ref_path = self._resolve_comparison_csv(compare_to)
        if ref_path is not None and ref_path.is_file():
            with ref_path.open("r", newline="", encoding="utf-8-sig") as handle:
                ref_rows = list(csv.DictReader(handle))
            ref_name = ref_path.parent.parent.name
            comparison_report = compare_scene_tables(
                scene_rows, ref_rows, self.output.name, ref_name
            )
            _atomic_json(
                comparison_report,
                diagnostic_dir / f"comparison_vs_{ref_name}.json",
            )

        report = {
            "run_name": self.output.name,
            "architecture": self.model.architecture_version,
            "threshold": self.threshold,
            "overall": overall,
            "by_province": {row["province"]: row for row in province_rows},
            "by_scale": {row["scale_bin"]: row for row in scale_rows},
            "hardest_scenes": [
                {
                    "scene_id": r["scene_id"],
                    "province": r["province"],
                    "scale_bin": r["scale_bin"],
                    "positive_ratio": r["positive_ratio"],
                    "scene_iou": r["scene_iou"],
                    "scene_precision": r["scene_precision"],
                    "scene_recall": r["scene_recall"],
                }
                for r in hardest_scenes
            ],
            "comparison": comparison_report,
            "test_accessed": False,
        }
        _atomic_json(report, diagnostic_dir / "val_analysis_report.json")
        self._log_analysis_table(overall, province_rows, scale_rows, comparison_report)
        return report

    def _resolve_comparison_csv(self, compare_to: Path | str | None) -> Path | None:
        if compare_to:
            candidate = Path(compare_to)
            if candidate.is_file():
                return candidate
            if (candidate / "diagnostics" / "val_per_scene.csv").is_file():
                return candidate / "diagnostics" / "val_per_scene.csv"
            sibling = self.output.parent / str(compare_to) / "diagnostics" / "val_per_scene.csv"
            if sibling.is_file():
                return sibling
            raise FileNotFoundError(f"Could not find val_per_scene.csv for --compare-to {compare_to}")
        # Auto-detect default spring_only reference in the same run_root if different from self
        for default_name in ("spv2_spring_only_s42", "baseline_spring_only_v1"):
            if self.output.name != default_name:
                auto_path = self.output.parent / default_name / "diagnostics" / "val_per_scene.csv"
                if auto_path.is_file():
                    return auto_path
        return None

    def _log_analysis_table(
        self,
        overall: dict[str, Any],
        province_rows: list[dict[str, Any]],
        scale_rows: list[dict[str, Any]],
        comparison: dict[str, Any] | None,
    ) -> None:
        self.log_metrics({
            "Val_Loss": round(float(overall["mean_scene_loss"]), 4),
            "Micro_IoU": round(float(overall["micro_iou"]), 4),
            "Macro_IoU": round(float(overall["macro_positive_scene_iou"]), 4),
            "Precision": round(float(overall["micro_precision"]), 4),
            "Recall": round(float(overall["micro_recall"]), 4),
            "F1_Score": round(float(overall["micro_f1"]), 4),
            "Area_MAE_ha": round(float(overall["mae_area_ha"]), 4),
            "Infer_ms": round(float(overall["inference_ms_per_scene"]), 2),
        })
        prov_headers = ["Province", "Scenes", "Micro_IoU", "Macro_IoU", "Precision", "Recall", "F1_Score", "Area_MAE_ha"]
        prov_data = [
            [
                row["province"],
                row["scene_count"],
                f"{row['micro_iou']:.4f}",
                f"{row['macro_positive_scene_iou']:.4f}",
                f"{row['micro_precision']:.4f}",
                f"{row['micro_recall']:.4f}",
                f"{row['micro_f1']:.4f}",
                f"{row['mae_area_ha']:.4f}",
            ]
            for row in province_rows
        ]
        scale_headers = ["Scale_Bin", "Scenes", "Micro_IoU", "Macro_IoU", "Precision", "Recall", "F1_Score", "Area_MAE_ha"]
        scale_data = [
            [
                row["scale_bin"],
                row["scene_count"],
                f"{row['micro_iou']:.4f}",
                f"{row['macro_positive_scene_iou']:.4f}",
                f"{row['micro_precision']:.4f}",
                f"{row['micro_recall']:.4f}",
                f"{row['micro_f1']:.4f}",
                f"{row['mae_area_ha']:.4f}",
            ]
            for row in scale_rows
        ]
        blocks = [
            _format_grid_table(prov_headers, prov_data),
            _format_grid_table(scale_headers, scale_data),
        ]
        if comparison is not None:
            ov = comparison["overall"]
            ci_low, ci_high = ov["delta_micro_iou_pp_ci95"]
            comp_headers = ["Reference_Run", "Delta_Micro_IoU_pp", "CI95_Low_pp", "CI95_High_pp", "Wins/Losses/Ties"]
            comp_data = [[
                comparison["reference_run"],
                f"{ov['delta_micro_iou_pp']:+.2f}",
                f"{ci_low:+.2f}",
                f"{ci_high:+.2f}",
                f"{ov['scene_wins']}/{ov['scene_losses']}/{ov['scene_ties']}",
            ]]
            blocks.append(_format_grid_table(comp_headers, comp_data))
        self.logger.info("\n" + "\n\n".join(blocks) + "\n")

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        resume: dict[str, Any] | None = None,
        compare_to: Path | str | None = None,
    ) -> dict[str, Any]:
        start_epoch = 1
        best_iou = -1.0
        best_epoch = 0
        patience_counter = 0
        if resume is not None:
            if resume.get("run_signature") != self.run_signature:
                raise RuntimeError("Checkpoint run signature differs from current code/config/data")
            self.model.load_state_dict(resume["model"])
            self.optimizer.load_state_dict(resume["optimizer"])
            self.scheduler.load_state_dict(resume["scheduler"])
            if self.scaler.is_enabled() and resume.get("scaler"):
                self.scaler.load_state_dict(resume["scaler"])
            start_epoch = int(resume["completed_epoch"]) + 1
            best_iou = float(resume["best_iou"])
            best_epoch = int(resume["best_epoch"])
            patience_counter = int(resume["patience_counter"])
            restore_rng_state(resume["rng_state"])
            self.logger.info(
                f"Checkpoint loaded successfully! epoch: {start_epoch - 1}, "
                f"path: {self.output / 'checkpoints' / 'last.pt'}"
            )

        if start_epoch > epochs:
            raise RuntimeError(
                f"Checkpoint already completed epoch {start_epoch - 1}, configured epochs={epochs}"
            )

        history_path = self.output / "training_history.csv"
        for epoch in range(start_epoch, epochs + 1):
            started = time.perf_counter()
            current_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
            (
                train_loss,
                spring_weight_mean,
                auxiliary_diagnostics,
            ) = self.train_epoch(train_loader, epoch, epochs)
            validation = self.validate(val_loader, epoch, epochs)
            improved = validation["val_micro_iou"] > best_iou
            if improved:
                best_iou = validation["val_micro_iou"]
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            elapsed = time.perf_counter() - started
            row = {
                "epoch": epoch, "train_loss": train_loss, **validation,
                "spring_weight_mean": spring_weight_mean,
                "lr_encoder": current_lrs[0], "lr_new_modules": current_lrs[1],
                "epoch_seconds": elapsed,
            }
            self._append_history(history_path, row)
            if auxiliary_diagnostics:
                self._append_auxiliary_diagnostics(
                    self.output / "diagnostics" / "auxiliary_diagnostics.csv",
                    auxiliary_diagnostics,
                )
            self.scheduler.step(validation["val_loss"])
            payload = {
                "format_version": 1,
                "architecture_version": self.model.architecture_version,
                "run_signature": self.run_signature,
                "completed_epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
                "best_iou": best_iou, "best_epoch": best_epoch,
                "patience_counter": patience_counter,
                "rng_state": capture_rng_state(),
            }
            _atomic_torch_save(payload, self.output / "checkpoints" / "last.pt")
            if improved:
                _atomic_torch_save(payload, self.output / "checkpoints" / "best.pt")
            aux_suffix = (
                f", Aux Loss: {auxiliary_diagnostics[0]['auxiliary_loss_mean']:.6f}"
                if auxiliary_diagnostics else ""
            )
            self.logger.info(
                f"Epoch[{epoch}], Train Loss: {train_loss:.6f}, "
                f"Val Loss: {validation['val_loss']:.6f}{aux_suffix}, "
                f"Val IoU: {validation['val_micro_iou']:.6f}, "
                f"Precision: {validation['val_precision']:.6f}, "
                f"Recall: {validation['val_recall']:.6f}, "
                f"Best IoU: {best_iou:.6f}@{best_epoch}, "
                f"Patience: {patience_counter}/{self.patience}, "
                f"lr: {current_lrs[0]:.6f}"
            )
            if patience_counter >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch}!\n")
                break

        best_checkpoint_path = self.output / "checkpoints" / "best.pt"
        best_checkpoint = torch.load(
            best_checkpoint_path, map_location=self.device, weights_only=False
        )
        if best_checkpoint.get("run_signature") != self.run_signature:
            raise RuntimeError("Best checkpoint run signature differs before threshold calibration")
        self.model.load_state_dict(best_checkpoint["model"])
        self.logger.info(f"Model weights loaded successfully! path: {best_checkpoint_path}")
        calibration = self.calibrate_validation_threshold(val_loader)
        headroom = self.audit_validation_fusion_headroom(val_loader)
        analysis = self.analyze_validation_results(val_loader, compare_to=compare_to)
        summary = {
            "architecture": self.model.architecture_version,
            "best_validation_micro_iou": best_iou,
            "best_epoch": best_epoch,
            "last_completed_epoch": epoch,
            "checkpoint_selection": "validation micro IoU at fixed threshold 0.5",
            "test_accessed": False,
            "validation_threshold_calibration": calibration,
            "validation_fusion_headroom": headroom,
            "validation_analysis": {
                "overall": analysis["overall"],
                "by_province": analysis["by_province"],
                "by_scale": analysis["by_scale"],
                "comparison": analysis["comparison"],
            },
        }
        generate_loss_curve(
            history_path,
            self.output / "diagnostics" / "loss_curve.png",
            best_epoch,
            self.output.name,
        )
        _atomic_json(summary, self.output / "training_summary.json")
        self.close()
        return summary

    @staticmethod
    def _append_history(path: Path, row: dict[str, Any]) -> None:
        if tuple(row) != HISTORY_FIELDS:
            raise ValueError(f"History schema mismatch: {tuple(row)} != {HISTORY_FIELDS}")
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _append_auxiliary_diagnostics(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            raise ValueError("Auxiliary diagnostics are empty")
        if any(tuple(row) != AUXILIARY_DIAGNOSTIC_FIELDS for row in rows):
            raise ValueError("Auxiliary diagnostic schema mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=AUXILIARY_DIAGNOSTIC_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)


def load_resume_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    return payload
