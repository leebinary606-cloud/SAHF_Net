"""Runtime configuration for the dual-season PV segmentation benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parent


def _resolve(path: str) -> Path:
    return (PROJECT_DIR / path).resolve()


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    source: Path

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def workspace_root(self) -> Path:
        return _resolve(self.data["workspace_root"])

    @property
    def patch_manifest(self) -> Path:
        return self.workspace_root / self.data["patch_manifest"]

    @property
    def scene_manifest(self) -> Path:
        return self.workspace_root / self.data["scene_manifest"]

    @property
    def normalization_file(self) -> Path:
        return self.workspace_root / self.data["normalization_file"]

    @property
    def pretrained_dir(self) -> Path:
        return _resolve(self.model["pretrained_dir"])

    @property
    def run_root(self) -> Path:
        return _resolve(self.raw["project"]["run_root"])


def load_config(path: str | Path = "configs/experiment.yaml") -> ExperimentConfig:
    source = Path(path)
    if not source.is_absolute():
        source = PROJECT_DIR / source
    source = source.resolve()
    with source.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    required = {"project", "data", "model", "training"}
    if not isinstance(raw, dict) or required.difference(raw):
        raise ValueError(f"Config must contain {sorted(required)}: {source}")
    cfg = ExperimentConfig(raw=raw, source=source)
    for label, target in (
        ("patch manifest", cfg.patch_manifest),
        ("scene manifest", cfg.scene_manifest),
        ("normalization", cfg.normalization_file),
        ("local MiT-B1 weights", cfg.pretrained_dir),
    ):
        if not target.exists():
            raise FileNotFoundError(f"Missing {label}: {target}")
    bands = tuple(cfg.data["bands"])
    if bands != ("B2", "B3", "B4", "B8", "B11", "B12"):
        raise ValueError(f"This experiment requires B2,B3,B4,B8,B11,B12; got {bands}")
    if cfg.training["validation_threshold"] != 0.5:
        raise ValueError("Development validation threshold is fixed at 0.5")
    expected_architectures = {
        "spring_only",
        "autumn_only",
        "concat",
        "aux_only",
    }
    architectures = cfg.model["architectures"]
    if set(architectures) != expected_architectures:
        raise ValueError(f"Model architectures must define {sorted(expected_architectures)}")
    auxiliary_weight = float(cfg.training["auxiliary_loss_weight"])
    if not 0.0 < auxiliary_weight <= 1.0:
        raise ValueError("Auxiliary loss weight must be in (0, 1]")
    return cfg
