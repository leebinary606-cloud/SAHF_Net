"""Paired Sentinel-2 train patches and full validation/test scenes."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


PATH_COLUMNS = {
    "patch": ("spring_patch_path", "autumn_patch_path", "mask_patch_path"),
    "scene": ("spring_path", "autumn_path", "scene_mask_path"),
}

PAIR_MODES = ("paired", "spring_duplicate", "mismatched_autumn")


def build_mismatched_indices(frame: pd.DataFrame, kind: str) -> np.ndarray:
    """Return a deterministic cross-scene derangement for Autumn controls.

    The counterfactual deliberately keeps the Spring image and label fixed but
    takes Autumn from another scene.  A cyclic derangement is preferable to a
    random permutation because it is reproducible without hidden RNG state and
    guarantees that no sample accidentally keeps its correctly paired season.
    """
    if kind not in PATH_COLUMNS:
        raise ValueError(f"Unknown dataset kind: {kind}")
    scene_column = "parent_scene_id" if kind == "patch" else "scene_id"
    if scene_column not in frame:
        raise ValueError(f"Manifest lacks {scene_column} required for mismatch control")
    scene_ids = frame[scene_column].astype(str).to_numpy()
    if len(scene_ids) < 2 or len(np.unique(scene_ids)) < 2:
        raise ValueError("Mismatched-Autumn control requires at least two scenes")
    indices = np.arange(len(frame), dtype=np.int64)
    for shift in range(1, len(frame)):
        candidate = np.roll(indices, -shift)
        if np.all(scene_ids[candidate] != scene_ids):
            return candidate
    raise ValueError("Cannot construct a cross-scene Autumn derangement")


def _resolve(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Manifest target missing: {target}")
    return target


def _read_triplet(paths: tuple[Path, Path, Path], expected_size: int,
                  require_spatial_alignment: bool = True) -> tuple[np.ndarray, ...]:
    with rasterio.open(paths[0]) as spring_src, rasterio.open(paths[1]) as autumn_src, rasterio.open(paths[2]) as mask_src:
        expected = [(6, expected_size, expected_size), (6, expected_size, expected_size), (1, expected_size, expected_size)]
        observed = [(src.count, src.height, src.width) for src in (spring_src, autumn_src, mask_src)]
        if observed != expected:
            raise ValueError(f"Raster shape mismatch: expected={expected}, observed={observed}, paths={paths}")
        if spring_src.crs != mask_src.crs:
            raise ValueError(f"Spring/mask CRS mismatch: {paths}")
        if not spring_src.transform.almost_equals(mask_src.transform):
            raise ValueError(f"Spring/mask grid mismatch: {paths}")
        if require_spatial_alignment and spring_src.crs != autumn_src.crs:
            raise ValueError(f"Spring/Autumn CRS mismatch: {paths}")
        if require_spatial_alignment and not spring_src.transform.almost_equals(
            autumn_src.transform
        ):
            raise ValueError(f"Spring/Autumn grid mismatch: {paths}")
        spring_ma = spring_src.read(masked=True, out_dtype="float32")
        autumn_ma = autumn_src.read(masked=True, out_dtype="float32")
        mask_ma = mask_src.read(1, masked=True, out_dtype="float32")
        spring = np.ma.filled(spring_ma, np.nan)
        autumn = np.ma.filled(autumn_ma, np.nan)
        mask = np.ma.filled(mask_ma, 0.0)
        valid = (
            ~np.ma.getmaskarray(spring_ma).any(0)
            & ~np.ma.getmaskarray(autumn_ma).any(0)
            & ~np.ma.getmaskarray(mask_ma)
            & np.isfinite(spring).all(0)
            & np.isfinite(autumn).all(0)
            & np.isfinite(mask)
        )
    valid_mask_values = mask[valid]
    if not ((valid_mask_values == 0.0) | (valid_mask_values == 1.0)).all():
        labels = np.unique(valid_mask_values)
        raise ValueError(f"Mask contains non-binary values: {labels[:20].tolist()}")
    return spring, autumn, mask[None], valid[None]


def _safe_index(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=np.abs(denominator) > 1e-6)
    return np.clip(result, -1.0, 1.0)


def spectral_temporal_guidance(spring: np.ndarray, autumn: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Build NDVI/NDBI/MNDWI mean, magnitude, and ordered-change maps.

    Band order is B2,B3,B4,B8,B11,B12. Indices are computed before image
    normalization so their physical ratios remain meaningful. The final three
    channels are Autumn minus Spring and intentionally retain seasonal order.
    """
    def indices(image: np.ndarray) -> np.ndarray:
        b3, b4, b8, b11 = image[1], image[2], image[3], image[4]
        return np.stack((
            _safe_index(b8 - b4, b8 + b4),       # NDVI
            _safe_index(b11 - b8, b11 + b8),     # NDBI
            _safe_index(b3 - b11, b3 + b11),     # MNDWI
        )).astype(np.float32)

    spring_indices = indices(spring)
    autumn_indices = indices(autumn)
    guidance = np.concatenate((
        0.5 * (spring_indices + autumn_indices),
        np.abs(spring_indices - autumn_indices),
        autumn_indices - spring_indices,
    )).astype(np.float32)
    return np.where(valid, guidance, 0.0).astype(np.float32)


class JointNormalizer:
    def __init__(self, path: Path) -> None:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        joint = payload["joint"]
        mean = np.asarray(joint["mean"], dtype=np.float32)
        std = np.asarray(joint["std"], dtype=np.float32)
        if mean.shape != (6,) or std.shape != (6,):
            raise ValueError("Normalization requires six band means and standard deviations")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Normalization means must be finite; standard deviations must be finite and positive")
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(image), image, self.mean)
        return ((filled - self.mean) / self.std).astype(np.float32)


def _augment(items: list[torch.Tensor], rng: random.Random) -> list[torch.Tensor]:
    if rng.random() < 0.5:
        items = [torch.flip(item, (-1,)) for item in items]
    if rng.random() < 0.5:
        items = [torch.flip(item, (-2,)) for item in items]
    turns = rng.randrange(4)
    return [torch.rot90(item, turns, (-2, -1)) for item in items] if turns else items


class PairedSentinelDataset(Dataset):
    """One dataset implementation for train patches or complete validation scenes."""

    def __init__(self, manifest: Path, workspace_root: Path, normalization_file: Path,
                 split: str, kind: str, size: int, augment: bool = False,
                 augmentation_seed: int = 0, pair_mode: str = "paired") -> None:
        if kind not in PATH_COLUMNS:
            raise ValueError(f"Unknown dataset kind: {kind}")
        if (kind, split) not in {("patch", "train"), ("scene", "val"), ("scene", "test")}:
            raise ValueError("Only train patches and full validation/test scenes are permitted")
        if kind == "scene" and augment:
            raise ValueError("Evaluation scenes must not use augmentation")
        if pair_mode not in PAIR_MODES:
            raise ValueError(f"Unknown pair_mode={pair_mode}; expected one of {PAIR_MODES}")
        frame = pd.read_csv(manifest)
        frame = frame.loc[frame["split"].astype(str).eq(split)].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No rows for kind={kind}, split={split}")
        self.frame = frame
        self.workspace_root = workspace_root
        self.normalizer = JointNormalizer(normalization_file)
        self.kind = kind
        self.size = int(size)
        self.augment = bool(augment)
        self.augmentation_seed = int(augmentation_seed)
        self.pair_mode = str(pair_mode)
        self.mismatched_indices = (
            build_mismatched_indices(frame, kind)
            if self.pair_mode == "mismatched_autumn"
            else None
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        epoch = 0
        if isinstance(index, tuple):
            epoch, index = index
        row = self.frame.iloc[index]
        spring_column, autumn_column, mask_column = PATH_COLUMNS[self.kind]
        autumn_row = (
            self.frame.iloc[int(self.mismatched_indices[index])]
            if self.mismatched_indices is not None
            else row
        )
        spring_path = _resolve(self.workspace_root, str(row[spring_column]))
        if self.pair_mode == "spring_duplicate":
            autumn_path = spring_path
        else:
            autumn_path = _resolve(self.workspace_root, str(autumn_row[autumn_column]))
        mask_path = _resolve(self.workspace_root, str(row[mask_column]))
        spring_raw, autumn_raw, mask, valid = _read_triplet(
            (spring_path, autumn_path, mask_path), self.size,
            require_spatial_alignment=self.pair_mode != "mismatched_autumn",
        )
        guidance = spectral_temporal_guidance(spring_raw, autumn_raw, valid)
        items = [
            torch.from_numpy(self.normalizer(spring_raw)),
            torch.from_numpy(self.normalizer(autumn_raw)),
            torch.from_numpy(guidance),
            torch.from_numpy(mask.astype(np.float32)),
            torch.from_numpy(valid.astype(np.bool_)),
        ]
        if self.augment:
            # The transform is a pure function of seed/epoch/index. Persistent
            # workers and checkpoint resume therefore produce the same sample.
            rng = random.Random(self.augmentation_seed + int(epoch) * len(self.frame) + int(index))
            items = _augment(items, rng)
        identifier = str(row["patch_id"] if self.kind == "patch" else row["scene_id"])
        autumn_identifier = str(
            autumn_row["parent_scene_id" if self.kind == "patch" else "scene_id"]
        )
        return {
            "spring": items[0], "autumn": items[1], "guidance": items[2],
            "mask": items[3], "valid_mask": items[4], "sample_id": identifier,
            "autumn_sample_id": autumn_identifier, "pair_mode": self.pair_mode,
        }


def manifest_pos_weight(frame: pd.DataFrame, power: float, maximum: float) -> tuple[float, dict[str, float]]:
    """Derive a transparent, moderate class weight from the train manifest."""
    positive = float(frame["positive_pixel_count"].sum())
    total = float(frame["total_pixel_count"].sum())
    negative = total - positive
    if positive <= 0 or negative <= 0:
        raise ValueError(f"Invalid train class counts: positive={positive}, negative={negative}")
    weight = min(float(maximum), (negative / positive) ** float(power))
    return weight, {
        "manifest_positive_pixels": positive,
        "manifest_negative_pixels": negative,
        "manifest_positive_ratio": positive / total,
        "pos_weight_power": float(power),
        "effective_pos_weight": weight,
    }
