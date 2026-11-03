from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from rasterio.transform import from_origin
from torch.utils.data import DataLoader

from data.dataset import PairedSentinelDataset, build_mismatched_indices
from models.fusion_model import (
    AutumnOnlyNet,
    SeasonalAuxiliaryControlNet,
    SeasonalConcatNet,
    SpringOnlyNet,
)
from main import selected_training_config
from training.engine import Trainer, build_optimizer, build_scheduler, compare_scene_tables
from training.loss import MaskedBCEDiceLoss


class DummyEncoder(nn.Module):
    channels = (8, 16, 32, 64)

    def __init__(self, _pretrained_dir: Path, _drop_path_rate: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(6, 8, 3, stride=4, padding=1),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
        ])

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for layer in self.layers:
            image = layer(image)
            features.append(image)
        return features


def write_raster(path: Path, value: float | np.ndarray, count: int, transform) -> None:
    if isinstance(value, np.ndarray):
        array = value.astype(np.float32)
    else:
        array = np.full((count, 16, 16), value, dtype=np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", width=array.shape[-1], height=array.shape[-2], count=count,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as target:
        target.write(array)


class PairModeTests(unittest.TestCase):
    def test_mismatch_is_deterministic_and_never_uses_same_scene(self) -> None:
        frame = pd.DataFrame({
            "parent_scene_id": ["A", "A", "B", "B", "C", "C"],
        })
        first = build_mismatched_indices(frame, "patch")
        second = build_mismatched_indices(frame, "patch")
        self.assertTrue(np.array_equal(first, second))
        identifiers = frame["parent_scene_id"].to_numpy()
        self.assertTrue(np.all(identifiers[first] != identifiers))

    def test_duplicate_and_mismatched_autumn_change_only_auxiliary_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index, (spring_value, autumn_value) in enumerate(((1.0, 2.0), (10.0, 20.0))):
                scene_id = f"S{index + 1}"
                transform = from_origin(100.0 + index, 35.0, 0.0001, 0.0001)
                spring = root / f"{scene_id}_spring.tif"
                autumn = root / f"{scene_id}_autumn.tif"
                mask = root / f"{scene_id}_mask.tif"
                write_raster(spring, spring_value, 6, transform)
                write_raster(autumn, autumn_value, 6, transform)
                write_raster(mask, 1.0, 1, transform)
                rows.append({
                    "scene_id": scene_id, "split": "val",
                    "spring_path": spring.name, "autumn_path": autumn.name,
                    "scene_mask_path": mask.name,
                })
            manifest = root / "scenes.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            normalization = root / "normalization.json"
            normalization.write_text(json.dumps({
                "joint": {"mean": [0.0] * 6, "std": [1.0] * 6}
            }), encoding="utf-8")

            duplicate = PairedSentinelDataset(
                manifest, root, normalization, "val", "scene", 16,
                pair_mode="spring_duplicate",
            )[0]
            mismatched = PairedSentinelDataset(
                manifest, root, normalization, "val", "scene", 16,
                pair_mode="mismatched_autumn",
            )[0]

            self.assertTrue(torch.equal(duplicate["spring"], duplicate["autumn"]))
            self.assertEqual(mismatched["sample_id"], "S1")
            self.assertEqual(mismatched["autumn_sample_id"], "S2")
            self.assertTrue(torch.all(mismatched["spring"] == 1.0))
            self.assertTrue(torch.all(mismatched["autumn"] == 20.0))
            self.assertTrue(torch.all(mismatched["mask"] == 1.0))


class SingleSeasonAndIntegratedAnalysisTests(unittest.TestCase):
    @staticmethod
    def build(model_class):
        with patch("models.fusion_model.SharedMiTB1", DummyEncoder):
            return model_class(
                Path("unused"), encoder_channels=DummyEncoder.channels,
                decoder_channels=8, fusion_dropout=0.0,
            ).eval()

    def test_spring_control_ignores_autumn(self) -> None:
        model = self.build(SpringOnlyNet)
        spring = torch.randn(2, 6, 32, 32)
        guidance = torch.randn(2, 9, 32, 32)
        with torch.no_grad():
            first = model(spring, torch.zeros_like(spring), guidance)
            second = model(spring, torch.randn_like(spring), guidance)
        self.assertTrue(torch.equal(first, second))

    def test_autumn_control_ignores_spring(self) -> None:
        model = self.build(AutumnOnlyNet)
        autumn = torch.randn(2, 6, 32, 32)
        guidance = torch.randn(2, 9, 32, 32)
        with torch.no_grad():
            first = model(torch.zeros_like(autumn), autumn, guidance)
            second = model(torch.randn_like(autumn), autumn, guidance)
        self.assertTrue(torch.equal(first, second))

    def test_control_objectives_do_not_inherit_auxiliary_losses(self) -> None:
        raw = {
            "seed": 42,
            "auxiliary_loss_weight": 0.10,
        }
        for variant in ("spring_only", "autumn_only", "concat"):
            selected = selected_training_config(raw, variant, seed_override=77)
            self.assertEqual(selected["auxiliary_loss_weight"], 0.0)
            self.assertEqual(selected["seed"], 77)
        aux_selected = selected_training_config(raw, "aux_only")
        self.assertEqual(aux_selected["auxiliary_loss_weight"], 0.10)

    def test_trainer_writes_integrated_validation_and_comparison_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for idx, prov in enumerate(("AH", "NX")):
                scene_id = f"{prov}_000{idx + 1}"
                transform = from_origin(116.0 + idx * 0.1, 34.0, 0.0001, 0.0001)
                spring = root / f"{scene_id}_spring.tif"
                autumn = root / f"{scene_id}_autumn.tif"
                mask = root / f"{scene_id}_mask.tif"
                write_raster(spring, 1.0 + idx, 6, transform)
                write_raster(autumn, 2.0 + idx, 6, transform)
                mask_arr = np.zeros((1, 16, 16), dtype=np.float32)
                mask_arr[:, : (4 if idx == 0 else 8), :8] = 1.0
                write_raster(mask, mask_arr, 1, transform)
                rows.append({
                    "scene_id": scene_id,
                    "parent_scene_id": scene_id,
                    "patch_id": f"{scene_id}_P0",
                    "province": prov,
                    "spatial_group_id": f"G_{idx}",
                    "split": "val",
                    "spring_path": spring.name,
                    "autumn_path": autumn.name,
                    "scene_mask_path": mask.name,
                    "spring_patch_path": spring.name,
                    "autumn_patch_path": autumn.name,
                    "mask_patch_path": mask.name,
                    "crs": "EPSG:4326",
                    "height": 16,
                    "transform_a": transform.a,
                    "transform_b": transform.b,
                    "transform_c": transform.c,
                    "transform_d": transform.d,
                    "transform_e": transform.e,
                    "transform_f": transform.f,
                })
            # Duplicate rows as train patches
            train_rows = [dict(r, split="train") for r in rows]
            manifest = root / "manifest.csv"
            pd.DataFrame(rows + train_rows).to_csv(manifest, index=False)
            norm = root / "norm.json"
            norm.write_text(json.dumps({"joint": {"mean": [0.0] * 6, "std": [1.0] * 6}}), encoding="utf-8")

            train_ds = PairedSentinelDataset(manifest, root, norm, "train", "patch", 16)
            val_ds = PairedSentinelDataset(manifest, root, norm, "val", "scene", 16)
            train_loader = DataLoader(train_ds, batch_size=2)
            val_loader = DataLoader(val_ds, batch_size=2)

            cfg = {
                "validation_threshold": 0.5,
                "gradient_clip_norm": 1.0,
                "early_stopping_patience": 5,
                "auxiliary_loss_weight": 0.0,
                "amp": False,
                "amp_dtype": "bfloat16",
                "encoder_learning_rate": 1e-4,
                "new_module_learning_rate": 2e-4,
                "weight_decay": 0.01,
                "scheduler": "warmup_reduce_on_plateau",
                "plateau_monitor": "val_loss",
                "warmup_epochs": 1,
                "min_encoder_learning_rate": 1e-5,
                "min_new_module_learning_rate": 1e-5,
                "plateau_factor": 0.5,
                "plateau_patience": 2,
                "plateau_threshold": 1e-4,
                "calibration_threshold_min": 0.4,
                "calibration_threshold_max": 0.6,
                "calibration_threshold_step": 0.1,
            }

            run1_dir = root / "runs" / "spv2_spring_only_s42"
            run1_dir.mkdir(parents=True)
            model1 = self.build(SpringOnlyNet).train()
            opt1 = build_optimizer(model1, cfg)
            sched1 = build_scheduler(opt1, cfg)
            trainer1 = Trainer(model1, MaskedBCEDiceLoss(1.0), opt1, sched1, torch.device("cpu"), run1_dir, cfg, "sig1")
            summary1 = trainer1.fit(train_loader, val_loader, epochs=1)
            self.assertTrue((run1_dir / "diagnostics" / "val_per_scene.csv").is_file())
            self.assertTrue((run1_dir / "diagnostics" / "val_per_province.csv").is_file())
            self.assertTrue((run1_dir / "diagnostics" / "val_by_scale.csv").is_file())
            self.assertTrue((run1_dir / "diagnostics" / "val_analysis_report.json").is_file())
            self.assertIn("AH", summary1["validation_analysis"]["by_province"])
            self.assertIn("NX", summary1["validation_analysis"]["by_province"])

            # Second run auto-compares against spv2_spring_only_s42 in the same run_root
            run2_dir = root / "runs" / "spv2_concat_paired_s42"
            run2_dir.mkdir(parents=True)
            model2 = self.build(SeasonalConcatNet).train()
            opt2 = build_optimizer(model2, cfg)
            sched2 = build_scheduler(opt2, cfg)
            trainer2 = Trainer(model2, MaskedBCEDiceLoss(1.0), opt2, sched2, torch.device("cpu"), run2_dir, cfg, "sig2")
            summary2 = trainer2.fit(train_loader, val_loader, epochs=1)
            comp = summary2["validation_analysis"]["comparison"]
            self.assertIsNotNone(comp)
            self.assertEqual(comp["reference_run"], "spv2_spring_only_s42")
            self.assertTrue((run2_dir / "diagnostics" / "comparison_vs_spv2_spring_only_s42.json").is_file())


if __name__ == "__main__":
    unittest.main()
