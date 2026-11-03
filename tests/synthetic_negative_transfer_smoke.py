"""One-step CPU smoke test for all retained single-season and dual-season models."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from models import (
    AutumnOnlyNet,
    SeasonalAuxiliaryControlNet,
    SeasonalConcatNet,
    SpringOnlyNet,
)
from training import MaskedBCEDiceLoss


class TinyEncoder(nn.Module):
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
        output = []
        for layer in self.layers:
            image = layer(image)
            output.append(image)
        return output


def main() -> None:
    torch.manual_seed(42)
    criterion = MaskedBCEDiceLoss(1.0)
    spring = torch.randn(2, 6, 32, 32)
    autumn = torch.randn_like(spring)
    guidance = torch.randn(2, 9, 32, 32)
    target = torch.randint(0, 2, (2, 1, 32, 32)).float()
    valid = torch.ones_like(target, dtype=torch.bool)
    results = {}
    for model_class in (SpringOnlyNet, AutumnOnlyNet, SeasonalConcatNet, SeasonalAuxiliaryControlNet):
        with patch("models.fusion_model.SharedMiTB1", TinyEncoder):
            model = model_class(
                Path("unused"), encoder_channels=TinyEncoder.channels,
                decoder_channels=8, fusion_dropout=0.0,
            ).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        logits, details = model(spring, autumn, guidance, return_details=True)
        loss = criterion(logits, target, valid)
        if "auxiliary" in details:
            aux = details["auxiliary"]
            loss = loss + 0.1 * 0.5 * (
                criterion(aux["spring_logits"], target, valid)
                + criterion(aux["autumn_logits"], target, valid)
            )
        loss.backward()
        gradients = [
            parameter.grad for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if tuple(logits.shape) != tuple(target.shape):
            raise RuntimeError(f"Unexpected output shape: {tuple(logits.shape)}")
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise RuntimeError(f"{model_class.__name__} produced invalid gradients")
        optimizer.step()
        results[model_class.__name__] = {
            "loss": float(loss.detach()),
            "finite_gradient_tensors": len(gradients),
        }
    print(results)


if __name__ == "__main__":
    main()
