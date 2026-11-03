"""Local MiT-B1 wrapper with an explicit six-band Sentinel-2 adaptation."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import SegformerConfig, SegformerModel


CHANNELS = (64, 128, 320, 512)


def adapt_rgb_projection(weight: torch.Tensor) -> torch.Tensor:
    """Map RGB [C, 3, 7, 7] weights to 6-band B2,B3,B4,B8,B11,B12 [C, 6, 7, 7]."""
    if tuple(weight.shape[1:]) != (3, 7, 7):
        raise ValueError(f"Expected RGB 7x7 projection, got {tuple(weight.shape)}")
    spectral_mean = weight.mean(dim=1, keepdim=True)
    expanded = torch.cat(
        (weight[:, 2:3], weight[:, 1:2], weight[:, 0:1], spectral_mean, spectral_mean, spectral_mean),
        dim=1,
    )
    return expanded * (3.0 / 6.0) ** 0.5


class SharedMiTB1(nn.Module):
    """Input [B, 6, H, W]; output 4-stage features ([B, 64, H/4, W/4] .. [B, 512, H/32, W/32])."""

    channels = CHANNELS

    def __init__(self, pretrained_dir: Path, drop_path_rate: float = 0.1) -> None:
        super().__init__()
        config = SegformerConfig.from_pretrained(str(pretrained_dir), local_files_only=True)
        if tuple(config.hidden_sizes) != CHANNELS:
            raise ValueError(f"Local weights are not MiT-B1: {config.hidden_sizes}")
        config.num_channels = 6
        config.output_hidden_states = True
        config.drop_path_rate = float(drop_path_rate)
        self.model = SegformerModel(config)

        official = SegformerModel.from_pretrained(str(pretrained_dir), local_files_only=True)
        state = official.state_dict()
        candidates = [
            key for key, value in state.items()
            if value.ndim == 4 and tuple(value.shape[1:]) == (3, 7, 7) and value.shape[0] == CHANNELS[0]
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Cannot identify official MiT-B1 RGB projection: {candidates}")
        state[candidates[0]] = adapt_rgb_projection(state[candidates[0]])
        self.model.load_state_dict(state, strict=True)
        self.adapted_projection_key = candidates[0]

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        output = self.model(pixel_values=image, output_hidden_states=True, return_dict=True)
        features = list(output.hidden_states or ())
        if len(features) != 4:
            raise RuntimeError(f"MiT-B1 returned {len(features)} stages, expected 4")
        return features
