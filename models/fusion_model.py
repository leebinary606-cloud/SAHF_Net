"""Core single-season and dual-season segmentation architectures."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .backbone import SharedMiTB1


def _groups(channels: int) -> int:
    for value in (32, 16, 8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class ConvNormAct(nn.Module):
    """Input [B, C_in, H, W]; output [B, C_out, H, W]."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1,
                 dropout: float = 0.0) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MatchedStageConcatFusion(nn.Module):
    """Input spring [B, C, H, W] and autumn [B, C, H, W]; output fused [B, C, H, W]."""

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = ConvNormAct(channels * 2, channels, 1, dropout)

    def forward(self, spring: torch.Tensor, autumn: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.cat((spring, autumn), dim=1))


class MultiscaleDecoder(nn.Module):
    """Input 4-stage encoder features and output_size (H, W); output logits [B, 1, H, W]."""

    def __init__(self, encoder_channels: tuple[int, ...], decoder_channels: int) -> None:
        super().__init__()
        self.projects = nn.ModuleList([
            ConvNormAct(channels, decoder_channels, 1) for channels in encoder_channels
        ])
        self.fuse = nn.Sequential(
            ConvNormAct(decoder_channels * len(encoder_channels), decoder_channels, 3, 0.1),
            nn.Conv2d(decoder_channels, 1, 1),
        )

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        projected = []
        for projection, feature in zip(self.projects, features):
            value = projection(feature)
            if value.shape[-2:] != target_size:
                value = F.interpolate(value, target_size, mode="bilinear", align_corners=False)
            projected.append(value)
        logits = self.fuse(torch.cat(projected, dim=1))
        return F.interpolate(logits, output_size, mode="bilinear", align_corners=False)


class SeasonalSingleSeasonNet(nn.Module):
    """Input spring [B, 6, H, W], autumn [B, 6, H, W], guidance [B, 9, H, W]; output logits [B, 1, H, W]."""

    season = "spring"
    architecture_version = "single_season_mit_b1_base"

    def __init__(self, pretrained_dir: Path,
                 encoder_channels: tuple[int, ...] = (64, 128, 320, 512),
                 decoder_channels: int = 128, drop_path_rate: float = 0.1,
                 fusion_dropout: float = 0.1) -> None:
        super().__init__()
        _ = float(fusion_dropout)
        self.shared_encoder = SharedMiTB1(pretrained_dir, drop_path_rate)
        if tuple(self.shared_encoder.channels) != tuple(encoder_channels):
            raise ValueError("Configured encoder channels do not match MiT-B1")
        self.decoder = MultiscaleDecoder(tuple(encoder_channels), decoder_channels)

    def forward(self, spring: torch.Tensor, autumn: torch.Tensor,
                guidance: torch.Tensor, return_details: bool = False):
        _ = guidance
        image = spring if self.season == "spring" else autumn
        logits = self.decoder(self.shared_encoder(image), image.shape[-2:])
        return (logits, {}) if return_details else logits


class SpringOnlyNet(SeasonalSingleSeasonNet):
    """Input spring [B, 6, H, W], autumn [B, 6, H, W], guidance [B, 9, H, W]; output logits [B, 1, H, W]."""

    season = "spring"
    architecture_version = "spring_only_mit_b1_v2"


class AutumnOnlyNet(SeasonalSingleSeasonNet):
    """Input spring [B, 6, H, W], autumn [B, 6, H, W], guidance [B, 9, H, W]; output logits [B, 1, H, W]."""

    season = "autumn"
    architecture_version = "autumn_only_mit_b1_v2"


class SeasonalConcatNet(nn.Module):
    """Input spring [B, 6, H, W], autumn [B, 6, H, W], guidance [B, 9, H, W]; output logits [B, 1, H, W]."""

    architecture_version = "seasonal_concat_control_v2"

    def __init__(self, pretrained_dir: Path, encoder_channels: tuple[int, ...] = (64, 128, 320, 512),
                 decoder_channels: int = 128, drop_path_rate: float = 0.1,
                 fusion_dropout: float = 0.1) -> None:
        super().__init__()
        self.shared_encoder = SharedMiTB1(pretrained_dir, drop_path_rate)
        if tuple(self.shared_encoder.channels) != tuple(encoder_channels):
            raise ValueError("Configured encoder channels do not match MiT-B1")
        channels = tuple(encoder_channels)
        self.fusions = nn.ModuleList([
            MatchedStageConcatFusion(channel, fusion_dropout) for channel in channels
        ])
        self.decoder = MultiscaleDecoder(channels, decoder_channels)

    def forward(self, spring: torch.Tensor, autumn: torch.Tensor, guidance: torch.Tensor,
                return_details: bool = False):
        _ = guidance
        spring_features = self.shared_encoder(spring)
        autumn_features = self.shared_encoder(autumn)
        fused = [
            module(spring_feature, autumn_feature)
            for module, spring_feature, autumn_feature in zip(
                self.fusions, spring_features, autumn_features
            )
        ]
        logits = self.decoder(fused, spring.shape[-2:])
        return (logits, {}) if return_details else logits


class SeasonalAuxiliaryControlNet(nn.Module):
    """Input spring [B, 6, H, W], autumn [B, 6, H, W], guidance [B, 9, H, W]; output logits [B, 1, H, W]."""

    architecture_version = "seasonal_decoupled_auxiliary_control_v2"

    def __init__(self, pretrained_dir: Path,
                 encoder_channels: tuple[int, ...] = (64, 128, 320, 512),
                 decoder_channels: int = 128, drop_path_rate: float = 0.1,
                 fusion_dropout: float = 0.1) -> None:
        super().__init__()
        self.shared_encoder = SharedMiTB1(pretrained_dir, drop_path_rate)
        if tuple(self.shared_encoder.channels) != tuple(encoder_channels):
            raise ValueError("Configured encoder channels do not match MiT-B1")
        channels = tuple(encoder_channels)
        self.fusions = nn.ModuleList([
            MatchedStageConcatFusion(channel, fusion_dropout) for channel in channels
        ])
        self.decoder = MultiscaleDecoder(channels, decoder_channels)
        self.auxiliary_decoder = MultiscaleDecoder(channels, decoder_channels)

    def _features(self, spring: torch.Tensor, autumn: torch.Tensor):
        spring_features = self.shared_encoder(spring)
        autumn_features = self.shared_encoder(autumn)
        fused = [
            module(spring_feature, autumn_feature)
            for module, spring_feature, autumn_feature in zip(
                self.fusions, spring_features, autumn_features
            )
        ]
        return spring_features, autumn_features, fused

    def forward(self, spring: torch.Tensor, autumn: torch.Tensor, guidance: torch.Tensor,
                return_details: bool = False):
        _ = guidance
        spring_features, autumn_features, fused = self._features(spring, autumn)
        output_size = spring.shape[-2:]
        base_logits = self.decoder(fused, output_size)
        if not return_details:
            return base_logits
        return base_logits, {
            "auxiliary": {
                "spring_logits": self.auxiliary_decoder(spring_features, output_size),
                "autumn_logits": self.auxiliary_decoder(autumn_features, output_size),
            }
        }

    def validation_headroom_outputs(self, spring: torch.Tensor, autumn: torch.Tensor,
                                    guidance: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return fixed candidates [B, 1, H, W] for Validation-only oracle audit."""
        _ = guidance
        spring_features, autumn_features, fused = self._features(spring, autumn)
        output_size = spring.shape[-2:]
        base_logits = self.decoder(fused, output_size)
        return {
            "base_logits": base_logits,
            "final_logits": base_logits,
            "spring_logits": self.auxiliary_decoder(spring_features, output_size),
            "autumn_logits": self.auxiliary_decoder(autumn_features, output_size),
        }
