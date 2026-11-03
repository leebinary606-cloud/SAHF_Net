"""Masked BCE plus positive-aware Dice for binary PV segmentation."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MaskedBCEDiceLoss(nn.Module):
    """Input logits [B, 1, H, W], target [B, 1, H, W], valid_mask [B, 1, H, W]; output scalar loss."""

    def __init__(self, pos_weight: float, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        target = target.float()
        valid = valid_mask.bool().expand_as(target)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Model logits contain non-finite values")
        if not valid.any():
            raise ValueError("Batch contains no valid pixels")
        bce_map = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none", pos_weight=self.pos_weight
        )
        bce = bce_map[valid].mean()

        probability = torch.sigmoid(logits)
        valid_float = valid.float()
        intersection = (probability * target * valid_float).sum(dim=(1, 2, 3))
        probability_sum = (probability * valid_float).sum(dim=(1, 2, 3))
        target_sum = (target * valid_float).sum(dim=(1, 2, 3))
        has_positive = target_sum > 0
        if has_positive.any():
            dice = 1.0 - ((2.0 * intersection[has_positive] + 1.0) /
                          (probability_sum[has_positive] + target_sum[has_positive] + 1.0)).mean()
        else:
            dice = logits.new_zeros(())
        result = self.bce_weight * bce + self.dice_weight * dice
        if not torch.isfinite(result):
            raise FloatingPointError("Masked BCE+Dice loss is non-finite")
        return result
