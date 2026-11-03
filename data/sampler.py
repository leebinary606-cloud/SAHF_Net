"""Epoch-addressable shuffled sampling for exact checkpoint resume."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler


class EpochShuffleSampler(Sampler[tuple[int, int]]):
    """Yield (epoch, index), making order and augmentation restart-independent."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[tuple[int, int]]:
        order = np.random.default_rng(self.seed + self.epoch).permutation(self.size)
        return iter((self.epoch, int(index)) for index in order)
