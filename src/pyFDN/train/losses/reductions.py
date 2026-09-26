from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class Reduction(ABC):
    """Base class for aggregation strategies."""

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class Mean(Reduction):
    """Arithmetic mean reduction."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean()


class MeanPerGroup(Reduction):
    """Mean averaged per first-dimension group, then across groups.

    Each group is normalized by its actual element count, giving equal
    weight to each group regardless of its size (unlike :meth:`Mean`
    which weights by total element count).

    Multi-resolution features (:class:`~pyFDN.train.losses.features.SpectrogramFeature`,
    :class:`~pyFDN.train.losses.features.PhaseSpectrogramFeature`) pad every
    resolution to a common shape and record each resolution's number of real
    (unpadded) entries in ``_group_counts`` on every call. Pass that feature
    here so each group is divided by its real count; the padding must then
    contribute zero distance (it does for the distances those features are
    used with). Without such a feature, each group is averaged over its full,
    padded size.
    """

    def __init__(self, feature: Any = None) -> None:
        self.feature = feature

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature is not None and hasattr(self.feature, "_group_counts"):
            counts = torch.tensor(
                self.feature._group_counts, device=x.device, dtype=x.dtype
            )
            group_sums = x.sum(dim=tuple(range(1, x.ndim)))
            return (group_sums / counts).mean()
        return x.mean(dim=tuple(range(1, x.ndim))).mean()


class Sum(Reduction):
    """Sum reduction."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


class Rms(Reduction):
    """Root-mean-square of squared errors: ``sqrt(mean(x))`` over all entries."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean().sqrt()


class MeanRmsOverGroups(Reduction):
    """Mean of per-group RMS values, groups split along dim 0."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        return flat.mean(dim=1).sqrt().mean()
