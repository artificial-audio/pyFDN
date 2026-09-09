from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

import torch

from ._targets import _CachedTarget
from .base import ResponseLoss

if TYPE_CHECKING:
    from pyFDN.train.response import Response


# --- Reductions ---

class Reduction(ABC):
    """Base class for aggregation strategies."""

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class Mean(Reduction):
    """Arithmetic mean reduction."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean()


class Sum(Reduction):
    """Sum reduction."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


# --- Distances ---

class Distance(ABC):
    """Base class for pairwise element comparison metrics."""

    @abstractmethod
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...


class SquaredError(Distance):
    """L2: Elementwise squared difference: (pred - target)^2."""

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target) ** 2


class AbsoluteError(Distance):
    """L1: Elementwise absolute difference: |pred - target|."""

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).abs()


class CircularDistance(Distance):
    r"""Circular angular metric: 1 - cos(pred - target) bounded in [0, 2]."""

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.cos(pred - target)


# --- Features ---

class Feature(ABC):
    """Base class for extracting a representation from an impulse response."""

    @abstractmethod
    def __call__(self, h: torch.Tensor, fs: float) -> torch.Tensor: ...


class Waveform(Feature):
    """Raw time-domain waveform feature."""

    def __call__(self, h: torch.Tensor, fs: float) -> torch.Tensor:
        return h


class Magnitude(Feature):
    """Frequency-domain magnitude spectrum via rfft."""

    def __init__(self, channels: str = "none") -> None:
        if channels not in ("sum", "mean", "none"):
            raise ValueError(f"channels must be 'sum', 'mean', or 'none'; got {channels!r}")
        self.channels = channels

    def __call__(self, h: torch.Tensor, fs: float) -> torch.Tensor:
        mag = torch.fft.rfft(h, dim=0).abs()
        if self.channels == "sum":
            return mag.sum(dim=1)
        if self.channels == "mean":
            return mag.mean(dim=1)
        return mag


class Phase(Feature):
    """Frequency-domain phase spectrum via rfft."""

    def __init__(self, channels: str = "none") -> None:
        if channels not in ("sum", "mean", "none"):
            raise ValueError(f"channels must be 'sum', 'mean', or 'none'; got {channels!r}")
        self.channels = channels

    def __call__(self, h: torch.Tensor, fs: float) -> torch.Tensor:
        phase = torch.angle(torch.fft.rfft(h, dim=0))
        if self.channels == "sum":
            return phase.sum(dim=1)
        if self.channels == "mean":
            return phase.mean(dim=1)
        return phase


# --- Composed Match class ---

class Match(ResponseLoss):
    r"""Compares a model's impulse response to a target using three steps:

        1. **Feature**: Extracts representations from both the model response and the
        target (e.g., raw waveform, magnitude spectrum, or phase).
        2. **Distance**: Computes the differences between the extracted
        features (e.g., squared error or circular phase difference).
        3. **Reduction**: Aggregates the elementwise differences into a scalar loss
        tensor (e.g., mean or sum).

        The reference target is aligned in length, shape, device, and dtype 
        using ``_CachedTarget``.

        Parameters
        ----------
        target : Any
            Reference impulse response (NumPy array, PyTorch tensor, or list).
        feature : Feature or callable
            Extracts the representation to compare: ``(h, fs) -> torch.Tensor``.
        distance : Distance or callable, default SquaredError()
            Calculates error between two features: ``(pred, target) -> torch.Tensor``.
        reduction : Reduction or callable, default Mean()
            Turns the error tensor into a single scalar: ``(diff) -> torch.Tensor``.
    """

    def __init__(
        self,
        target: Any,
        feature: Feature | Callable[[torch.Tensor, float], torch.Tensor],
        distance: Distance | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = SquaredError(),
        reduction: Reduction | Callable[[torch.Tensor], torch.Tensor] = Mean(),
    ) -> None:
        self._target = _CachedTarget(target)
        self.feature = feature
        self.distance = distance
        self.reduction = reduction

    def __call__(self, response: Response) -> torch.Tensor:
        ref_h = self._target(response)
        pred_feat = self.feature(response.h, response.fs)
        ref_feat = self.feature(ref_h, response.fs)

        diff = self.distance(pred_feat, ref_feat)
        return self.reduction(diff)