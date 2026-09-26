from __future__ import annotations

from abc import ABC, abstractmethod

import torch


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


class FlatnessRatioDistance(Distance):
    r"""Squared error of the spectral flatness, per channel.

    The flatness of a magnitude spectrum :math:`|H|` over its frequency axis
    (dim 0) is the ratio of its geometric to its arithmetic mean,

    .. math:: F = \frac{\exp\big(\overline{\log |H|}\big)}{\overline{|H|}},

    1 for a perfectly flat spectrum and towards 0 for a peaky one. A silent
    spectrum (arithmetic mean at the numerical floor) has flatness 0, so it
    is never mistaken for a flat one.

    Parameters
    ----------
    target_flatness : float or None
        Fit the prediction's flatness to this constant, in ``[0, 1]``, and
        ignore the reference feature. ``None`` (default) compares against the
        flatness of the reference feature instead.
    """

    def __init__(self, target_flatness: float | None = None) -> None:
        if target_flatness is not None and not 0.0 <= target_flatness <= 1.0:
            raise ValueError(
                f"target_flatness must be in [0, 1]; got {target_flatness}"
            )
        self.target_flatness = (
            None if target_flatness is None else float(target_flatness)
        )

    @staticmethod
    def flatness(magnitude: torch.Tensor) -> torch.Tensor:
        """Geometric over arithmetic mean along dim 0, 0 for a silent spectrum."""
        tiny = torch.finfo(magnitude.dtype).tiny
        geometric = torch.exp(
            torch.log(magnitude.clamp_min(tiny)).mean(dim=0, keepdim=True)
        )
        arithmetic = magnitude.mean(dim=0, keepdim=True)
        silent = arithmetic <= tiny
        ratio = geometric / arithmetic.clamp_min(tiny)
        return torch.where(silent, torch.zeros_like(ratio), ratio)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flatness = self.flatness(pred)
        if self.target_flatness is not None:
            return (pred_flatness - self.target_flatness) ** 2
        return (pred_flatness - self.flatness(target)) ** 2


class ConstantTargetDistance(Distance):
    """Squared error of ``pred`` against a constant (ignores the reference).

    For the targetless losses that fit to a constant rather than a reference
    IR, e.g. :class:`~pyFDN.FlatMagnitude` (``|H| = target``) and
    :class:`~pyFDN.FlatSpectrogram` (normalized spectra equal to 1).
    """

    def __init__(self, target_value: float) -> None:
        self.target_value = float(target_value)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - self.target_value) ** 2


class MaskedSquaredError(Distance):
    """Squared error over the entries where the reference is above ``floor_db``.

    The mask follows the reference, so only the part of the curve carrying
    signal counts; entries below the floor (noise) are ignored. Returns the
    masked entries only, as a 1-D tensor, so any plain reduction (e.g.
    :class:`~pyFDN.train.losses.reductions.Rms`) averages over them alone.
    """

    def __init__(self, floor_db: float = -45.0) -> None:
        self.floor_db = float(floor_db)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target > self.floor_db
        if not bool(mask.any()):
            raise ValueError(
                f"the reference never rises above floor_db={self.floor_db}; "
                "it carries no decay to fit"
            )
        return (pred[mask] - target[mask]) ** 2


class CompressedEnergyDistance(Distance):
    """Reference-normalized, compressed squared error per cumulation direction.

    Each ``(direction, channel, freq, frame)`` surface is normalized by the
    reference's total energy (one constant per direction, so relative levels
    between paths survive) and compressed with ``power`` above a ``floor_db``
    floor before comparing.
    """

    def __init__(self, power: float = 0.5, floor_db: float = -100.0) -> None:
        self.power = float(power)
        if not 0.0 < self.power <= 1.0:
            raise ValueError(
                f"power must be in (0, 1]; got {self.power} (1 is no compression, "
                "smaller compresses harder)"
            )
        self.floor_db = float(floor_db)

    def _compressed(self, surface: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        floor = 10.0 ** (self.floor_db / 10.0)
        return (surface / scale).clamp_min(floor) ** self.power

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scales = [
            d.amax(dim=(-2, -1)).mean().clamp_min(torch.finfo(d.dtype).tiny)
            for d in target
        ]
        if not all(bool(s > 0) for s in scales):
            raise ValueError("the reference carries no energy to fit")
        compressed = torch.stack(
            [
                self._compressed(p, s) - self._compressed(t, s)
                for p, t, s in zip(pred, target, scales, strict=True)
            ]
        )
        return compressed**2
