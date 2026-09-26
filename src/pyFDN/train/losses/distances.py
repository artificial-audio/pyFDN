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


class AsymmetricPowerDistance(Distance):
    r"""Asymmetric penalty: peaks raised to ``peak_power``, dips always squared.

    For flatness measures, peaks ring audibly while dips are inaudible, so peaks
    are penalized harder. Expects magnitude spectra (non-negative values).

    Parameters
    ----------
    peak_power : float
        Exponent for positive deviations (peaks). Must be >= 2. Dips always use power 2.
    """

    def __init__(self, peak_power: float = 4.0) -> None:
        self.peak_power = float(peak_power)
        if self.peak_power < 2.0:
            raise ValueError(f"peak_power must be at least 2; got {self.peak_power}")

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        rms = (target**2).mean(dim=0, keepdim=True).sqrt()
        eps = torch.finfo(target.dtype).tiny
        normalized = pred / rms.clamp_min(eps)
        deviation = normalized - 1.0

        peaks = deviation.clamp_min(0.0) ** self.peak_power
        dips = deviation.clamp_max(0.0) ** 2
        return peaks + dips


class FlatnessRatioDistance(Distance):
    r"""Spectral flatness: geometric mean / arithmetic mean per frequency.

    Used to measure flatness of a spectrum. Returns MSE between the target
    flatness (default 1.0) and the pred flatness, which is the ratio of
    geometric to arithmetic mean normalized by target's ratio.

    This distance assumes pred and target are both magnitude spectra.
    """

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(pred.dtype).tiny
        pred_geom = torch.exp(torch.log(pred.clamp_min(eps)).mean(dim=0, keepdim=True))
        pred_arith = pred.mean(dim=0, keepdim=True).clamp_min(eps)
        pred_flatness = pred_geom / pred_arith

        target_geom = torch.exp(
            torch.log(target.clamp_min(eps)).mean(dim=0, keepdim=True)
        )
        target_arith = target.mean(dim=0, keepdim=True).clamp_min(eps)
        target_flatness = target_geom / target_arith

        return (pred_flatness - target_flatness) ** 2


class ConstantTargetDistance(Distance):
    """Compare pred to a fixed constant target value (ignores ref_feat).

    Useful for losses like FlatMagnitude that fit to a constant rather than
    a reference IR. The target parameter in Match is ignored; instead pred is
    compared to the constant value provided at construction.
    """

    def __init__(self, target_value: float) -> None:
        self.target_value = float(target_value)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - self.target_value) ** 2


class OnesTargetDistance(Distance):
    """Compare pred to a tensor of all ones (ignores ref_feat).

    Useful for losses like FlatSpectrogram that fit normalized spectra to flat
    (where flat = ones after normalization).
    """

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - 1.0) ** 2


class MaskedSquaredError(Distance):
    """Squared error over entries where the reference is above ``floor_db``.

    The mask follows the reference, so only the part of the curve carrying
    signal counts; entries below the floor (noise) are ignored.
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
        out = torch.zeros_like(pred)
        out[mask] = (pred[mask] - target[mask]) ** 2
        # Stash the count so Rms reads only the masked entries.
        self._count = mask.sum()
        return out


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
