"""Losses on the impulse response in the time domain."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._targets import _CachedTarget
from .base import ResponseLoss

if TYPE_CHECKING:
    import torch

    from pyFDN.train.response import Response


class MatchImpulseResponse(ResponseLoss):
    """Mean squared error against a reference impulse response, sample by sample.

    The strictest of the matching losses -- it fits phase as well as magnitude,
    which for a reverberator is usually more than you want. Reach for
    :class:`~pyFDN.MatchSpectrogram` unless you are fitting an early part or a
    short filter.

    Parameters
    ----------
    target : array_like
        Reference IR, shape ``(n_samples,)``, ``(n_samples, n_out)`` or
        ``(n_samples, n_out, n_in)``. Zero-padded or truncated to the model's
        ``nfft``.
    """

    def __init__(self, target: Any) -> None:
        self._target = _CachedTarget(target)

    def __call__(self, response: Response) -> torch.Tensor:
        import torch

        return torch.nn.functional.mse_loss(response.h, self._target(response))


class Energy(ResponseLoss):
    """Squared deviation of the response's total energy from ``target``.

    A blunt level anchor: useful next to a magnitude-only loss to stop an
    objective that is invariant to overall gain from drifting.
    """

    def __init__(self, target: float = 1.0) -> None:
        self.target = float(target)

    def __call__(self, response: Response) -> torch.Tensor:
        energy = (response.h**2).sum()
        return (energy - self.target) ** 2
