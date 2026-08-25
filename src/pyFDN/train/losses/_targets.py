"""Shared handling of a reference impulse response held by a loss."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

    from pyFDN.train.response import Response


def align_target(target: Any, response: Response) -> torch.Tensor:
    """A reference IR as a tensor shaped exactly like ``response.h``.

    Accepts ``(n_samples,)``, ``(n_samples, n_out)`` or the full
    ``(n_samples, n_out, n_in)`` IR matrix (:func:`pyFDN.build_to_impz`'s
    shape), and zero-pads or truncates it in time to the response length. The
    channel counts must match the model's -- a mismatch is a mistake worth
    raising rather than broadcasting away.
    """
    import torch

    arr = np.asarray(target, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None, None]
    elif arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim != 3:
        raise ValueError(
            "target must be 1-, 2- or 3-D (n_samples[, n_out[, n_in]]); got "
            f"shape {arr.shape}"
        )
    if arr.shape[1:] != (response.n_out, response.n_in):
        raise ValueError(
            f"target has {arr.shape[1]} outputs and {arr.shape[2]} inputs, but "
            f"the model has n_out={response.n_out}, n_in={response.n_in}"
        )

    aligned = np.zeros((response.n_samples, response.n_out, response.n_in))
    length = min(arr.shape[0], response.n_samples)
    aligned[:length] = arr[:length]
    return torch.as_tensor(aligned, device=response.h.device, dtype=response.h.dtype)


class _CachedTarget:
    """Aligns a target once, on the first response it sees.

    The alignment needs the response's length, device and dtype, which are only
    known at the first step -- but they do not change after it, so the loss
    holds the raw target and converts lazily.
    """

    def __init__(self, target: Any) -> None:
        self._raw = target
        self._aligned: torch.Tensor | None = None

    def __call__(self, response: Response) -> torch.Tensor:
        if self._aligned is None:
            self._aligned = align_target(self._raw, response)
        return self._aligned
