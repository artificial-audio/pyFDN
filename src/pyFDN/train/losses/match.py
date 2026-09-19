from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch

from .base import ResponseLoss
from ._targets import _CachedTarget
from .reductions import Reduction, Mean
from .distances import Distance, SquaredError
from .features import Feature

if TYPE_CHECKING:
    from pyFDN.train.response import Response


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
        self._target = _CachedTarget(target) if target is not None else None
        self.feature = feature
        self.distance = distance
        self.reduction = reduction

    def __call__(self, response: Response) -> torch.Tensor:
        pred_feat = self.feature(response.h, response.fs)
        if self._target is not None:
            ref_feat = self.feature(self._target(response), response.fs)
            diff = self.distance(pred_feat, ref_feat)
        else:
            diff = self.distance(pred_feat, torch.zeros_like(pred_feat))
        return self.reduction(diff)
