"""Costs on a model parameter, rather than on its response.

Each takes a :class:`~pyFDN.train.params.ParamRef` naming the parameter it acts
on, so it works on any FDN structure::

    A = pyFDN.param(model, "feedback")
    loss = pyFDN.FlatMagnitude() + 0.2 * pyFDN.Sparsity(A)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import ParameterLoss

if TYPE_CHECKING:
    import torch

    from pyFDN.train.params import ParamRef


class Sparsity(ParameterLoss):
    """Density penalty on a square mixing matrix, after *Optimizing Tiny
    Colorless FDNs* (Dal Santo et al.).

    Rewards a *dense* matrix -- 0 when :math:`|A|` is maximally dense (its
    entries all :math:`1/\\sqrt{N}`, i.e. best mixing) and 1 when fully sparse.
    Registered with a positive weight it therefore pushes the feedback matrix
    away from the sparse, poorly-mixing corners of SO(N) that a magnitude-only
    objective is otherwise happy to sit in.

    Parameters
    ----------
    ref : ParamRef
        The matrix to penalize, e.g. ``pyFDN.param(model, "feedback")``. Must be
        square.
    """

    def __init__(self, ref: ParamRef) -> None:
        super().__init__(ref)
        shape = ref.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(
                f"Sparsity needs a square matrix; {ref.name!r} has shape {shape}"
            )

    def penalty(self, value: torch.Tensor) -> torch.Tensor:
        n = value.shape[-1]
        root_n = float(np.sqrt(n))
        return -(value.abs().sum() - n * root_n) / (n * (root_n - 1.0))


class L1(ParameterLoss):
    """Mean absolute value of a parameter -- pushes it toward sparse."""

    def penalty(self, value: torch.Tensor) -> torch.Tensor:
        return value.abs().mean()


class L2(ParameterLoss):
    """Mean squared value of a parameter -- plain weight decay."""

    def penalty(self, value: torch.Tensor) -> torch.Tensor:
        return (value**2).mean()
