"""Training losses for FDNs -- see :mod:`pyFDN.train.losses.base` for how they
compose, and :mod:`pyFDN.train.presets` for ready-made combinations.
"""

from __future__ import annotations

from .base import Loss, ParameterLoss, ResponseLoss, Scaled, Sum
from .parameter import L1, L2, Sparsity
from .spectral import (
    AsymmetricFlatMagnitude,
    FlatMagnitude,
    FlatSpectrogram,
    MatchMagnitude,
    MatchMelSpectrogram,
    MatchSpectrogram,
)
from .temporal import (
    Energy,
    MatchCumulativeEnergy,
    MatchEnergyDecay,
    MatchImpulseResponse,
)

__all__ = [
    # composition
    "Loss",
    "ResponseLoss",
    "ParameterLoss",
    "Sum",
    "Scaled",
    # response losses
    "FlatMagnitude",
    "AsymmetricFlatMagnitude",
    "FlatSpectrogram",
    "MatchMagnitude",
    "MatchSpectrogram",
    "MatchMelSpectrogram",
    "MatchImpulseResponse",
    "MatchEnergyDecay",
    "MatchCumulativeEnergy",
    "Energy",
    # parameter losses
    "Sparsity",
    "L1",
    "L2",
]
