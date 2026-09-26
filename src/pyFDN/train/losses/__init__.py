"""Training losses for FDNs -- see :mod:`pyFDN.train.losses.base` for how they
compose into an objective.
"""

from __future__ import annotations

from .base import Loss, ParameterLoss, ResponseLoss, Scaled, Sum
from .distances import CircularDistance, SquaredError
from .features import Magnitude, Phase, Waveform
from .match import Match
from .parameter import L1, L2, Sparsity
from .reductions import Mean
from .spectral import (
    AsymmetricFlatMagnitude,
    FlatMagnitude,
    FlatSpectrogram,
    MatchMagnitude,
    MatchMelMagnitude,
    MatchMelSpectrogram,
    MatchPhase,
    MatchPhaseSpectrogram,
    MatchSpectrogram,
    SpectralFlatness,
)
from .temporal import (
    Energy,
    MatchCumulativeEnergy,
    MatchEnergyDecay,
    MatchImpulseResponse,
)

__all__ = [
    # match (the Sum reduction stays in .reductions: ``Sum`` here is the
    # loss composition from .base)
    "Match",
    "Mean",
    "SquaredError",
    "CircularDistance",
    "Waveform",
    "Magnitude",
    "Phase",
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
    "SpectralFlatness",
    "MatchMagnitude",
    "MatchPhase",
    "MatchPhaseSpectrogram",
    "MatchMelMagnitude",
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
