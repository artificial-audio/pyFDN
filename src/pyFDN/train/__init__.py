"""Training pipeline for FDNs -- an explicit three-step API over flamo.

1. **build** a trainable flamo model from a config (:func:`build_fdn`, or
   :func:`trainable_from_build` from an existing :class:`~pyFDN.FDNBuild`).
2. **train** it toward an objective (:func:`train_fdn`), built by composing
   losses with ``+`` and ``*``::

       loss = pyFDN.FlatMagnitude() + 0.2 * pyFDN.Sparsity(pyFDN.param(model, "feedback"))

   Losses come in two families: those that read the model's impulse response
   (:class:`~pyFDN.train.losses.base.ResponseLoss`) and those that put a cost on
   one of its parameters (:class:`~pyFDN.train.losses.base.ParameterLoss`).
3. **extract** an :class:`~pyFDN.FDNBuild` back out
   (:func:`pyFDN.extract_build`), plus a :class:`TrainLog`.
"""

from __future__ import annotations

from .build import (
    LOSSLESS_ALIAS_DECAY_DB,
    MatrixParam,
    Trainable,
    build_fdn,
    build_set_decay,
    trainable_from_build,
    trainable_from_preset,
)
from .engine import TrainLog, train_fdn
from .filters import AttenuationFilter, EQDesign, OutputEQ
from .losses import (
    L1,
    L2,
    AsymmetricFlatMagnitude,
    Energy,
    FlatMagnitude,
    FlatSpectrogram,
    Loss,
    MatchCumulativeEnergy,
    MatchEnergyDecay,
    MatchImpulseResponse,
    MatchMagnitude,
    MatchMelSpectrogram,
    MatchSpectrogram,
    ParameterLoss,
    ResponseLoss,
    Sparsity,
    MatchDC,
    MatchESR,
    MatchLogCosh,
    MatchSDSDR,
    MatchSISDR,
    MatchSNR,
)
from .params import ParamRef, param, params
from .response import Response, impulse_excitation, model_response

__all__ = [
    # build
    "build_fdn",
    "trainable_from_build",
    "trainable_from_preset",
    "build_set_decay",
    "Trainable",
    "MatrixParam",
    "LOSSLESS_ALIAS_DECAY_DB",
    "AttenuationFilter",
    "OutputEQ",
    "EQDesign",
    # train
    "train_fdn",
    "TrainLog",
    # what a loss sees
    "Response",
    "model_response",
    "impulse_excitation",
    "param",
    "params",
    "ParamRef",
    # losses
    "Loss",
    "ResponseLoss",
    "ParameterLoss",
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
    "Sparsity",
    "L1",
    "L2",
    "MatchDC",
    "MatchESR",
    "MatchLogCosh",
    "MatchSDSDR",
    "MatchSISDR",
    "MatchSNR",
]
