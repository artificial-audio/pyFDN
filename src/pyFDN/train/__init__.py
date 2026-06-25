"""Training pipeline for FDNs -- an explicit three-step API over flamo.

1. **build** a trainable flamo model from a config (:func:`build_fdn`, or
   :func:`trainable_from_build` from an existing :class:`~pyFDN.FDNBuild`).
2. **train** the model toward a *mode* (:func:`train_fdn`) -- ``colorless``,
   ``match_spectrogram`` or ``match_mel_spectrogram``, with optional ``target``
   data.
3. **extract** an :class:`~pyFDN.FDNBuild` back out
   (:func:`pyFDN.extract_build`), plus metadata
   (:class:`TrainLog`).

The dataclasses and numpy metrics here import without torch; the builder, engine
and criteria import torch/flamo lazily, so ``import pyFDN`` stays torch-free.
"""

from __future__ import annotations

from .build import MatrixParam, Trainable, build_fdn, trainable_from_build, with_decay
from .engine import TrainLog, train_fdn
from .metrics import (
    edc_l1,
    flatness_from_magnitude,
    magnitude_response,
    mr_stft_distance,
    octave_colouration,
    spectral_flatness,
)
from .objectives import Objective

__all__ = [
    # build
    "build_fdn",
    "trainable_from_build",
    "with_decay",
    "Trainable",
    "MatrixParam",
    # train
    "train_fdn",
    "Objective",
    "TrainLog",
    # metrics
    "spectral_flatness",
    "flatness_from_magnitude",
    "octave_colouration",
    "edc_l1",
    "mr_stft_distance",
    "magnitude_response",
]
