"""Training pipeline for FDNs -- an explicit three-step API over flamo.

1. **build** a trainable flamo model from a config (:func:`build_fdn`, or
   :func:`trainable_from_build` from an existing :class:`~pyFDN.FDNBuild`).
2. **train** the model toward a *mode* (:func:`train_fdn`) -- ``colorless``,
   ``match_spectrogram`` or ``match_mel_spectrogram``, with optional ``target``
   data.
3. **extract** an :class:`~pyFDN.FDNBuild` back out
   (:func:`pyFDN.extract_build`), plus a :class:`TrainLog`.
"""

from __future__ import annotations

from .build import MatrixParam, Trainable, build_fdn, trainable_from_build, with_decay
from .engine import TrainLog, train_fdn
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
]
