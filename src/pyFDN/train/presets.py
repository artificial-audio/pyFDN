"""Ready-made training objectives.

Each preset is a short, readable expression over :mod:`pyFDN.train.losses` --
they exist to save typing, not to hide anything. Read one and you have read the
whole objective::

    def colorless(model, *, sparsity=0.2):
        return FlatMagnitude() + sparsity * Sparsity(param(model, "feedback"))

:func:`pyFDN.train_fdn` also accepts a preset's name as a string, which is
shorthand for calling it.
"""

from __future__ import annotations

from typing import Any

from .losses import (
    FlatMagnitude,
    Loss,
    MatchMelSpectrogram,
    MatchSpectrogram,
    Sparsity,
)
from .params import param


def colorless(model: Any, *, sparsity: float = 0.2) -> Loss:
    """Flat magnitude response, plus a density reward on the feedback matrix.

    The objective of *Differentiable FDNs for Colorless Reverberation* (Dal
    Santo et al.) and its "tiny colorless FDN" follow-up. ``sparsity=0`` drops
    the parameter term and fits the magnitude alone.
    """
    loss: Loss = FlatMagnitude()
    if sparsity:
        loss = loss + sparsity * Sparsity(param(model, "feedback"))
    return loss


def match_spectrogram(
    model: Any,
    target: Any,
    *,
    sparsity: float = 0.2,
    nfft: tuple[int, ...] = (256, 512, 1024),
) -> Loss:
    """Multi-resolution STFT match to ``target``, plus the density reward."""
    loss: Loss = MatchSpectrogram(target, nfft=nfft)
    if sparsity:
        loss = loss + sparsity * Sparsity(param(model, "feedback"))
    return loss


def match_mel_spectrogram(
    model: Any,
    target: Any,
    *,
    sparsity: float = 0.2,
    nfft: tuple[int, ...] = (256, 512, 1024),
) -> Loss:
    """Mel-scaled multi-resolution STFT match, plus the density reward."""
    loss: Loss = MatchMelSpectrogram(target, nfft=nfft)
    if sparsity:
        loss = loss + sparsity * Sparsity(param(model, "feedback"))
    return loss


# Preset name -> (factory, needs a target IR).
_PRESETS: dict[str, tuple[Any, bool]] = {
    "colorless": (colorless, False),
    "match_spectrogram": (match_spectrogram, True),
    "match_mel_spectrogram": (match_mel_spectrogram, True),
}

PRESET_NAMES = tuple(_PRESETS)


def resolve(loss: Loss | str, model: Any, *, target: Any = None, **kwargs: Any) -> Loss:
    """Return ``loss`` unchanged, or build the named preset for ``model``.

    :func:`pyFDN.train_fdn` calls this so that a string and a hand-built
    objective can be passed interchangeably.
    """
    if isinstance(loss, Loss):
        if target is not None:
            raise ValueError(
                "target= only applies to a preset name; a loss object holds its "
                "own reference data (e.g. MatchSpectrogram(target))."
            )
        return loss
    if not isinstance(loss, str):
        raise TypeError(
            f"loss must be a pyFDN Loss or a preset name {PRESET_NAMES}; "
            f"got {type(loss).__name__}"
        )
    if loss not in _PRESETS:
        raise ValueError(f"unknown preset {loss!r}; choose from {PRESET_NAMES}")
    factory, needs_target = _PRESETS[loss]
    if needs_target:
        if target is None:
            raise ValueError(f"preset {loss!r} requires target=")
        return factory(model, target, **kwargs)
    if target is not None:
        raise ValueError(f"preset {loss!r} takes no target=")
    return factory(model, **kwargs)
