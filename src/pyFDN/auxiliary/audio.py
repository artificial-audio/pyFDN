"""Helpers for the compact audio examples distributed with pyFDN."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import soundfile as sf

_AUDIO_ROOT = files("pyFDN.resources").joinpath("audio")


@lru_cache(maxsize=1)
def _metadata() -> tuple[dict[str, Any], ...]:
    metadata_file = _AUDIO_ROOT.joinpath("metadata.json")
    return tuple(json.loads(metadata_file.read_text(encoding="utf-8")))


def available_audio() -> tuple[str, ...]:
    """Return the names accepted by :func:`load_audio`."""

    return tuple(sorted(item["name"] for item in _metadata()))


def audio_metadata(name: str) -> dict[str, Any]:
    """Return attribution and license metadata for a packaged audio sample."""

    sample_name = PurePosixPath(name).stem
    for item in _metadata():
        if item["name"] == sample_name or item["filename"] == name:
            return dict(item)
    choices = ", ".join(available_audio())
    raise ValueError(f"Unknown audio sample '{name}'. Available samples: {choices}")


def load_audio(
    name: str,
    *,
    fs: int | None = None,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load an audio sample distributed with pyFDN.

    Parameters
    ----------
    name : str
        Sample name, with or without the ``.wav`` extension. See
        :func:`available_audio`.
    fs : int, optional
        Target sampling rate. A differing source rate is resampled.
    mono : bool
        If ``True``, retain only the first channel of multichannel audio.

    Returns
    -------
    signal : np.ndarray
        Audio samples as ``float64``.
    fs : int
        Sampling rate of the returned signal.
    """

    metadata = audio_metadata(name)
    resource = _AUDIO_ROOT.joinpath(metadata["category"], metadata["filename"])
    with as_file(resource) as path:
        data, file_fs = sf.read(path, dtype="float64")

    if mono and data.ndim > 1:
        data = data[:, 0]

    if fs is not None and file_fs != fs:
        from scipy.signal import resample

        new_length = int(round(len(data) * fs / file_fs))
        data = resample(data, new_length)
        file_fs = fs

    return data, int(file_fs)
