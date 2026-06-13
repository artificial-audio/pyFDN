"""Helpers for loading the audio files packaged with pyFDN."""

from __future__ import annotations

from importlib.resources import files

import numpy as np


def load_audio(
    name: str,
    *,
    fs: int | None = None,
    package: str = "pyFDN.audio",
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load a packaged audio file as a NumPy array.

    Parameters
    ----------
    name : str
        File name within ``package`` (e.g. ``"synth_dry.wav"``).
    fs : int, optional
        Target sampling rate. If given and different from the file's rate,
        the signal is resampled to ``fs``.
    package : str
        Importable package holding the audio resource.
    mono : bool
        If True, keep only the first channel of multichannel files.

    Returns
    -------
    (signal, fs) : tuple[np.ndarray, int]
        Samples as float64 and the (possibly resampled) sampling rate.
    """
    import soundfile as sf

    path = files(package) / name
    try:
        with path.open("rb") as f:
            data, file_fs = sf.read(f, dtype="float64")
    except FileNotFoundError:
        raise FileNotFoundError(f"{package}/{name} not found.") from None

    if mono and data.ndim > 1:
        data = data[:, 0]

    if fs is not None and file_fs != fs:
        from scipy.signal import resample

        data = resample(data, int(round(len(data) * fs / file_fs)))
        file_fs = fs

    return data, file_fs
