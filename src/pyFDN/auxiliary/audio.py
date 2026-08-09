"""Helpers for loading the audio files packaged with pyFDN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

def load_audio(
    source_dir,
    name: str,
    *,
    fs: int | None = None,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load a packaged audio sample.

    Parameters
    ----------
    name : str
        Name of the sample.
    source_dir : Path
        Root directory to search for the audio file.
    fs : int, optional
        Target sampling rate. If given and different from the original
        sampling rate, the signal is resampled.
    mono : bool
        If True, keep only the first channel of multichannel audio.

    Returns
    -------
    signal : np.ndarray
        Audio samples as float64.
    fs : int
        Sampling rate of the returned signal.
    """

    # Strip file extension if provided
    sample_name = name.rsplit(".", 1)[0] if "." in name else name
    filename = f"{sample_name}.wav"

    try:
        path = find_file(Path(source_dir), filename)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Unknown sample '{sample_name}'. "
            f"No file named '{filename}' found under '{source_dir}'."
        ) from exc

    data, file_fs = sf.read(str(path), dtype="float64")

    if mono and data.ndim > 1:
        data = data[:, 0]

    if fs is not None and file_fs != fs:
        from scipy.signal import resample

        new_length = int(round(len(data) * fs / file_fs))
        data = resample(data, new_length)
        file_fs = fs

    return data, file_fs

def find_file(root, filename):
    for item in root.iterdir():
        if item.is_file() and item.name == filename:
            return item
        if item.is_dir():
            try:
                return find_file(item, filename)
            except FileNotFoundError:
                pass
    raise FileNotFoundError(filename)
