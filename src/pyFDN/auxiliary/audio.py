"""Helpers for loading the audio files packaged with pyFDN."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np

AUDIO_SOURCE_DIR = Path(__file__).resolve().parent.parent / "audio"


def load_audio(
    name: str,
    *,
    fs: int | None = None,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load a packaged audio sample.

    Parameters
    ----------
    name : str
        Name of the sample (e.g. ``"synth_dry"`` or ``"synth_dry.wav"``).
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
    import soundfile as sf

    # Strip file extension if provided
    sample_name = name.rsplit(".", 1)[0] if "." in name else name

    samples_dict = list_samples()
    if sample_name not in samples_dict:
        raise ValueError(
            f"Unknown sample '{sample_name}'. "
            f"Available samples: {list(samples_dict.keys())}"
        )

    relative_path = samples_dict[sample_name]
    path = files("pyFDN.audio") / relative_path

    with path.open("rb") as f:
        data, file_fs = sf.read(f, dtype="float64")

    if mono and data.ndim > 1:
        data = data[:, 0]

    if fs is not None and file_fs != fs:
        from scipy.signal import resample

        new_length = int(round(len(data) * fs / file_fs))
        data = resample(data, new_length)
        file_fs = fs

    return data, file_fs


def list_samples() -> dict[str, str]:
    """Scan the audio folder and return a dictionary of file names to relative paths.

    Returns
    -------
    dict[str, str]
        Dictionary mapping file names to their relative paths within the audio folder.
    """
    samples = {}
    for path in sorted(AUDIO_SOURCE_DIR.rglob("*.wav")):
        relative = path.relative_to(AUDIO_SOURCE_DIR)
        filename = path.stem  # Get the file name without extension
        samples[filename] = relative.as_posix()
    return samples


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
