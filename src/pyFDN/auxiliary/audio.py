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

def load_sample(
    name: str,
    *,
    fs: int | None = None,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load a packaged audio sample.

    Parameters
    ----------
    name : str
        Name of the sample (e.g. ``"synth_dry"``).
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

    samples_dict = list_samples()
    if name not in samples_dict:
        raise ValueError(
            f"Unknown sample '{name}'. "
            f"Available samples: {list(samples_dict.keys())}"
        )

    relative_path = samples_dict[name]
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