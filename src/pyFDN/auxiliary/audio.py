"""Helpers for loading the audio files packaged with pyFDN."""

from __future__ import annotations

from importlib.resources import files

import numpy as np

SAMPLES = {
    "synth_dry": {
        "file": "general/synth_dry.wav",
        "description": "",
        "Source": "",
    },
    "speech1": {
        "file": "speech/p008_emo_contentment_sentences.wav",
        "description": "Human speech",
        "Source": "Richter, J., Wu, Y.-C., Krenn, S., Welker, S., Lay, B., Watanabe, S., Richard, A., Gerkmann, T. (2024) EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation. Proc. Interspeech 2024, 4873-4877, doi: 10.21437/Interspeech.2024-153",
    },
    "drum1": {
        "file": "drums/drum.wav",
        "description": "Synthesized Drum",
        "Source": "www.openairlib.net",
    },
    "string1": {
        "file": "strings/cl-class-bb-arp-des-32.wav",
        "description": "String Instrument",
        "Source": "www.openairlib.net",
    },
    "wind": {
        "file": "wind/tr-1967-ex1-32.wav",
        "description": "Piccolo trumpet",
        "Source": "www.openairlib.net",
    },
}


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
        Name of the sample (e.g. ``"room_small"``).
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

    if name not in SAMPLES:
        raise ValueError(
            f"Unknown sample '{name}'. "
            f"Available samples: {list(SAMPLES.keys())}"
        )

    filename = SAMPLES[name]["file"]
    path = files("pyFDN.audio") / filename

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


def list_samples():
    return SAMPLES.copy()