import numpy as np

def hertz2unit(hz, fs):
    """Convert frequency (Hz) to normalized frequency (0–1)."""
    return np.asarray(hz) / fs * 2
