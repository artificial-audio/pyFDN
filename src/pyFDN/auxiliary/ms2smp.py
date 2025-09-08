import numpy as np

def ms2smp(ms, fs):
    """Convert milliseconds to samples."""
    return np.round(np.array(ms) * fs / 1000).astype(int)