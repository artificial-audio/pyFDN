import numpy as np

def db2mag(db):
    """Convert dB to magnitude."""
    return 10 ** (db / 20)

def RT602slope(RT60, fs):
    """Convert RT60 to decay slope per sample."""
    # -60 dB decay over RT60 seconds, per sample
    return -60.0 / (RT60 * fs)

def clip(x, minmax):
    """Clip values in x to the range [min, max]."""
    return np.minimum(np.maximum(x, minmax[0]), minmax[1])

def slope2RT60(slope, fs):
    """Convert energy decay slope to RT60 in seconds."""
    # MATLAB: RT60 = (-60./ clip(slope, [-Inf, -eps]) )./fs;
    slope = clip(slope, [-np.inf, -np.finfo(float).eps])
    return (-60.0 / slope) / fs

def design_one_pole_filter(HDc, HNyq):
    r = HDc / HNyq
    a1 = (1 - r) / (1 + r)
    b0 = (1 - a1) * HNyq
    b = b0[:, np.newaxis, np.newaxis]
    a = np.ones_like(b)
    a = np.concatenate([a, a1[:, np.newaxis, np.newaxis]], axis=2)
    return b, a

def one_pole_absorption(RT_DC, RT_NY, delays, fs):
    HDc = db2mag(delays * RT602slope(RT_DC, fs))
    HNyq = db2mag(delays * RT602slope(RT_NY, fs))
    b, a = design_one_pole_filter(HDc, HNyq)
    return b, a
