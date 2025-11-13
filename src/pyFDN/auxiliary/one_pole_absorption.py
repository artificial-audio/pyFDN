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
    # Return SOS format: shape (6, N) with [b0, b1, b2, a0, a1, a2] per channel
    N = len(b0)
    sos = np.zeros((6, N))
    sos[0, :] = b0  # b0
    sos[1, :] = 0   # b1
    sos[2, :] = 0   # b2
    sos[3, :] = 1   # a0
    sos[4, :] = a1  # a1
    sos[5, :] = 0   # a2
    return sos


def one_pole_absorption(RT_DC, RT_NY, delays, fs):
    HDc = db2mag(delays * RT602slope(RT_DC, fs))
    HNyq = db2mag(delays * RT602slope(RT_NY, fs))
    sos = design_one_pole_filter(HDc, HNyq)
    return sos
