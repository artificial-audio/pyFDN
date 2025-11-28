from __future__ import annotations
import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike
from pyFDN.auxiliary.rt60_to_slope import rt60_to_slope
from pyFDN.auxiliary.db2mag import db2mag
from pyFDN.auxiliary.RT602slope import RT602slope


def one_pole_absorption(rt_dc: float, rt_ny: float, delays: ArrayLike, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Design one-pole absorption filters according to specified T60.

    Returns (b, a) format where b has shape (N, 1, 1) and a has shape (N, 1, 2).
    """
    delays_arr = np.asarray(delays, dtype=float)
    HDc = np.power(10.0, delays_arr * rt60_to_slope(rt_dc, fs) / 20.0)
    HNy = np.power(10.0, delays_arr * rt60_to_slope(rt_ny, fs) / 20.0)
    return _design_one_pole_filters(HDc, HNy)


def _design_one_pole_filters(h_dc: np.ndarray, h_ny: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = h_dc / h_ny
    a1 = (1.0 - r) / (1.0 + r)
    b0 = (1.0 - a1) * h_ny

    num_filters = h_dc.size
    b = b0.reshape(num_filters, 1, 1)
    a = np.zeros((num_filters, 1, 2), dtype=float)
    a[:, 0, 0] = 1.0
    a[:, 0, 1] = a1
    return b, a


def clip(x, minmax):
    """Clip values in x to the range [min, max]."""
    return np.minimum(np.maximum(x, minmax[0]), minmax[1])


def slope2RT60(slope, fs):
    """Convert energy decay slope to RT60 in seconds."""
    # MATLAB: RT60 = (-60./ clip(slope, [-Inf, -eps]) )./fs;
    slope = clip(slope, [-np.inf, -np.finfo(float).eps])
    return (-60.0 / slope) / fs


def design_one_pole_filter(HDc, HNyq):
    """Design one-pole filter, returning SOS format: shape (6, N) with [b0, b1, b2, a0, a1, a2] per channel."""
    r = HDc / HNyq
    a1 = (1 - r) / (1 + r)
    b0 = (1 - a1) * HNyq
    N = len(b0)
    sos = np.zeros((6, N))
    sos[0, :] = b0  # b0
    sos[1, :] = 0   # b1
    sos[2, :] = 0   # b2
    sos[3, :] = 1   # a0
    sos[4, :] = a1  # a1
    sos[5, :] = 0   # a2
    return sos


# Backward compatibility function for SOS format
def one_pole_absorption_sos(RT_DC, RT_NY, delays, fs):
    """Design one-pole absorption filters, returning SOS format for backward compatibility.

    This function maintains compatibility with code expecting SOS format output.
    Returns SOS format: shape (6, N) with [b0, b1, b2, a0, a1, a2] per channel.
    """
    HDc = db2mag(delays * RT602slope(RT_DC, fs))
    HNyq = db2mag(delays * RT602slope(RT_NY, fs))
    sos = design_one_pole_filter(HDc, HNyq)
    return sos

