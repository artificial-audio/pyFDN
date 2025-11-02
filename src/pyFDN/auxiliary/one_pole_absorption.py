from __future__ import annotations
import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike
from pyFDN.auxiliary.rt60_to_slope import rt60_to_slope


def one_pole_absorption(rt_dc: float, rt_ny: float, delays: ArrayLike, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Design one-pole absorption filters according to specified T60."""

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
