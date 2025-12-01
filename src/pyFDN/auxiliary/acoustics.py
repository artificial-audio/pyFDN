"""Acoustics and RT60 related functions."""
from __future__ import annotations
import warnings
from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike
from numpy.linalg import svd
from scipy.signal import firwin2, freqz, group_delay
from scipy.interpolate import interp1d

from pyFDN.auxiliary.utils import db2mag, mag2db, hertz2unit


def rt60_to_slope(rt60: ArrayLike, fs: float) -> np.ndarray:
    """Convert a 60 dB decay time to an energy decay slope (dB per sample)."""

    rt_arr = np.asarray(rt60, dtype=float)
    return -60.0 / (rt_arr * fs)


def slope_to_rt60(slope: ArrayLike, fs: float) -> np.ndarray:
    """Convert slope (dB/sample) to T60 in seconds."""
    return -60.0 / (slope * fs)


def absorption_filters(frequency, targetRT60, filterOrder, delays, fs):
    """
    Generate FIR absorption filters for each channel.
    frequency: [freq_points]
    targetRT60: shape (freq_points, channels)
    delays: array of length channels
    """
    num_channels = len(delays)
    unit_freq = hertz2unit(frequency, fs)
    FIR = np.zeros((num_channels, filterOrder + 1))

    if filterOrder == 0:
        rt60 = targetRT60[0, :]
        db = delays * rt60_to_slope(rt60, fs)
        FIR[:, 0] = db2mag(db)
    else:
        for ch in range(num_channels):
            rt60 = targetRT60[:, ch]
            delay = delays[ch] + int(np.ceil(filterOrder / 2))
            db = delay * rt60_to_slope(rt60, fs)
            target_amp = db2mag(db)
            # firwin2 expects normalized [0..1] freqs and gain values
            FIR[ch, :] = firwin2(filterOrder + 1, unit_freq, target_amp)
    return FIR


def absorption_to_t60(filterCoeffs, delays, nfft, fs):
    """Compute T60 from recursive absorption filter with delay."""
    filterLen = filterCoeffs.shape[1]
    response = np.fft.fft(filterCoeffs, nfft, axis=1)
    freq = np.linspace(0, fs/2, nfft // 2, endpoint=False)

    response = response[:, :nfft // 2]
    freq = freq[:nfft // 2]

    totalDelay = delays[:, None] + filterLen / 2
    decayPerSample = mag2db(np.abs(response)) / totalDelay
    T60 = slope_to_rt60(decayPerSample, fs)
    return T60.T, freq  # shape: (freq_points, channels)





def one_pole_absorption(rt_dc: float, rt_ny: float, delays: ArrayLike, fs: float) -> np.ndarray:
    """Design one-pole absorption filters according to specified T60.

    Returns SOS format: shape (6, N) with [b0, b1, b2, a0, a1, a2] per channel.
    """
    delays_arr = np.asarray(delays, dtype=float)
    
    # Calculate target gains
    slope_dc = rt60_to_slope(rt_dc, fs)
    slope_ny = rt60_to_slope(rt_ny, fs)
    
    # Convert to linear magnitude
    h_dc = db2mag(delays_arr * slope_dc)
    h_ny = db2mag(delays_arr * slope_ny)
    
    # Design filters
    r = h_dc / h_ny
    a1 = (1.0 - r) / (1.0 + r)
    b0 = (1.0 - a1) * h_ny

    num_filters = h_dc.size
    sos = np.zeros((6, num_filters))
    sos[0, :] = b0  # b0
    sos[3, :] = 1.0 # a0
    sos[4, :] = a1  # a1
    
    return sos



