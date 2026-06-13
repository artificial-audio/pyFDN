"""Frequency response probing of a cascade of biquad sections.

Translation of probeSOS.m from fdnToolbox.
"""

# TODO: check whether this resampling step is used anywhere.

from __future__ import annotations

import numpy as np
from scipy.signal import freqz


def probe_sos(
    sos: np.ndarray,
    control_frequencies: np.ndarray | None = None,
    fft_len: int = 4096,
    fs: float = 48000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the magnitude response of each biquad at control frequencies.

    Args:
        sos: Filter matrix of shape ``(num_bands, 6)`` with columns
             ``[b0, b1, b2, a0, a1, a2]`` (rows are independent sections).
        control_frequencies: Frequencies in Hz at which to evaluate.
        fft_len: FFT length for the frequency response computation.
        fs: Sampling frequency in Hz.

    Returns:
        ``(G, H, W)`` where

        - ``G`` — magnitude in dB, shape ``(len(control_frequencies), num_bands)``.
        - ``H`` — complex frequency response, shape ``(fft_len, num_bands)``.
        - ``W`` — frequency axis in Hz, shape ``(fft_len, num_bands)``.
    """
    if control_frequencies is None:
        control_frequencies = np.array([])

    num_freq = sos.shape[0]
    H = np.zeros((fft_len, num_freq), dtype=complex)
    W = np.zeros((fft_len, num_freq))
    G = np.zeros((len(control_frequencies), num_freq))

    for band in range(num_freq):
        b = sos[band, :3]
        a = sos[band, 3:6]
        w, h = freqz(b, a, worN=fft_len, fs=fs)
        mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-300)).astype(float)
        g = np.interp(control_frequencies, w.astype(float), mag_db)
        G[:, band] = g
        H[:, band] = h
        W[:, band] = w

    return G, H, W
