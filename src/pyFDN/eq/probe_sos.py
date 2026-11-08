"""Frequency response probing of a cascade of biquad sections.

Translation of probeSOS.m from fdnToolbox.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import freqz


def probe_sos(
    sos: np.ndarray,
    fs: float,
    *,
    control_frequencies: np.ndarray | None = None,
    nfft: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the magnitude response of each biquad at control frequencies.

    Parameters
    ----------
    sos
        Filter matrix of shape ``(num_bands, 6)`` with columns
        ``[b0, b1, b2, a0, a1, a2]`` (rows are independent sections).
    fs
        Sampling frequency in Hz.
    control_frequencies
        Frequencies in Hz at which to evaluate.
    nfft
        Number of frequency points of the response.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        ``(G, H, W)`` where

        - ``G`` — magnitude in dB, shape ``(len(control_frequencies), num_bands)``.
        - ``H`` — complex frequency response, shape ``(nfft, num_bands)``.
        - ``W`` — frequency axis in Hz, shape ``(nfft, num_bands)``.
    """
    if control_frequencies is None:
        control_frequencies = np.array([])

    num_freq = sos.shape[0]
    H = np.zeros((nfft, num_freq), dtype=complex)
    W = np.zeros((nfft, num_freq))
    G = np.zeros((len(control_frequencies), num_freq))

    for band in range(num_freq):
        b = sos[band, :3]
        a = sos[band, 3:6]
        w, h = freqz(b, a, worN=nfft, fs=fs)
        mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-300)).astype(float)
        g = np.interp(control_frequencies, w.astype(float), mag_db)
        G[:, band] = g
        H[:, band] = h
        W[:, band] = w

    return G, H, W
