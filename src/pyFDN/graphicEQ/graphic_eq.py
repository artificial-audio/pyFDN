"""Proportional parametric graphic equalizer.

Translation of graphicEQ.m from fdnToolbox.

References:
    Välimäki and Reiss, "All About Audio Equalization: Solutions and Frontiers,"
    Applied Sciences, vol. 6, no. 5, p. 129, 2016.

    Jot, "Proportional Parametric Equalizers - Application to Digital
    Reverberation and Environmental Audio Processing," AES Conv. 2015.
"""

from __future__ import annotations

import numpy as np

from .bandpass_filter import bandpass_filter
from .shelving_filter import shelving_filter


def graphic_eq(
    center_omega: np.ndarray,
    shelving_omega: np.ndarray,
    R: float,
    gain_db: np.ndarray,
) -> np.ndarray:
    """Build a graphic EQ as a bank of independent biquad sections.

    Band layout (total ``len(center_omega) + len(shelving_omega) + 1`` sections):

    - Band 0: flat gain section.
    - Band 1: low shelving filter.
    - Bands 2 … N-2: peaking bandpass filters.
    - Band N-1: high shelving filter.

    Args:
        center_omega: Center frequencies of bandpass bands in radians,
                      shape ``(num_center,)``.
        shelving_omega: Cut-off frequencies of shelving bands in radians,
                        shape ``(2,)`` — ``[low_crossover, high_crossover]``.
        R: Bandwidth parameter; quality factor is ``sqrt(R) / (R - 1)``.
        gain_db: Command gains in dB for each section, shape
                 ``(num_center + 3,)`` (flat + low shelf + bandpass + high shelf).

    Returns:
        SOS matrix of shape ``(num_bands, 6)`` with columns
        ``[b0, b1, b2, a0, a1, a2]``.
    """
    center_omega = np.asarray(center_omega, dtype=float)
    shelving_omega = np.asarray(shelving_omega, dtype=float)
    gain_db = np.asarray(gain_db, dtype=float)

    num_freq = len(center_omega) + len(shelving_omega) + 1
    if len(gain_db) != num_freq:
        raise ValueError(f"Expected {num_freq} gains, got {len(gain_db)}")

    sos = np.zeros((num_freq, 6))
    Q = np.sqrt(R) / (R - 1)

    for band in range(num_freq):
        g = 10.0 ** (gain_db[band] / 20.0)  # dB → linear
        if band == 0:
            b = np.array([g, 0.0, 0.0])
            a = np.array([1.0, 0.0, 0.0])
        elif band == 1:
            b, a = shelving_filter(shelving_omega[0], g, "low")
        elif band == num_freq - 1:
            b, a = shelving_filter(shelving_omega[1], g, "high")
        else:
            b, a = bandpass_filter(center_omega[band - 2], g, Q)
        sos[band, :3] = b
        sos[band, 3:] = a

    return sos
