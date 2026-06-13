"""Peaking bandpass filter design for graphic EQ.

Translation of bandpassFilter.m from fdnToolbox.
Equation (29) in Välimäki and Reiss, "All About Audio Equalization:
Solutions and Frontiers," Applied Sciences, vol. 6, no. 5, p. 129, 2016.
"""

from __future__ import annotations

import numpy as np


def bandpass_filter(
    omega_c: float,
    gain: float,
    Q: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Design a peaking bandpass biquad filter.

    Args:
        omega_c: Center frequency in radians.
        gain: Linear gain at center frequency.
        Q: Quality factor.

    Returns:
        ``(b, a)`` — numerator and denominator coefficients of length 3.
    """
    band_width = omega_c / Q
    t = np.tan(band_width / 2)
    sg = np.sqrt(gain)

    b = np.array([sg + gain * t, -2 * sg * np.cos(omega_c), sg - gain * t])
    a = np.array([sg + t, -2 * sg * np.cos(omega_c), sg - t])
    return b, a
