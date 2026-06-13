"""Shelving filter design for graphic EQ.

Translation of shelvingFilter.m from fdnToolbox.
Equations (18) and (20) in Välimäki and Reiss, "All About Audio Equalization:
Solutions and Frontiers," Applied Sciences, vol. 6, no. 5, p. 129, 2016.
"""

from __future__ import annotations

import numpy as np


def shelving_filter(
    omega_c: float,
    gain: float,
    filter_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Design a shelving biquad filter.

    Args:
        omega_c: Cut-off frequency in radians.
        gain: Linear gain (not dB).
        filter_type: ``"low"`` for low-shelf, ``"high"`` for high-shelf.

    Returns:
        ``(b, a)`` — numerator and denominator coefficients of length 3.
    """
    t = np.tan(omega_c / 2)
    t2 = t**2
    g2 = gain**0.5
    g4 = gain**0.25
    sqrt2 = np.sqrt(2.0)

    b = np.array(
        [
            g2 * t2 + sqrt2 * t * g4 + 1,
            2 * g2 * t2 - 2,
            g2 * t2 - sqrt2 * t * g4 + 1,
        ]
    )
    a = np.array(
        [
            g2 + sqrt2 * t * g4 + t2,
            2 * t2 - 2 * g2,
            g2 - sqrt2 * t * g4 + t2,
        ]
    )
    b = g2 * b

    if filter_type == "high":
        b, a = a * gain, b
    elif filter_type != "low":
        raise ValueError(f"filter_type must be 'low' or 'high', got {filter_type!r}")

    return b, a
