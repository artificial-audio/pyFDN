"""Shelving filter design for graphic EQ.

Translation of shelvingFilter.m from fdnToolbox.
Equations (18) and (20) in Välimäki and Reiss, "All About Audio Equalization:
Solutions and Frontiers," Applied Sciences, vol. 6, no. 5, p. 129, 2016.
"""

from __future__ import annotations

import math
from typing import Any

from ._backend import array_namespace


def shelving_filter(
    omega_c: float,
    gain: Any,
    filter_type: str,
) -> tuple[Any, Any]:
    """Design a shelving biquad filter.

    Args:
        omega_c: Cut-off frequency in radians.
        gain: Linear gain (not dB). A scalar, or an array or tensor of gains to
              design the same shelf for -- see :mod:`._backend`.
        filter_type: ``"low"`` for low-shelf, ``"high"`` for high-shelf.

    Returns:
        ``(b, a)`` — numerator and denominator coefficients, shape
        ``(3,) + gain.shape``.
    """
    if filter_type not in ("low", "high"):
        raise ValueError(f"filter_type must be 'low' or 'high', got {filter_type!r}")

    xp = array_namespace(gain)

    t = math.tan(float(omega_c) / 2)
    t2 = t * t
    sqrt2 = math.sqrt(2.0)
    g2 = gain**0.5
    g4 = gain**0.25
    one = xp.ones_like(gain)

    b = g2 * xp.stack(
        [
            g2 * t2 + sqrt2 * t * g4 + one,
            2 * g2 * t2 - 2 * one,
            g2 * t2 - sqrt2 * t * g4 + one,
        ],
        0,
    )
    a = xp.stack(
        [
            g2 + sqrt2 * t * g4 + t2 * one,
            2 * t2 * one - 2 * g2,
            g2 - sqrt2 * t * g4 + t2 * one,
        ],
        0,
    )

    if filter_type == "high":
        return a * gain, b
    return b, a
