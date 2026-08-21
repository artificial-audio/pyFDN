"""Peaking bandpass filter design for graphic EQ.

Translation of bandpassFilter.m from fdnToolbox.
Equation (29) in Välimäki and Reiss, "All About Audio Equalization:
Solutions and Frontiers," Applied Sciences, vol. 6, no. 5, p. 129, 2016.
"""

from __future__ import annotations

import math
from typing import Any

from ._backend import array_namespace


def bandpass_filter(
    omega_c: float,
    gain: Any,
    Q: float,
) -> tuple[Any, Any]:
    """Design a peaking bandpass biquad filter.

    Args:
        omega_c: Center frequency in radians.
        gain: Linear gain at center frequency. A scalar, or an array or tensor
              of gains to design the same band for -- see :mod:`._backend`.
        Q: Quality factor.

    Returns:
        ``(b, a)`` — numerator and denominator coefficients, shape
        ``(3,) + gain.shape``.
    """
    xp = array_namespace(gain)

    t = math.tan(float(omega_c) / (2 * Q))
    cos2 = -2 * math.cos(float(omega_c))
    sg = xp.sqrt(gain)
    one = xp.ones_like(gain)

    b = xp.stack([sg + gain * t, cos2 * sg, sg - gain * t], 0)
    a = xp.stack([sg + t * one, cos2 * sg, sg - t * one], 0)
    return b, a
