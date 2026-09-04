"""Biquad coefficient primitives used by the EQ designs.

The functions in this module describe filter sections, not why a filter is
being designed.  Their gains are linear amplitudes and their frequencies are
in radians.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._backend import array_namespace


def lowshelf_biquad(omega_c: float, gain: Any) -> tuple[Any, Any]:
    """Return ``(b, a)`` for a second-order low-shelf section."""
    return _shelf_biquad(omega_c, gain, high=False)


def highshelf_biquad(omega_c: float, gain: Any) -> tuple[Any, Any]:
    """Return ``(b, a)`` for a second-order high-shelf section."""
    return _shelf_biquad(omega_c, gain, high=True)


def _shelf_biquad(omega_c: float, gain: Any, *, high: bool) -> tuple[Any, Any]:
    xp = array_namespace(gain)
    if xp is np:
        gain = np.asarray(gain, dtype=float)

    t = math.tan(float(omega_c) / 2.0)
    t2 = t * t
    sqrt2 = math.sqrt(2.0)
    sqrt_gain = gain**0.5
    fourth_root_gain = gain**0.25
    one = xp.ones_like(gain)

    b = sqrt_gain * xp.stack(
        [
            sqrt_gain * t2 + sqrt2 * t * fourth_root_gain + one,
            2.0 * sqrt_gain * t2 - 2.0 * one,
            sqrt_gain * t2 - sqrt2 * t * fourth_root_gain + one,
        ],
        0,
    )
    a = xp.stack(
        [
            sqrt_gain + sqrt2 * t * fourth_root_gain + t2 * one,
            2.0 * t2 * one - 2.0 * sqrt_gain,
            sqrt_gain - sqrt2 * t * fourth_root_gain + t2 * one,
        ],
        0,
    )
    return (a * gain, b) if high else (b, a)


def peaking_biquad(
    omega_c: float,
    gain: Any,
    q: float,
) -> tuple[Any, Any]:
    """Return ``(b, a)`` for a peaking section."""
    xp = array_namespace(gain)
    if xp is np:
        gain = np.asarray(gain, dtype=float)

    t = math.tan(float(omega_c) / (2.0 * q))
    cos2 = -2.0 * math.cos(float(omega_c))
    sqrt_gain = xp.sqrt(gain)
    one = xp.ones_like(gain)

    b = xp.stack([sqrt_gain + gain * t, cos2 * sqrt_gain, sqrt_gain - gain * t], 0)
    a = xp.stack([sqrt_gain + t * one, cos2 * sqrt_gain, sqrt_gain - t * one], 0)
    return b, a


def first_order_shelf_biquad(
    gain_dc: Any,
    gain_nyquist: Any,
    omega_c: float,
) -> Any:
    """Return a normalized one-section SOS from two linear amplitudes."""
    xp = array_namespace(gain_dc)
    if xp is np:
        gain_dc = np.asarray(gain_dc, dtype=float)
        gain_nyquist = np.asarray(gain_nyquist, dtype=float)

    t = math.tan(float(omega_c))
    sqrt_ratio = xp.sqrt(gain_dc / gain_nyquist)
    a0 = t / sqrt_ratio + 1.0
    b0 = (t * sqrt_ratio + 1.0) * gain_nyquist / a0
    b1 = (t * sqrt_ratio - 1.0) * gain_nyquist / a0
    a1 = (t / sqrt_ratio - 1.0) / a0
    zero, one = xp.zeros_like(b0), xp.ones_like(b0)
    return xp.stack([b0, b1, zero, one, a1, zero], 0)[None]


def one_pole_biquad(gain_dc: Any, gain_nyquist: Any) -> Any:
    """Return a normalized one-pole section from two linear amplitudes."""
    xp = array_namespace(gain_dc)
    if xp is np:
        gain_dc = np.asarray(gain_dc, dtype=float)
        gain_nyquist = np.asarray(gain_nyquist, dtype=float)

    ratio = gain_dc / gain_nyquist
    a1 = (1.0 - ratio) / (1.0 + ratio)
    b0 = (1.0 - a1) * gain_nyquist
    zero, one = xp.zeros_like(b0), xp.ones_like(b0)
    return xp.stack([b0, zero, zero, one, a1, zero], 0)[None]


__all__ = [
    "first_order_shelf_biquad",
    "highshelf_biquad",
    "lowshelf_biquad",
    "one_pole_biquad",
    "peaking_biquad",
]
