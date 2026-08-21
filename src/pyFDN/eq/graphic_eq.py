"""Proportional parametric graphic equalizer.

Translation of graphicEQ.m from fdnToolbox.

References:
    Välimäki and Reiss, "All About Audio Equalization: Solutions and Frontiers,"
    Applied Sciences, vol. 6, no. 5, p. 129, 2016.

    Jot, "Proportional Parametric Equalizers - Application to Digital
    Reverberation and Environmental Audio Processing," AES Conv. 2015.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._backend import array_namespace
from .bandpass_filter import bandpass_filter
from .shelving_filter import shelving_filter


def graphic_eq(
    center_omega: np.ndarray,
    shelving_omega: np.ndarray,
    R: float,
    gain_db: Any,
) -> Any:
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
        gain_db: Command gains in dB, one per section: shape
                 ``(num_center + 3,)``, or ``(num_center + 3, ...)`` to design
                 that many EQs at once. A torch tensor designs the bank
                 differentiably -- see :mod:`._backend`.

    Returns:
        SOS bank of shape ``(num_bands, 6) + gain_db.shape[1:]``, the section
        coefficients ``[b0, b1, b2, a0, a1, a2]`` along axis 1. Sections are
        *not* normalized to ``a0 = 1``.
    """
    xp = array_namespace(gain_db)
    if xp is np:
        gain_db = np.asarray(gain_db, dtype=float)
    center_omega = np.asarray(center_omega, dtype=float)
    shelving_omega = np.asarray(shelving_omega, dtype=float)

    num_freq = len(center_omega) + len(shelving_omega) + 1
    if gain_db.shape[0] != num_freq:
        raise ValueError(f"Expected {num_freq} gains, got {gain_db.shape[0]}")

    Q = math.sqrt(R) / (R - 1)
    gains = 10.0 ** (gain_db / 20.0)  # dB → linear

    sections = []
    for band in range(num_freq):
        g = gains[band]
        if band == 0:
            zero, one = xp.zeros_like(g), xp.ones_like(g)
            b, a = xp.stack([g, zero, zero], 0), xp.stack([one, zero, zero], 0)
        elif band == 1:
            b, a = shelving_filter(shelving_omega[0], g, "low")
        elif band == num_freq - 1:
            b, a = shelving_filter(shelving_omega[1], g, "high")
        else:
            b, a = bandpass_filter(center_omega[band - 2], g, Q)
        sections.append(xp.concatenate([b, a], 0))

    return xp.stack(sections, 0)
