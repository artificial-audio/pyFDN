"""10-band graphic EQ design via constrained least squares.

Translation of designGEQ.m from fdnToolbox.

Reference:
    Välimäki and Reiss, "All About Audio Equalization: Solutions and Frontiers,"
    Applied Sciences, vol. 6, no. 5, p. 129, 2016.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from ..auxiliary.utils import hertz_to_rad
from .graphic_eq import graphic_eq
from .probe_sos import probe_sos


def design_geq(
    target_g: np.ndarray,
    fs: float = 48000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Design a 10-band graphic EQ matching a target magnitude response.

    The EQ has 8 peaking bandpass bands plus low and high shelving filters,
    plus a flat-gain section (11 sections total).

    Args:
        target_g: Target magnitude response in dB at 10 frequency bands
                  (DC=1 Hz, 63, 125, 250, 500, 1k, 2k, 4k, 8k Hz, Nyquist).
                  Shape ``(10,)``.
        fs: Sampling frequency in Hz (default 48000).

    Returns:
        ``(sos, target_f)`` where

        - ``sos`` — single SOS cascade of shape ``(n_sections, 6)`` (n_sections = 11).
        - ``target_f`` — 10-point frequency grid used for the target.
    """
    target_g = np.asarray(target_g, dtype=float).ravel()
    fft_len = 2**16

    center_frequencies = np.array(
        [63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float
    )
    shelving_crossover = np.array([46.0, 11360.0])
    num_freq = len(center_frequencies) + len(shelving_crossover)  # 10

    shelving_omega = hertz_to_rad(shelving_crossover, fs)
    center_omega = hertz_to_rad(center_frequencies, fs)
    R = 2.7

    num_control = 100
    control_frequencies = np.round(np.logspace(0, np.log10(fs / 2.1), num_control + 1))

    target_f = np.concatenate([[1.0], center_frequencies, [float(fs)]])
    target_interp = np.interp(control_frequencies, target_f, target_g)

    prototype_gain = 10.0
    prototype_gain_array = prototype_gain * np.ones(num_freq + 1)
    prototype_sos = graphic_eq(center_omega, shelving_omega, R, prototype_gain_array)
    G, _, _ = probe_sos(prototype_sos, control_frequencies, fft_len, fs)
    G = G / prototype_gain

    upper_bound = np.concatenate([[np.inf], 2 * prototype_gain * np.ones(num_freq)])
    lower_bound = -upper_bound

    result = lsq_linear(G, target_interp, bounds=(lower_bound, upper_bound))
    opt_g = result.x

    sos = graphic_eq(center_omega, shelving_omega, R, opt_g)
    return sos, target_f
