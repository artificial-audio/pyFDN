"""Absorption filter design via graphic EQ for FDN delay lines.

Translation of absorptionGEQ.m from fdnToolbox.

Reference:
    Schlecht and Habets, "Accurate reverberation time control in feedback
    delay networks," Proc. DAFx, 2017.
"""

from __future__ import annotations

import numpy as np

from ..auxiliary.acoustics import rt_to_slope
from .design_geq import design_geq


def absorption_geq(
    rt: np.ndarray,
    delays: np.ndarray,
    fs: float,
) -> np.ndarray:
    """Design per-delay GEQ absorption filters matching target reverberation times.

    Each delay line gets its own cascade of biquad sections whose combined
    attenuation per round trip matches the desired RT at each frequency band.

    Args:
        rt: Target reverberation time in seconds at 10 frequency bands,
            shape ``(10,)`` or broadcastable.
        delays: Delay lengths in samples, shape ``(num_delays,)``.
        fs: Sampling frequency in Hz.

    Returns:
        Per-channel SOS bank of shape ``(num_bands, 6, num_delays)`` (the
        canonical SOS bank layout) where ``num_bands = 11`` (flat + low-shelf
        + 8 bandpass + high-shelf). All sections are normalised so ``a[0] = 1``.
    """
    rt = np.asarray(rt, dtype=float).ravel()
    delays = np.asarray(delays, dtype=float).ravel()

    target_g = rt_to_slope(rt, fs)  # dB / sample (negative)

    num_delays = len(delays)
    prototype_sos, _ = design_geq(target_g * delays[0], fs=fs)
    num_bands = prototype_sos.shape[0]

    sos_out = np.zeros((num_bands, 6, num_delays))
    for i, delay in enumerate(delays):
        opt_sos, _ = design_geq(target_g * delay, fs=fs)
        opt_sos = opt_sos / opt_sos[:, 3:4]  # normalise a0 = 1
        sos_out[:, :, i] = opt_sos

    return sos_out
