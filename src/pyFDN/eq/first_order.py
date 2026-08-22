"""First-order shelving filter design.

A first-order shelf has exactly two degrees of freedom once its crossover is
fixed -- its value at DC and its value at Nyquist -- so those two numbers are
the whole design. What lies between them is not free, which is the point: this
is the one-biquad end of the complexity scale that :mod:`pyFDN.eq.design_geq`
occupies the other end of.

Reference:
    Jot, "Proportional Parametric Equalizers - Application to Digital
    Reverberation and Environmental Audio Processing," AES Conv. 2015.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..auxiliary.utils import db_to_lin
from ._backend import array_namespace

#: Number of design parameters: the value at DC and the value at Nyquist.
N_ENDPOINTS = 2
#: Number of biquad sections the design produces.
N_SECTIONS = 1


def shelf_crossover_omega(fs: float, crossover_frequency: float | None = None) -> float:
    """Crossover in radians, defaulted and clamped.

    The default is ``fs/8``, the midpoint of the bilinear-warped frequency axis.
    Anything above ``fs/5`` is clamped, since the pole leaves the unit circle at
    ``fs/4``.
    """
    f_c = fs / 8.0 if crossover_frequency is None else float(crossover_frequency)
    return min(f_c, fs / 5.0) / fs * 2.0 * math.pi


def first_order_shelf_sos(h_dc: Any, h_ny: Any, omega: float) -> Any:
    """Endpoint gains (linear) to a one-section SOS bank.

    Written against whichever array namespace its arguments come from, so a
    torch pair of endpoints gives a torch SOS bank with gradients reaching back
    to them -- see :mod:`pyFDN.eq._backend`.

    Parameters
    ----------
    h_dc, h_ny : array_like or torch.Tensor
        Linear-magnitude gains at DC and at Nyquist, scalars or one per channel.
    omega : float
        Crossover in radians, from :func:`shelf_crossover_omega`.

    Returns
    -------
    np.ndarray or torch.Tensor
        SOS bank of shape ``(1, 6) + h_dc.shape``, normalized to ``a0 = 1``.
    """
    xp = array_namespace(h_dc)
    if xp is np:
        h_dc = np.asarray(h_dc, dtype=float)
        h_ny = np.asarray(h_ny, dtype=float)

    t = math.tan(float(omega))
    sqrt_k = xp.sqrt(h_dc / h_ny)

    a0 = t / sqrt_k + 1.0
    b0 = (t * sqrt_k + 1.0) * h_ny / a0
    b1 = (t * sqrt_k - 1.0) * h_ny / a0
    a1 = (t / sqrt_k - 1.0) / a0

    zero, one = xp.zeros_like(b0), xp.ones_like(b0)
    return xp.stack([b0, b1, zero, one, a1, zero], 0)[None]


def first_order_absorption(
    rt_dc: float,
    rt_ny: float,
    delays: ArrayLike,
    fs: float,
    crossover_frequency: float | None = None,
) -> np.ndarray:
    """Design first-order shelving absorption filters for a target decay.

    Each delay line gets a first-order shelving filter whose gain matches the
    target decay (rt_dc at DC, rt_ny at Nyquist) for its delay length, with the
    shelf transition at crossover_frequency.

    Reference: Jot, J. M., "Proportional parametric equalizers - Application to
    digital reverberation and environmental audio processing", AES 2015.

    Parameters
    ----------
    rt_dc : float
        Reverberation time in seconds at DC.
    rt_ny : float
        Reverberation time in seconds at Nyquist.
    delays : array-like
        Delay lengths in samples, one per channel.
    fs : float
        Sampling rate in Hz.
    crossover_frequency : float, optional
        Shelf crossover frequency in Hz. Defaults to fs/8, the midpoint of the
        warped (bilinear) frequency axis. Values above fs/5 are clamped to fs/5
        since a too high crossover leads to an unstable filter (fs/4 is the limit).

    Returns
    -------
    np.ndarray
        One-section per-channel SOS bank of shape ``(1, 6, N)`` (the canonical
        SOS bank layout); section rows are ``[b0, b1, b2, a0, a1, a2]``
        (b2 = a2 = 0 for these first-order filters).
    """
    # lazy: auxiliary.acoustics re-exports the designs in this package, so
    # importing it at module level would cycle.
    from ..auxiliary.acoustics import rt_to_slope

    # ravel: the SOS bank's channel axis is flat, so a delays array with any
    # shape at all designs one filter per element, in order.
    delays_arr = np.asarray(delays, dtype=float).ravel()
    h_dc = db_to_lin(delays_arr * rt_to_slope(rt_dc, fs))
    h_ny = db_to_lin(delays_arr * rt_to_slope(rt_ny, fs))
    return first_order_shelf_sos(
        h_dc, h_ny, shelf_crossover_omega(fs, crossover_frequency)
    )


def first_order_shelving_eq(
    db_dc: ArrayLike,
    db_nyquist: ArrayLike,
    fs: float,
    crossover_frequency: float | None = None,
) -> np.ndarray:
    """Design first-order shelving EQ filters from gains in dB at DC and Nyquist.

    Unlike :func:`first_order_absorption` (whose gains are derived from a
    reverberation time and a delay length), the shelf endpoints are specified
    directly as decibel gains. Useful as a per-output tone correction (post EQ).

    Parameters
    ----------
    db_dc : array-like
        Gain in dB at DC, scalar or one value per channel.
    db_nyquist : array-like
        Gain in dB at Nyquist, scalar or one value per channel. Broadcast
        against ``db_dc`` to a common number of channels.
    fs : float
        Sampling rate in Hz.
    crossover_frequency : float, optional
        Shelf crossover frequency in Hz. Defaults to fs/8; clamped to fs/5.

    Returns
    -------
    np.ndarray
        One-section per-channel SOS bank of shape ``(1, 6, N)`` (canonical SOS
        bank layout); section rows are ``[b0, b1, b2, a0, a1, a2]``.
    """
    db_dc_arr, db_ny_arr = np.broadcast_arrays(
        np.asarray(db_dc, dtype=float).ravel(),
        np.asarray(db_nyquist, dtype=float).ravel(),
    )
    return first_order_shelf_sos(
        db_to_lin(db_dc_arr),
        db_to_lin(db_ny_arr),
        shelf_crossover_omega(fs, crossover_frequency),
    )
