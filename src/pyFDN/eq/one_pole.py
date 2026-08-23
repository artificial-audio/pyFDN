"""One-pole absorption filter design.

The cheapest of the three designs in this package: a single pole, two degrees of
freedom (DC and Nyquist), and no zero to place. Like the first-order shelf it is
specified by its two endpoints, but its transition is not adjustable -- where
:func:`pyFDN.first_order_absorption` lets you put the crossover, this one puts it
wherever the endpoint ratio implies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..auxiliary.utils import db_to_lin
from ._backend import array_namespace

#: Number of design parameters: the value at DC and the value at Nyquist.
N_ENDPOINTS = 2
#: Number of biquad sections the design produces.
N_SECTIONS = 1


def one_pole_sos(h_dc: Any, h_ny: Any) -> Any:
    """Endpoint gains (linear) to a one-section SOS bank.

    Written against whichever array namespace its arguments come from -- see
    :mod:`pyFDN.eq._backend`.

    The pole is ``-a1`` with ``a1 = (1 - r) / (1 + r)`` and ``r = h_dc / h_ny``,
    so any pair of positive endpoint gains gives ``|a1| < 1``: the design is
    unconditionally stable.

    Parameters
    ----------
    h_dc, h_ny : array_like or torch.Tensor
        Linear-magnitude gains at DC and at Nyquist, scalars or one per channel.

    Returns
    -------
    np.ndarray or torch.Tensor
        SOS bank of shape ``(1, 6) + h_dc.shape``, normalized to ``a0 = 1``.
    """
    xp = array_namespace(h_dc)
    if xp is np:
        h_dc = np.asarray(h_dc, dtype=float)
        h_ny = np.asarray(h_ny, dtype=float)

    r = h_dc / h_ny
    a1 = (1.0 - r) / (1.0 + r)
    b0 = (1.0 - a1) * h_ny

    zero, one = xp.zeros_like(b0), xp.ones_like(b0)
    return xp.stack([b0, zero, zero, one, a1, zero], 0)[None]


def one_pole_absorption(
    rt_dc: float, rt_ny: float, delays: ArrayLike, fs: float
) -> np.ndarray:
    """Design one-pole absorption filters according to specified reverb time.

    Returns a one-section per-channel SOS bank of shape ``(1, 6, N)`` (the
    canonical SOS bank layout; section rows are ``[b0, b1, b2, a0, a1, a2]``).
    """
    # lazy: auxiliary.acoustics re-exports the designs in this package, so
    # importing it at module level would cycle.
    from ..auxiliary.acoustics import rt_to_slope

    # ravel: the SOS bank's channel axis is flat, so a delays array with any
    # shape at all designs one filter per element, in order.
    delays_arr = np.asarray(delays, dtype=float).ravel()
    h_dc = db_to_lin(delays_arr * rt_to_slope(rt_dc, fs))
    h_ny = db_to_lin(delays_arr * rt_to_slope(rt_ny, fs))
    return one_pole_sos(h_dc, h_ny)
