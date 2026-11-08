"""Impulse responses of delay state-space (DSS) systems."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.translate.dss_to_td import dss_to_td


def dss_to_impz(
    delays: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    ir_len: int,
) -> np.ndarray:
    """
    Compute MIMO impulse response from delay state-space (DSS) representation.

    Runs one simulation per input channel (Dirac at t=0 on that channel only)
    and stacks the results into a single array. A :class:`~pyFDN.DSS` unpacks
    straight in: ``dss_to_impz(*dss, ir_len)``. With filters, build the ``td``
    graph instead: ``dss_to_td(*dss, **filters).impulse_response(ir_len)``.

    Parameters
    ----------
    delays : list or array
        Delay lengths in samples
    A, B, C, D : array-like
        Delay state-space matrices.
    ir_len : int
        Length of impulse response in samples

    Returns
    -------
    impulse_response : ndarray
        Shape [ir_len, num_outputs, num_inputs]
    """
    return dss_to_td(delays, A, B, C, D).impulse_response(ir_len)
