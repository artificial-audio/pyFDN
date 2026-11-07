"""One-shot processing of a delay state-space (DSS) system."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.translate.dss_to_td import dss_to_td


def process_dss(
    delays: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    signal: ArrayLike,
) -> np.ndarray:
    """Process a signal through a delay state-space system.

    One-shot form of ``dss_to_td(...).process_signal(signal)``: a fresh
    :mod:`pyFDN.td` graph is built for every call. The system comes first, as
    in :func:`pyFDN.dss_to_impz` and :func:`scipy.signal.lfilter`, so a
    :class:`~pyFDN.DSS` unpacks straight in: ``process_dss(*dss, x)``. With
    filters, build the graph:
    ``dss_to_td(*dss, **filters).process_signal(x, squeeze=True)``.

    Parameters
    ----------
    delays
        Positive delay lengths in samples, shape ``(N,)``.
    A, B, C, D
        Feedback, input, output, and direct gain matrices.
    signal
        Input of shape ``(num_samples,)`` or ``(num_samples, num_inputs)``.

    Returns
    -------
    np.ndarray
        Processed signal with singleton dimensions removed.
    """
    return dss_to_td(delays, A, B, C, D).process_signal(signal, squeeze=True)
