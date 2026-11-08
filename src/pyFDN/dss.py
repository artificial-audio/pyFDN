"""The delay state-space (DSS) matrices of an FDN."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class DSS(NamedTuple):
    """Delay state-space system: the standard matrices of an FDN, nothing else.

    ``delays`` holds one length per delay line, in samples. ``A`` is the
    ``(N, N)`` feedback matrix, ``B`` the ``(N, inputs)`` input gains, ``C``
    the ``(outputs, N)`` output gains and ``D`` the ``(outputs, inputs)``
    direct path. A filter (FIR) feedback matrix is not a DSS matrix; it goes
    into a graph's ``post_matrix`` hook, as a ``td.MatrixFIR`` or a
    :func:`pyFDN.fir_matrix_module`.

    The field order is the argument order of the ``dss_to_*`` functions, so a
    system unpacks straight into them. Filters and the sample rate are not part
    of the system; pass them alongside, e.g.
    ``pyFDN.dss_to_td(*dss, post_delay=sos)``.
    """

    delays: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray


__all__ = ["DSS"]
