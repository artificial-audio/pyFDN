# dss_to_impz.py
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.process import process_dss, process_fdn

if TYPE_CHECKING:
    from pyFDN.build import FDNBuild


def dss_to_impz(
    ir_len: int,
    delays: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
) -> np.ndarray:
    """
    Compute MIMO impulse response from delay state-space (DSS) representation.

    Runs one simulation per input channel (Dirac at t=0 on that channel only)
    and stacks the results into a single array.

    Parameters
    ----------
    ir_len : int
        Length of impulse response in samples
    delays : list or array
        Delay lengths in samples
    A, B, C, D : array-like
        Delay state-space matrices (static, numeric only).
        For a complete :class:`pyFDN.FDNBuild` with filter hooks, use
        :func:`pyFDN.build_to_impz`.

    Returns
    -------
    impulse_response : ndarray
        Shape [ir_len, num_outputs, num_inputs]
    """
    num_inputs = np.asarray(B).shape[1]
    out_list = []

    for j in range(num_inputs):
        input_signal = np.zeros((ir_len, num_inputs))
        input_signal[0, j] = 1.0
        out_j = process_dss(input_signal, delays, A, B, C, D)
        if out_j.ndim == 1:
            out_j = out_j[:, np.newaxis]
        out_list.append(out_j)

    return np.stack(out_list, axis=-1)


def build_to_impz(build: FDNBuild, ir_len: int) -> np.ndarray:
    """Render an :class:`FDNBuild` to a time-domain impulse response.

    Time-domain sibling of the FLAMO render path (:func:`pyFDN.build_to_flamo`
    -> :func:`pyFDN.flamo_time_response`): runs one :func:`pyFDN.process_fdn`
    graph render per input channel (a Dirac on that channel). The graph contains
    the build's three filter hooks as :class:`pyFDN.td.SOSBank` nodes:
    ``post_delay`` on the delay output, ``post_matrix`` on the feedback path,
    and ``post_output`` on the wet signal. Unlike the FFT-based FLAMO render
    this does not time-alias, so a long or near-lossless decay is rendered
    faithfully up to ``ir_len``.

    Extends :func:`dss_to_impz` (numeric state-space only) with the build's
    filter hooks.

    Parameters
    ----------
    build : FDNBuild
        Complete FDN parameters.
    ir_len : int
        Impulse-response length in samples.

    Returns
    -------
    np.ndarray
        Impulse response of shape ``(ir_len, num_outputs, num_inputs)``. Use
        ``.squeeze()`` for a 1-D array from a single-in/single-out FDN.
    """
    num_inputs = np.asarray(build.B).shape[1]

    out_list = []
    for j in range(num_inputs):
        input_signal = np.zeros((ir_len, num_inputs))
        input_signal[0, j] = 1.0
        # process_fdn creates a fresh graph, so state cannot leak between the
        # one-impulse-per-input simulations.
        out_j = process_fdn(input_signal, build)
        if out_j.ndim == 1:
            out_j = out_j[:, np.newaxis]
        out_list.append(out_j)

    return np.stack(out_list, axis=-1)
