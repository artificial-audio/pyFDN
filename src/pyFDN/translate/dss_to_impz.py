# dss_to_impz.py
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.process import process_fdn

if TYPE_CHECKING:
    from pyFDN.generate.fdn_matrix_gallery import FDNBuild


def dss_to_impz(
    ir_len: int,
    delays: ArrayLike,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
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
    A, B, C, D : ndarray
        Delay state-space matrices (static, numeric only).
        For FDNs with absorption filters use dss_to_flamo.

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
        out_j = process_fdn(input_signal, delays, A, B, C, D)
        if out_j.ndim == 1:
            out_j = out_j[:, np.newaxis]
        out_list.append(out_j)

    return np.stack(out_list, axis=-1)


def build_to_impz(build: FDNBuild, ir_len: int) -> np.ndarray:
    """Render an :class:`FDNBuild` to a time-domain impulse response.

    Time-domain sibling of the FLAMO render path (:func:`pyFDN.build_to_flamo`
    -> :func:`pyFDN.flamo_time_response`): runs one :func:`pyFDN.process_fdn`
    block simulation per input channel (a Dirac on that channel), with the
    build's per-delay-line absorption (``build.filters``, e.g. from
    :func:`pyFDN.build_set_decay`) applied inside the loop as a
    :class:`pyFDN.SOSFilterBank`. Unlike the FFT-based FLAMO render this does
    not time-alias, so a long or near-lossless decay is rendered faithfully up
    to ``ir_len``.

    Extends :func:`dss_to_impz` (numeric state-space only) with the build's
    absorption filters.

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

    Raises
    ------
    ValueError
        If ``build.post_eq`` is set: output EQ is not applied by
        :func:`process_fdn`. Use :func:`pyFDN.build_to_flamo` +
        :func:`pyFDN.flamo_time_response` for output-EQ builds.
    """
    from pyFDN.dsp.sos_filter_bank import SOSFilterBank

    if build.post_eq is not None:
        raise ValueError(
            "build_to_impz does not apply build.post_eq (output EQ); "
            "use build_to_flamo + flamo_time_response for output-EQ builds."
        )

    n = np.asarray(build.A).shape[0]
    num_inputs = np.asarray(build.B).shape[1]

    out_list = []
    for j in range(num_inputs):
        # A fresh filter bank per channel so absorption state does not leak
        # between the per-input simulations (SOSFilterBank is stateful).
        absorption = (
            SOSFilterBank(sos=build.filters, num_channels=n)
            if build.filters is not None
            else None
        )
        input_signal = np.zeros((ir_len, num_inputs))
        input_signal[0, j] = 1.0
        out_j = process_fdn(
            input_signal,
            build.delays,
            build.A,
            build.B,
            build.C,
            build.D,
            post_delay=absorption,
        )
        if out_j.ndim == 1:
            out_j = out_j[:, np.newaxis]
        out_list.append(out_j)

    return np.stack(out_list, axis=-1)
