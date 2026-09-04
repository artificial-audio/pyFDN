"""Compact DSS processing and convenience rendering of complete FDN builds."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.generate.fdn_matrix_gallery import FDNBuild
from pyFDN.td.operators import MatrixFIR, RecursionState
from pyFDN.translate.dss_to_td import build_to_td


def process_dss(
    input_signal: ArrayLike,
    delays: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    *,
    post_delay: Any | None = None,
    post_matrix: Any | None = None,
    post_output: Any | None = None,
) -> np.ndarray:
    """Process a delay state-space system using block processing.

    This is a compact, pre-wired alternative to manually assembling a
    :mod:`pyFDN.td` graph. The three optional hooks are runtime objects that
    implement ``process_block(block)``. To process an :class:`pyFDN.FDNBuild`,
    use :func:`pyFDN.build_to_td` or :func:`pyFDN.process_fdn`; those functions
    convert the build's baked SOS arrays into :class:`pyFDN.td.SOSBank` nodes.

    The recursion is

    ``delay -> post_delay -> C`` on the wet path and
    ``delay -> post_delay -> A -> post_matrix -> + B input`` in the loop.
    ``post_output`` processes the wet signal before the direct ``D`` path is
    added.

    Parameters
    ----------
    input_signal
        Input of shape ``(num_samples,)`` or ``(num_samples, num_inputs)``.
    delays
        Positive delay lengths in samples, shape ``(N,)``.
    A
        Static feedback matrix ``(N, N)`` or FIR polynomial matrix
        ``(N, N, order)`` in the ``z^-1`` convention.
    B, C, D
        Static input, output, and direct gain matrices.
    post_delay, post_matrix, post_output
        Optional runtime operators implementing ``process_block(block)``.

    Returns
    -------
    np.ndarray
        Processed signal with singleton dimensions removed.
    """
    x = np.asarray(input_signal, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if x.ndim != 2:
        raise ValueError("Input signal must be a 1-D or 2-D array")

    A_mat = np.asarray(A, dtype=float)
    B_mat = np.asarray(B, dtype=float)
    C_mat = np.asarray(C, dtype=float)
    D_mat = np.asarray(D, dtype=float)

    delays_arr = np.asarray(delays, dtype=int).reshape(-1)
    if np.any(delays_arr <= 0):
        raise ValueError("Delays must be positive integers")

    if A_mat.ndim == 3:
        feedback_filter: MatrixFIR | None = MatrixFIR(A_mat)
    elif A_mat.ndim == 2:
        feedback_filter = None
    else:
        raise ValueError("A must be a 2-D (static) or 3-D (FIR) matrix")

    max_block_size = min(int(2**12), int(np.min(delays_arr)))
    delay_bank = RecursionState(delays_arr, max_block_size)

    num_samples = x.shape[0]
    num_outputs = C_mat.shape[0]
    output = np.zeros((num_samples, num_outputs), dtype=float)

    start = 0
    while start < num_samples:
        block_size = min(max_block_size, num_samples - start)
        block_in = x[start : start + block_size, :]

        delay_out = delay_bank.get_values(block_size)
        if post_delay is not None:
            delay_out = post_delay.process_block(delay_out)

        if feedback_filter is not None:
            feedback = feedback_filter.process_block(delay_out)
        else:
            feedback = delay_out @ A_mat.T
        if post_matrix is not None:
            feedback = post_matrix.process_block(feedback)

        wet_signal = delay_out @ C_mat.T
        if post_output is not None:
            wet_signal = post_output.process_block(wet_signal)

        delay_bank.set_values(block_in @ B_mat.T + feedback)

        output[start : start + block_size] = wet_signal + block_in @ D_mat.T
        delay_bank.advance(block_size)
        start += block_size

    return output.squeeze()


def process_fdn(input_signal: ArrayLike, build: FDNBuild) -> np.ndarray:
    """Process a signal through a fresh time-domain graph built from ``build``.

    This is the one-shot convenience form of
    ``build_to_td(build).process_signal(input_signal)``. A fresh graph is built
    for every call, so delay and filter state cannot leak between independent
    renders. Use :func:`pyFDN.build_to_td` directly to process a stream over
    multiple calls to :meth:`pyFDN.td.TimeOperator.process_block` or to reset
    and reuse the graph.

    Parameters
    ----------
    input_signal
        Input of shape ``(num_samples,)`` or ``(num_samples, num_inputs)``.
    build
        Complete baked FDN configuration.

    Returns
    -------
    np.ndarray
        Processed signal with singleton dimensions removed, matching the
        historical ``process_fdn`` output convention.
    """
    return build_to_td(build).process_signal(input_signal, squeeze=True)
