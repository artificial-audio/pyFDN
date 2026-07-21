"""FDN processing functions."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.dsp.dfilt_matrix import FIRMatrixFilter
from pyFDN.dsp.feedback_delay import FeedbackDelay


def process_fdn(
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
    """Simulate the feedback delay network using block processing.

    Recursion per block (same ordering as the MATLAB ``processFDN``):
    delay output -> optional post-delay filter -> output gains C, and in the feedback
    path: absorbed delay output -> feedback matrix A -> optional post-matrix filter -> + B input.
    The wet signal is processed with an optional post-output filter before being added to the direct signal.

    Parameters
    ----------
    input_signal : array
        Input of shape (num_samples,) or (num_samples, num_inputs).
    delays : array
        Delay lengths in samples, shape (N,).
    A : array
        Feedback matrix: static (N, N) or FIR polynomial (N, N, order) in
        z^{-1} convention.
    B, C, D : array
        Static input, output, and direct gains.
    post_delay : object or None, optional
        An optional filter applied to the delay output before feedback processing.
        Must implement a `filter` method that accepts and processes the delay output.
    post_matrix : object or None, optional
        An optional filter applied to the feedback signal after the feedback matrix
        multiplication. Must implement a `filter` method that accepts and processes the feedback signal.
    post_output : object or None, optional
        An optional filter applied to the wet signal (output signal) before it is
        added to the direct signal. Must implement a `filter` method that accepts and processes the wet signal.

    Returns
    -------
    output : ndarray
        Shape (num_samples, num_outputs), squeezed.
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
        feedback_filter: FIRMatrixFilter | None = FIRMatrixFilter(A_mat)
    elif A_mat.ndim == 2:
        feedback_filter = None
    else:
        raise ValueError("A must be a 2-D (static) or 3-D (FIR) matrix")

    max_block_size = min(int(2**12), int(np.min(delays_arr)))
    delay_bank = FeedbackDelay(delays_arr, max_block_size)

    num_samples = x.shape[0]
    num_outputs = C_mat.shape[0]
    output = np.zeros((num_samples, num_outputs), dtype=float)

    start = 0
    while start < num_samples:
        block_size = min(max_block_size, num_samples - start)
        block_in = x[start : start + block_size, :]

        delay_out = delay_bank.get_values(block_size)  # (block, N)
        if post_delay is not None:
            delay_out = post_delay.filter(delay_out)

        if feedback_filter is not None:
            feedback = feedback_filter.filter(delay_out)
        else:
            feedback = delay_out @ A_mat.T
        if post_matrix is not None:
            feedback = post_matrix.filter(feedback)

        wet_signal = delay_out @ C_mat.T
        if post_output is not None:
            wet_signal = post_output.filter(wet_signal)

        delay_bank.set_values(block_in @ B_mat.T + feedback)

        output[start : start + block_size] = wet_signal + block_in @ D_mat.T
        delay_bank.advance(block_size)
        start += block_size

    return output.squeeze()
