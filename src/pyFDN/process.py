"""FDN processing functions."""
from __future__ import annotations
import numpy as np
from typing import Optional
from numpy.typing import ArrayLike
from pyFDN.dsp.filter_matrix import FilterMatrix
from pyFDN.dsp.feedback_delay import FeedbackDelay
from pyFDN.auxiliary.filters import ZTF

def process_fdn(
    input_signal: ArrayLike,
    delays: ArrayLike,
    feedback_matrix: ArrayLike | ZTF,
    input_gain: ArrayLike,
    output_gain: ArrayLike,
    direct: ArrayLike,
    *,
    absorption_filters: Optional[ZTF | ArrayLike] = None,
    extra_matrix: Optional[ArrayLike | ZTF] = None,
) -> np.ndarray:
    """Simulate the feedback delay network using block processing.

    Input is (num_samples, num_inputs); output is (num_samples, num_outputs).
    """
    x = np.asarray(input_signal, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if x.ndim != 2:
        raise ValueError("Input signal must be a 1-D or 2-D array")

    delays_arr = np.asarray(delays, dtype=int).reshape(-1)
    if delays_arr.ndim != 1:
        raise ValueError("Delays must be a 1-D array")
    if np.any(delays_arr <= 0):
        raise ValueError("Delays must be positive integers")

    feedback = FilterMatrix.from_data(feedback_matrix)
    inputs = FilterMatrix.from_data(input_gain)
    outputs = FilterMatrix.from_data(output_gain)
    absorption = (
        FilterMatrix.from_data(absorption_filters, is_diagonal=True)
        if absorption_filters is not None
        else None
    )
    extra = FilterMatrix.from_data(extra_matrix) if extra_matrix is not None else None

    direct_matrix = np.asarray(direct, dtype=float)
    if direct_matrix.ndim != 2:
        raise ValueError("Direct matrix must be 2-D")
    if direct_matrix.shape[1] != x.shape[1]:
        raise ValueError("Direct matrix column count must match number of inputs")

    max_block_size = min(int(2 ** 12), int(np.min(delays_arr)))
    delay_bank = FeedbackDelay(delays_arr, max_block_size)

    num_samples = x.shape[0]
    num_outputs = outputs.output_channels
    output = np.zeros((num_samples, num_outputs), dtype=float)

    start = 0
    while start < num_samples:
        block_size = min(max_block_size, num_samples - start)
        block_slice = slice(start, start + block_size)
        block_in = x[block_slice, :]

        delay_out = delay_bank.get_values(block_size)
        if absorption is not None:
            delay_out = absorption.filter(delay_out)

        feedback_block = feedback.filter(delay_out)
        if extra is not None:
            feedback_block = extra.filter(feedback_block)

        delay_input = inputs.filter(block_in) + feedback_block
        delay_bank.set_values(delay_input)

        block_out = outputs.filter(delay_out)
        output[block_slice, :] = block_out + block_in @ direct_matrix.T

        delay_bank.advance(block_size)
        start += block_size

    return output.squeeze()
