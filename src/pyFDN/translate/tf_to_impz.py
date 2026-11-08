# tf_to_impz.py
"""
Matrix transfer function to impulse response.

Given numerator matrix tfB and common denominator tfA (z^{-1} convention),
compute the MIMO impulse response by filtering a unit impulse for each input channel.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import lfilter


def tf_to_impz(
    tfB: ArrayLike,
    tfA: ArrayLike,
    ir_len: int,
) -> np.ndarray:
    """
    Compute MIMO impulse response from matrix transfer function (numerator/denominator).

    Parameters
    ----------
    tfB : array-like
        Numerator polynomials, shape (num_output, num_input, order) in z^{-1} convention.
    tfA : array-like
        Common denominator polynomial, shape (order,) in z^{-1} convention.
    ir_len : int
        Length of impulse response in samples.

    Returns
    -------
    impulse_response : ndarray
        Shape (ir_len, num_output, num_input). Channel (:, o, i) is the response
        at output o to a unit impulse at input i.
    """
    tfB = np.asarray(tfB, dtype=float)
    tfA = np.asarray(tfA, dtype=float).ravel()
    num_output, num_input, _ = tfB.shape

    impulse = np.zeros(ir_len)
    impulse[0] = 1.0

    ir = np.zeros((ir_len, num_output, num_input), dtype=float)
    for o in range(num_output):
        for i in range(num_input):
            b = np.asarray(tfB[o, i, :], dtype=float).ravel()
            ir[:, o, i] = lfilter(b, tfA, impulse)
    return ir
