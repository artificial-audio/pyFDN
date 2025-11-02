from __future__ import annotations
from numpy.typing import ArrayLike
import numpy as np

from pyFDN.helpers.utils import ensure_3d

def matrix_convolution(A: ArrayLike, B: ArrayLike) -> np.ndarray:
    """Matrix polynomial multiplication by convolution."""

    A_arr = ensure_3d(A)
    B_arr = ensure_3d(B)
    if A_arr.shape[1] != B_arr.shape[0]:
        raise ValueError("Inner dimensions must agree")

    m, n, order_a = A_arr.shape
    _, k, order_b = B_arr.shape
    result = np.zeros((m, k, order_a + order_b - 1), dtype=np.result_type(A_arr, B_arr))

    for row in range(m):
        for col in range(k):
            acc = np.zeros(order_a + order_b - 1, dtype=result.dtype)
            for inter in range(n):
                acc += np.convolve(A_arr[row, inter, :], B_arr[inter, col, :])
            result[row, col, :] = acc
    return result
