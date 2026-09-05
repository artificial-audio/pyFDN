# dss_to_tf.py
"""
Convert delay state-space (DSS) FDN to matrix transfer function form.

Similar to ss2tf but with per-delay lengths. Supports multiple inputs and outputs.
Derived from block matrix determinant (see determinant#Block_matrices on Wikipedia).
Matches the reference dss2tf.m.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.math import general_char_poly


def dss_to_tf(
    delays: ArrayLike,
    A: np.ndarray,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """
    From a delay state-space (DSS) system to a transfer function matrix.

    No ``build_to_tf`` counterpart exists: the result does not depend on
    ``fs`` or an :class:`~pyFDN.FDNBuild`'s filter hooks, so pass
    ``build.delays`` and ``build.A``/``B``/``C``/``D`` directly.

    Parameters
    ----------
    delays : array-like
        Delays in samples, shape (N,).
    A : ndarray
        Feedback matrix, shape (N, N) or (N, N, order) for polynomial matrix.
    B : array-like
        Input gains, shape (N, num_input).
    C : array-like
        Output gains, shape (num_output, N).
    D : array-like
        Direct gains, shape (num_output, num_input).

    Returns
    -------
    tfB : ndarray
        Numerator of transfer function matrix, shape (num_output, num_input, order).
    tfA : ndarray
        Denominator polynomial (common), shape (order,) in z^{-1} convention.
    """
    delays = np.asarray(delays, dtype=int).ravel()
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    tfA = general_char_poly(delays, A)
    num_output = C.shape[0]
    num_input = B.shape[1]

    # Compute all numerators and get max length (can exceed len(tfA) for 2D A)
    numerators = []
    for i_out in range(num_output):
        for i_in in range(num_input):
            num = _numerator(delays, A, B[:, i_in], C[i_out, :], D[i_out, i_in], tfA)
            numerators.append((i_out, i_in, num))
    order_b = max(len(num) for (_, _, num) in numerators)
    order = max(len(tfA), order_b)
    tfB = np.zeros((num_output, num_input, order), dtype=float)
    for i_out, i_in, num in numerators:
        tfB[i_out, i_in, : len(num)] = num
    return tfB, tfA


def _numerator(
    delays: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: float,
    tfA: np.ndarray,
) -> np.ndarray:
    """Single (output, input) channel numerator: (d+1)*tfA - GCP(delays, A + b*c)."""
    A_mod = A.copy()
    if A.ndim == 2:
        A_mod += np.outer(b, c)
    else:
        # Polynomial A: rank-1 modification is a constant (degree-0) matrix
        A_mod[:, :, 0] += np.outer(b, c)
    gcp = general_char_poly(delays, A_mod)
    n_len = max(len(tfA), len(gcp))
    num = np.zeros(n_len)
    num[: len(tfA)] = (d + 1.0) * tfA
    num[: len(gcp)] -= gcp
    return num
