"""
Allpass FDN helpers (Poletti MIMO reverberator, uniallpass test, etc.).

Based on Poletti (1995) and "Allpass Feedback Delay Networks" by Sebastian J. Schlecht.
"""
from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_discrete_lyapunov

from pyFDN.auxiliary.math import general_char_poly
from pyFDN.generate.is_almost_zero import is_almost_zero


def poletti_allpass(g: float, U: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create Poletti's MIMO unitary reverberator (allpass FDN).

    From Poletti, M. (1995). A unitary reverberator for reduced colouration
    in assisted reverberation systems. INTER-NOISE and NOISE-CON, 5, 1223–1232.

    Parameters
    ----------
    g : float
        Scalar feedback gain (e.g. 0.7).
    U : ndarray (N, N)
        Unitary (orthogonal) feedback matrix.

    Returns
    -------
    A, B, C, D : ndarray
        Delay state-space matrices: A = -g*U, B = (1+g)*I, C = (1-g)*U, D = g*I.
    """
    U = np.asarray(U, dtype=float)
    N = U.shape[0]
    A = -g * U
    B = (1 + g) * np.eye(N)
    C = (1 - g) * U
    D = g * np.eye(N)
    return A, B, C, D


def is_uniallpass(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    tol: float = 1e-9,
) -> tuple[bool, np.ndarray]:
    """
    Test whether the FDN is uniallpass (lossless with a diagonal Lyapunov matrix).

    See Michaletzky, G. Factorization of discrete-time all-pass functions;
    and "Allpass Feedback Delay Networks" by Sebastian J. Schlecht.

    Parameters
    ----------
    A, B, C, D : array-like
        Delay state-space matrices (feedback, input gain, output gain, direct).
    tol : float
        Tolerance for zero and diagonal checks.

    Returns
    -------
    is_a : bool
        True if the system is uniallpass.
    P : ndarray
        Solution of discrete Lyapunov A P A' - P + B B' = 0; diagonal if uniallpass.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    # P = dlyap(A, B @ B')
    P = solve_discrete_lyapunov(A, B @ B.T)
    # Check P is diagonal
    off_diag = P - np.diag(np.diag(P))
    if not is_almost_zero(off_diag, tol=tol):
        warnings.warn("P is not diagonal; system is not uniallpass.")
        return False, P
    # Test: PP - U @ PP @ U' ≈ 0 with PP = blkdiag(P, I)
    U = np.block([[A, B], [C, D]])
    nd = D.shape[0]
    PP = np.zeros_like(U)
    n = A.shape[0]
    PP[:n, :n] = P
    PP[n:, n:] = np.eye(nd)
    test = PP - U @ PP @ U.T
    is_a = is_almost_zero(test, tol=tol)
    return is_a, P


def is_allpass(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    delays: ArrayLike,
    tol: float = 1e-9,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Test whether the delay state-space system is allpass.

    Checks that the determinant transfer function has numerator = reversed(denominator)
    (up to sign). See "Allpass Feedback Delay Networks" by Sebastian J. Schlecht.

    Parameters
    ----------
    A, B, C, D : array-like
        Delay state-space matrices.
    delays : array-like
        Delay lengths (samples), length N.
    tol : float
        Tolerance for coefficient comparison.

    Returns
    -------
    is_a : bool
        True if allpass.
    den : ndarray
        Denominator polynomial (z^{-1} ordering).
    num : ndarray
        Numerator polynomial (z^{-1} ordering).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    delays = np.asarray(delays, dtype=int).ravel()
    D_inv = np.linalg.inv(D)
    A_eff = A - B @ D_inv @ C
    den = general_char_poly(delays, A)
    num = general_char_poly(delays, A_eff) * np.linalg.det(D)
    # Allpass: numerator equals reversed(denominator) up to sign
    den_rev = np.flip(den)
    max_len = max(len(den_rev), len(num))
    den_rev = np.pad(den_rev.astype(float), (0, max_len - len(den_rev)))
    num_pad = np.pad(num.astype(float), (0, max_len - len(num)))
    if np.abs(num_pad[-1]) > 1e-15:
        sign = np.sign(num_pad[-1])
        diff = den_rev - num_pad * sign
    else:
        diff = den_rev - num_pad
    is_a = is_almost_zero(diff, tol=tol)
    return is_a, den, num


def series_allpass(g: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create Schroeder's series allpass FDN (SISO).

    Iterative series connection of feedforward/back allpass filters (same as
    seriesAllpass.m). Each stage appends one delay line via seriesFDNinAllpass.
    From Schroeder & Logan (1961). "Colorless" artificial reverberation.
    IRE Trans. Audio AU-9, 209–214. See "Allpass Feedback Delay Networks", Schlecht.

    Parameters
    ----------
    g : array-like, shape (N,)
        Per-section gains (e.g. in (0, 1)).

    Returns
    -------
    A : ndarray (N, N)
        Feedback matrix.
    B : ndarray (N, 1)
        Input gain (column vector).
    C : ndarray (1, N)
        Output gain (row vector).
    D : ndarray (1, 1)
        Direct gain (scalar).
    """

    def series_fdn_in_allpass(
        allpass_gain: float,
        matrix: np.ndarray,
        input_gain: np.ndarray,
        output_gain: np.ndarray,
        direct: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Series connection of an FDN with a feedforward/back allpass (seriesFDNinAllpass.m)."""
        g2 = 1 - allpass_gain**2
        s_matrix = np.block([
            [matrix, np.zeros((matrix.shape[0], 1))],
            [output_gain * g2, np.array([[allpass_gain]])],
        ])
        s_input_gain = np.vstack([input_gain, direct * g2])
        s_output_gain = np.hstack([-allpass_gain * output_gain, np.array([[1.0]])])
        s_direct = -allpass_gain * direct
        return s_matrix, s_input_gain, s_output_gain, s_direct

    g = np.asarray(g, dtype=float).ravel()
    N = len(g)
    if N == 0:
        raise ValueError("g must have at least one element")
    matrix = np.array([[g[0]]])
    input_gain = np.array([[1 - g[0] ** 2]])
    output_gain = np.array([[1.0]])
    direct = np.array([[-g[0]]])
    for it in range(1, N):
        matrix, input_gain, output_gain, direct = series_fdn_in_allpass(
            g[it], matrix, input_gain, output_gain, direct
        )
    return matrix, input_gain, output_gain, direct


def nested_allpass(g: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create Gardner's nested allpass FDN (SISO).

    Iteratively nests a feedforward/back allpass around the previous FDN.
    From Gardner, W. G. (1992). A real-time multichannel room simulator.
    J. Acoust. Soc. Am. 92, 1–23. See "Allpass Feedback Delay Networks", Schlecht.

    Parameters
    ----------
    g : array-like, shape (N,)
        Feedforward/back gains for each nesting stage.

    Returns
    -------
    A : ndarray (N, N)
        Feedback matrix.
    B : ndarray (N, 1)
        Input gain (column vector).
    C : ndarray (1, N)
        Output gain (row vector).
    D : ndarray (1, 1)
        Direct gain (scalar).
    """
    g = np.asarray(g, dtype=float).ravel()
    N = len(g)
    if N == 0:
        raise ValueError("g must have at least one element")
    # Initial: single allpass stage
    matrix = np.array([[g[0]]])
    input_gain = np.array([[1 - g[0] ** 2]])
    output_gain = np.array([[1.0]])
    direct = np.array([[-g[0]]])
    for it in range(1, N):
        ga = g[it]
        # [matrix, input_gain; output_gain*ga, direct*ga]
        n_matrix = np.block([
            [matrix, input_gain],
            [output_gain * ga, direct * ga],
        ])
        n_input_gain = np.vstack([np.zeros_like(input_gain), np.array([[1 - ga**2]])])
        n_output_gain = np.hstack([output_gain, direct])
        n_direct = np.array([[-ga]])
        matrix = n_matrix
        input_gain = n_input_gain
        output_gain = n_output_gain
        direct = n_direct
    return matrix, input_gain, output_gain, direct


def is_paraunitary(
    ir: ArrayLike,
    tol: float = 1e-9,
) -> tuple[bool, np.ndarray, float]:
    """
    Test whether a MIMO impulse response is paraunitary (lossless).

    For real IR matrix H(t), checks that sum_t H(t) H(t)' = I (output correlation)
    and sum_t H(t)' H(t) = I (input correlation).

    Parameters
    ----------
    ir : ndarray, shape (ir_len, n_out, n_in)
        Impulse response [time, output, input].
    tol : float
        Tolerance for identity check.

    Returns
    -------
    is_p : bool
        True if paraunitary.
    test_matrix : ndarray
        Output correlation matrix (n_out, n_out); should be identity.
    max_off_diagonal : float
        Max absolute off-diagonal value in test_matrix.
    """
    ir = np.asarray(ir, dtype=float)
    # ir: (T, n_out, n_in)
    n_out = ir.shape[1]
    n_in = ir.shape[2]
    # R_out = sum_t ir[t] @ ir[t].T  (n_out, n_out)
    R_out = np.einsum("tij,tkj->ik", ir, ir)
    R_in = np.einsum("tji,tjk->ik", ir, ir)
    I_out = np.eye(n_out)
    I_in = np.eye(n_in)
    off_out = R_out - np.diag(np.diag(R_out))
    off_in = R_in - np.diag(np.diag(R_in))
    max_off = float(max(np.max(np.abs(off_out)), np.max(np.abs(off_in))))
    is_p = is_almost_zero(R_out - I_out, tol=tol) and is_almost_zero(R_in - I_in, tol=tol)
    return is_p, R_out, max_off
