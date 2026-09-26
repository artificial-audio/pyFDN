"""
Allpass tests for FDNs (allpass, uniallpass, paraunitary).

The allpass structure builders live in :mod:`pyFDN.generate.structures` and
are re-exported here so ``pyFDN.allpass.series_allpass`` keeps working.

Based on Poletti (1995) and "Allpass Feedback Delay Networks" by Sebastian J. Schlecht.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_discrete_lyapunov

from pyFDN.auxiliary.math import general_char_poly
from pyFDN.auxiliary.utils import is_almost_zero
from pyFDN.generate.structures import nested_allpass as nested_allpass
from pyFDN.generate.structures import poletti_allpass as poletti_allpass
from pyFDN.generate.structures import series_allpass as series_allpass

# How far inside the unit circle the eigenvalues of A must sit for the discrete
# Lyapunov equation solved by is_uniallpass to be well posed.
_STABILITY_TOL = 1e-9


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
        All-NaN when A is not strictly stable, see the note below.

    Notes
    -----
    The Lyapunov equation only has a (unique, finite) solution when A is
    strictly stable.  If A itself is lossless -- as in the allpass-in-FDN
    structure, where the feedback matrix has all its eigenvalues on the unit
    circle -- no such P exists, and the linear system scipy solves is exactly
    singular.  Rather than let that surface as an ill-conditioned solve (which
    raises or returns garbage depending on the LAPACK build), the spectral
    radius is checked up front and the system reported as not uniallpass.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    # P exists only for a strictly stable A; on the unit circle the Lyapunov
    # operator P -> A P A' - P is singular (eigenvalue pairs with λi·λj = 1).
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A)))) if A.size else 0.0
    if spectral_radius >= 1.0 - _STABILITY_TOL:
        warnings.warn(
            f"A has spectral radius {spectral_radius:.6g} >= 1; the discrete "
            "Lyapunov equation has no finite solution, so the system is not "
            "uniallpass.",
            stacklevel=2,
        )
        return False, np.full_like(A, np.nan)
    # P = dlyap(A, B @ B')
    P = solve_discrete_lyapunov(A, B @ B.T)
    # Check P is diagonal
    off_diag = P - np.diag(np.diag(P))
    if not is_almost_zero(off_diag, tol=tol):
        warnings.warn("P is not diagonal; system is not uniallpass.", stacklevel=2)
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
    delays: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    tol: float = 1e-9,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Test whether the delay state-space system is allpass.

    Checks that the determinant transfer function has numerator = reversed(denominator)
    (up to sign). See "Allpass Feedback Delay Networks" by Sebastian J. Schlecht.

    Parameters
    ----------
    delays : array-like
        Delay lengths (samples), length N.
    A, B, C, D : array-like
        Delay state-space matrices.
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
    A_eff = A - B @ np.linalg.solve(D, C)
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
    is_p = is_almost_zero(R_out - I_out, tol=tol) and is_almost_zero(
        R_in - I_in, tol=tol
    )
    return is_p, R_out, max_off
