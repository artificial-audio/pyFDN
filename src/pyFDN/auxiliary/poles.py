"""Pole utilities (conjugate pairing, etc.)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def reduce_conjugate_pairs(
    poles: np.ndarray | ArrayLike,
    *,
    tol_real: float = 1e-10,
    tol_pair: float = 1e-8,
    verbose: bool = False,
    strict: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group poles into real and conjugate pairs using optimal assignment.

    For real-coefficient systems, poles are either real or occur in conjugate
    pairs. This uses the linear sum assignment problem (Hungarian method): cost
    :math:`C[i,j] = |poles[j] - conj(poles[i])|`; the minimum-cost permutation
    pairs each pole with its conjugate (or itself for real poles). Then:

    - Real: assignment[i] == i and C[i,i] < tol_real (i.e. :math:`|Im(pole_i)|` small).
    - Conjugate pair: assignment[i] == j, assignment[j] == i, C[i,j] < tol_pair.
    - Unpaired: otherwise (ambiguous or numerical orphans).

    Unpaired poles are reported via ``non_paired`` AND a :class:`UserWarning`
    so callers cannot silently lose poles to imprecise pairing. Set
    ``strict=True`` to raise :class:`ValueError` instead of warning.

    Returns
    -------
    poles_out : np.ndarray
        One representative per real pole and per conjugate pair (imag >= 0).
    is_conjugate : np.ndarray
        Boolean, same length as poles_out: False for real, True for conjugate pair or unpaired.
    non_paired : np.ndarray
        Poles that could not be paired.
    """
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    n_poles = poles.size
    cost = np.abs(poles[None, :] - np.conj(poles[:, None]))

    if n_poles == 0:
        return poles.copy(), np.array([], dtype=bool), np.array([], dtype=np.complex128)

    row_ind, col_ind = linear_sum_assignment(cost)
    pair_index = np.empty(n_poles, dtype=np.intp)
    pair_index[row_ind] = col_ind

    pair_type = np.zeros(n_poles, dtype=int)
    for i in range(n_poles):
        if pair_type[i] != 0:
            continue
        j = pair_index[i]
        c = cost[i, j]
        if j == i:
            pair_type[i] = 1 if c < tol_real else -1
        elif pair_index[j] == i and c < tol_pair:
            pair_type[i] = 2
            pair_type[j] = 3
        else:
            pair_type[i] = -1
            if j != i:
                pair_type[j] = -1

    if verbose:
        print("Poles reduction summary:")
        print(f"Number of Poles: {n_poles}")
        print(f"Number of Real Poles: {np.sum(pair_type == 1)}")
        print(
            f"Number of Conjugate Pairs: {np.sum(pair_type == 2)}; Number of Complex Poles: {np.sum(pair_type == 2) * 2}"
        )
        print(f"Number of Unpaired Poles: {np.sum(pair_type == -1)}")
        print(f"List all unpaired poles: {poles[pair_type == -1]}")

    is_conjugate = np.ones(n_poles, dtype=bool)
    is_conjugate[pair_type == 1] = False

    non_paired = poles[pair_type == -1]

    if non_paired.size > 0:
        msg = (
            f"reduce_conjugate_pairs: {non_paired.size} unpaired pole(s) "
            f"will be dropped (tol_pair={tol_pair:.1e}, tol_real={tol_real:.1e}). "
            f"This often indicates imprecise pole estimates upstream."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)

    select = (pair_type == 1) | (pair_type == 2)  # | (pair_type == -1)

    # Mirror the poles to the upper half of the complex plane
    poles = np.real(poles) + 1j * np.abs(np.imag(poles))
    return poles[select], is_conjugate[select], non_paired


def residue_at_pole(
    P: np.ndarray, dP: np.ndarray, B: np.ndarray, C: np.ndarray
) -> tuple[complex, np.ndarray, np.ndarray, np.ndarray]:
    """Residue terms of ``C P(z)^{-1} B`` at a pole ``z_k`` where ``P`` is singular.

    With ``r`` and ``l`` the right and left null vectors of ``P(z_k)``, the
    residue is ``(C r)(l^H B) / (l^H P'(z_k) r)``. ``P`` is scaled before the
    SVD (null vectors are scale-invariant) so poles far from the unit circle,
    where ``z^m`` terms blow up, stay well conditioned.

    Returns
    -------
    denominator : complex
        ``l^H P'(z_k) r``; zero for a multiple pole.
    numerator : ndarray ``(n_out, n_in)``
        ``(C r)(l^H B)``.
    right, left : ndarray ``(N,)``
        The null vectors.
    """
    scale = np.max(np.abs(P))
    if scale > 0 and np.isfinite(scale):
        P = P / scale
    u, _, vh = np.linalg.svd(P)
    right = vh.conj().T[:, -1]
    left = u[:, -1]
    denominator = complex(np.vdot(left, dP @ right))
    numerator = np.outer(C @ right, left.conj() @ B)
    return denominator, numerator, right, left


def residues_from_terms(
    numerators: np.ndarray, denominators: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Residues ``numerator / denominator``, zero (with a warning) for multipoles.

    Returns ``(residues, undriven)`` where ``undriven = 1 / denominator``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        undriven = 1.0 / denominators
        residues = numerators / denominators[:, None, None]
    if np.any(~np.isfinite(undriven)):
        warnings.warn(
            "There are multipoles. The residues are set to zero.", stacklevel=3
        )
    undriven = np.where(np.isfinite(undriven), undriven, 0.0)
    residues = np.where(np.isfinite(residues), residues, 0.0)
    return residues, undriven
