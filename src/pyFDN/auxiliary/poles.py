"""Pole utilities (conjugate pairing, etc.)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def _conjugate_pair_indices(
    poles: np.ndarray, tol_real: float, tol_pair: float
) -> np.ndarray:
    """Sorted matching, with bounded local assignments for ambiguous clusters."""
    if not np.all(np.isfinite(poles)):
        raise ValueError("Expected finite poles")
    mate = np.full(poles.size, -1, dtype=np.intp)
    real = np.abs(poles.imag) < tol_real / 2
    mate[real] = np.flatnonzero(real)
    positive = np.flatnonzero(~real & (poles.imag > 0))
    negative = np.flatnonzero(~real & (poles.imag < 0))
    upper = poles[positive]
    lower = poles[negative].conj()
    ip = np.lexsort((upper.imag, upper.real))
    im = np.lexsort((lower.imag, lower.real))
    if positive.size == negative.size and np.all(
        np.abs(upper[ip] - lower[im]) < tol_pair
    ):
        resolved = np.array_equal(upper[ip], lower[im]) or upper.size < 2
        if not resolved:
            # Disjoint tolerance balls make the sorted matching unambiguous.
            points = np.column_stack((upper.real, upper.imag))
            distances, _ = cKDTree(points).query(points, k=2)
            resolved = bool(np.all(distances[:, 1] >= 2 * tol_pair))
        if resolved:
            mate[positive[ip]], mate[negative[im]] = negative[im], positive[ip]
            return mate
    indices = np.concatenate((positive, negative))
    if positive.size == 0 or negative.size == 0:
        return mate
    values = np.concatenate((upper, lower))
    coordinates = np.column_stack((values.real, values.imag))
    tree = cKDTree(coordinates)
    visited = np.zeros(indices.size, dtype=bool)
    for seed in range(indices.size):
        if visited[seed]:
            continue
        component = [seed]
        visited[seed] = True
        cursor = 0
        while cursor < len(component):
            neighbors = tree.query_ball_point(coordinates[component[cursor]], tol_pair)
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    component.append(neighbor)
            # Bound both local assignment memory and pathological cluster work.
            if len(component) > 512:
                raise ValueError(
                    "Ambiguous conjugate cluster exceeds the 512-pole local assignment limit"
                )
            cursor += 1
        local = np.asarray(component)
        left = local[local < positive.size]
        right = local[local >= positive.size]
        if left.size == 0 or right.size == 0:
            continue
        cost = np.abs(values[left, None] - values[right])
        allowed = cost < tol_pair
        # Prefer the maximum number of legal pairs before minimizing distances.
        penalty = (max(left.size, right.size) + 1) * tol_pair
        rows, cols = linear_sum_assignment(np.where(allowed, cost, penalty))
        valid = allowed[rows, cols]
        a, b = indices[left[rows[valid]]], indices[right[cols[valid]]]
        mate[a], mate[b] = b, a
    return mate


def reduce_conjugate_pairs(
    poles: np.ndarray | ArrayLike,
    *,
    tol_real: float = 1e-10,
    tol_pair: float = 1e-8,
    verbose: bool = False,
    strict: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group real poles and conjugate pairs without a global pairwise cost matrix.

    Sorted upper/lower-half-plane matching handles complete, well-resolved sets.
    Otherwise a spatial tree identifies tolerance-connected components, with
    optimal assignments restricted to local components of at most 512 poles.
    Larger ambiguous components raise rather than allocating quadratic storage.
    Storage is linear in the pole count plus one bounded local cost matrix.
    A real pole satisfies ``2*abs(imag(pole)) < tol_real``; pair distances must
    be below ``tol_pair``. Output order follows the original input order.

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
    mate = _conjugate_pair_indices(poles, tol_real, tol_pair)
    indices = np.arange(n_poles)
    select = (mate >= 0) & (indices <= mate)
    is_conjugate = mate != indices
    if verbose:
        print(
            f"Poles reduction: {np.sum(select & ~is_conjugate)} real, "
            f"{np.sum(select & is_conjugate)} conjugate pairs, {np.sum(mate < 0)} unpaired"
        )

    non_paired = poles[mate < 0]

    if non_paired.size > 0:
        msg = (
            f"reduce_conjugate_pairs: {non_paired.size} unpaired pole(s) "
            f"will be dropped (tol_pair={tol_pair:.1e}, tol_real={tol_real:.1e}). "
            f"This often indicates imprecise pole estimates upstream."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)

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
    # These vectors are retained for every mode; views would keep full SVD matrices.
    right = vh.conj().T[:, -1].copy()
    left = u[:, -1].copy()
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
