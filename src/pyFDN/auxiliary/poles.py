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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group poles into real and conjugate pairs using optimal assignment.

    For real-coefficient systems, poles are either real or occur in conjugate
    pairs. This uses the linear sum assignment problem (Hungarian method): cost
    C[i,j] = |poles[j] - conj(poles[i])|; the minimum-cost permutation pairs each
    pole with its conjugate (or itself for real poles). Then:
    - Real: assignment[i] == i and C[i,i] < tol_real (i.e. |Im(pole_i)| small).
    - Conjugate pair: assignment[i] == j, assignment[j] == i, C[i,j] < tol_pair.
    - Unpaired: otherwise (ambiguous or numerical orphans).

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
        print(f"Number of Conjugate Pairs: {np.sum(pair_type == 2)}; Number of Complex Poles: {np.sum(pair_type == 2) * 2}")
        print(f"Number of Unpaired Poles: {np.sum(pair_type == -1)}")
        print(f"List all unpaired poles: {poles[pair_type == -1]}")
        
    is_conjugate = np.ones(n_poles, dtype=bool)
    is_conjugate[pair_type == 1] = False

    non_paired = poles[pair_type == -1]
    
    select = (pair_type == 1) | (pair_type == 2) # | (pair_type == -1)
    
    # Mirror the poles to the upper half of the complex plane
    poles = np.real(poles) + 1j * np.abs(np.imag(poles))
    return poles[select], is_conjugate[select], non_paired