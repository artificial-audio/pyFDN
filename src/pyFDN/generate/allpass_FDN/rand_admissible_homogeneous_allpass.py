"""
Random admissible diagonal matrix for homogeneous uniallpass FDN.

See "Allpass Feedback Delay Networks" by Sebastian J. Schlecht.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rand_admissible_homogeneous_allpass(
    G: ArrayLike,
    range_: tuple[float, float] | ArrayLike,
) -> np.ndarray:
    """
    Generate a random admissible diagonal matrix P for homogeneous uniallpass FDN.

    Used with homogeneous_allpass_fdn(G, P) to obtain a uniallpass FDN with
    homogeneous decay. Admissibility is defined by the construction in the paper.

    Parameters
    ----------
    G : (N, N) array-like
        Diagonal attenuation matrix with 0 < diag(G) < 1.
    range_ : tuple (low, high) or array-like of length 2
        Random range with 0 < low < high < 1. Diagonal entries of P are built
        from random ratios in [low, high] scaled by diag(G)^2.

    Returns
    -------
    P : (N, N) ndarray
        Admissible diagonal matrix (first diagonal entry 1, rest from cumprod).

    Examples
    --------
    >>> G = np.diag([0.9, 0.8, 0.7])
    >>> P = rand_admissible_homogeneous_allpass(G, (0.3, 0.8))
    >>> A, b, c, d, U = homogeneous_allpass_fdn(G, P)
    """
    G = np.asarray(G, dtype=float)
    N = G.shape[0]
    if G.shape != (N, N):
        raise ValueError("G must be square")
    range_arr = np.asarray(range_, dtype=float).ravel()
    if range_arr.shape[0] != 2:
        raise ValueError("range_ must be a pair (low, high)")
    low, high = float(range_arr[0]), float(range_arr[1])
    if not (0 < low < high < 1):
        raise ValueError("range_ must satisfy 0 < low < high < 1")

    ratios = np.diag(G) ** 2
    rand_ratios = (np.random.rand(N) * (high - low) + low) * ratios
    cum = np.cumprod(1.0 / rand_ratios[1:])
    diag_P = np.concatenate([[1.0], cum])
    return np.diag(diag_P)
