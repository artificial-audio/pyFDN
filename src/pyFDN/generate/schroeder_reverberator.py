"""Schroeder reverberator as a feedback delay network.

Translation of SchroederReverberator.m from fdnToolbox.
Original MATLAB: Sebastian Jiro Schlecht, 28 December 2019.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def schroeder_reverberator(
    allpass_gain: ArrayLike,
    comb_gain: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create combs and allpass filters as a single FDN.

    Combines parallel comb filters with a series allpass section into one
    feedback delay network.  See Schlecht (2017), *Feedback delay networks in
    artificial reverberation and reverberation enhancement*.

    Parameters
    ----------
    allpass_gain : array-like, shape (Na,)
        Feedforward/back gains for the series allpass stages.
    comb_gain : array-like, shape (Nc,)
        Feedback gains for the parallel comb filters.
    b : array-like, shape (Nc,) or (Nc, 1)
        Input gains of the comb filters.
    c : array-like, shape (Nc,) or (1, Nc)
        Output gains of the comb filters.
    d : float
        Direct gain.

    Returns
    -------
    A : ndarray, shape (Na+Nc, Na+Nc)
        FDN feedback matrix.
    B : ndarray, shape (Na+Nc, 1)
        FDN input gains.
    C : ndarray, shape (1, Na+Nc)
        FDN output gains.
    D : ndarray, shape (1, 1)
        FDN direct gain.

    Example
    -------
    >>> import numpy as np
    >>> g_ap = np.array([0.5, 0.4, 0.3])
    >>> g_c  = np.array([0.7, 0.6, 0.5])
    >>> A, B, C, D = schroeder_reverberator(g_ap, g_c, np.ones(3)/3, np.ones(3)/3, 0.0)
    >>> A.shape
    (6, 6)
    """
    from ..auxiliary.allpass import series_allpass

    allpass_gain = np.asarray(allpass_gain, dtype=float).ravel()
    comb_gain = np.asarray(comb_gain, dtype=float).ravel()
    b = np.asarray(b, dtype=float).reshape(-1, 1)  # (Nc, 1)
    c = np.asarray(c, dtype=float).reshape(1, -1)  # (1, Nc)
    d = float(d)

    N_c = len(comb_gain)
    N_a = len(allpass_gain)

    AP_A, AP_B, AP_C, AP_D = series_allpass(allpass_gain)

    P = np.diag(comb_gain)  # (Nc, Nc)
    S = AP_B @ c  # (Na, 1) @ (1, Nc) → (Na, Nc)

    A = np.block(
        [
            [P, np.zeros((N_c, N_a))],
            [S, AP_A],
        ]
    )
    B = np.vstack([b, np.zeros((N_a, 1))])  # (Nc+Na, 1)
    C = np.hstack([AP_D * c, AP_C])  # (1, Nc+Na)
    D_out = np.array([[d]]) * AP_D  # (1, 1)

    return A, B, C, D_out
