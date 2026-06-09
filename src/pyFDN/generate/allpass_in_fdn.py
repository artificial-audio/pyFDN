"""Allpass-in-FDN construction of size [2N, 2N].

Translation of allpassInFDN.m from fdnToolbox.
Original MATLAB: Sebastian Jiro Schlecht, 28 December 2019.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def allpass_in_fdn(
    g: ArrayLike,
    A: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create an allpass structure embedded in an FDN of size [2N, 2N].

    See Schlecht, S. (2017). *Feedback delay networks in artificial
    reverberation and reverberation enhancement*.

    Parameters
    ----------
    g : array-like, shape (N,)
        Per-channel feedforward/back allpass gains.
    A : array-like, shape (N, N)
        Inner FDN feedback matrix.
    b : array-like, shape (N,) or (N, 1)
        Input gains of the inner FDN.
    c : array-like, shape (N,) or (1, N)
        Output gains of the inner FDN.
    d : float
        Direct gain.

    Returns
    -------
    A_out : ndarray, shape (2N, 2N)
        FDN feedback matrix.
    B_out : ndarray, shape (2N, 1)
        FDN input gains.
    C_out : ndarray, shape (1, 2N)
        FDN output gains.
    D_out : ndarray, shape (1, 1)
        FDN direct gain.

    Example
    -------
    >>> import numpy as np
    >>> from pyFDN.generate.random_orthogonal import random_orthogonal
    >>> g = np.random.randn(3)
    >>> A, B, C, D = allpass_in_fdn(g, random_orthogonal(3),
    ...                              np.ones((3, 1)), np.ones((1, 3)), 0.0)
    >>> A.shape
    (6, 6)
    """
    g = np.asarray(g, dtype=float).ravel()  # (N,)
    A = np.asarray(A, dtype=float)  # (N, N)
    b = np.asarray(b, dtype=float).reshape(-1, 1)  # (N, 1)
    c = np.asarray(c, dtype=float).reshape(1, -1)  # (1, N)

    N = len(g)
    G = np.diag(g)  # (N, N)
    I = np.eye(N)

    A_out = np.block(
        [
            [-A @ G, A],
            [I - G @ G, G],
        ]
    )
    B_out = np.vstack([b, np.zeros((N, 1))])  # (2N, 1)
    C_out = np.hstack([g.reshape(1, -1), c])  # (1, 2N)
    D_out = np.array([[d]])  # (1, 1)

    return A_out, B_out, C_out, D_out
