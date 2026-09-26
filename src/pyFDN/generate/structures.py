"""Classic reverberator structures written as delay state-space systems.

Each builder returns the (A, B, C, D) matrices of a DSS system; pair them
with delays to render. See Schlecht, "Allpass Feedback Delay Networks," and
Schlecht (2017), *Feedback delay networks in artificial reverberation and
reverberation enhancement*.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def poletti_allpass(
    g: float, U: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def series_allpass(
    g: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        s_matrix = np.block(
            [
                [matrix, np.zeros((matrix.shape[0], 1))],
                [output_gain * g2, np.array([[allpass_gain]])],
            ]
        )
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


def nested_allpass(
    g: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        n_matrix = np.block(
            [
                [matrix, input_gain],
                [output_gain * ga, direct * ga],
            ]
        )
        n_input_gain = np.vstack([np.zeros_like(input_gain), np.array([[1 - ga**2]])])
        n_output_gain = np.hstack([output_gain, direct])
        n_direct = np.array([[-ga]])
        matrix = n_matrix
        input_gain = n_input_gain
        output_gain = n_output_gain
        direct = n_direct
    return matrix, input_gain, output_gain, direct


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
    >>> from pyFDN import random_orthogonal
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
