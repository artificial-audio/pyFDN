import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import block_diag


def dss_to_ss(
    delays: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a delay state-space (DSS) system to standard state-space.

    Gives every delay-line sample its own state, so the ``N x N`` DSS system
    becomes a ``sum(delays) x sum(delays)`` standard state-space system. The
    result does not depend on ``fs`` or filter hooks; a :class:`~pyFDN.DSS`
    unpacks straight in: ``dss_to_ss(*dss)``.

    Parameters
    ----------
    delays : array-like
        Delay lengths in samples (min 3 samples).
    A : array-like
        Feedback matrix (N, N).
    B, C, D : array-like
        Input ``(N, inputs)``, output ``(outputs, N)`` and direct
        ``(outputs, inputs)`` gains.

    Returns
    -------
    AA : ndarray
        State-space transition matrix.
    bb : ndarray
        State-space input gains.
    cc : ndarray
        State-space output gains.
    dd : ndarray
        State-space direct gains.
    """
    delays_arr = np.asarray(delays, dtype=int).ravel()
    A = np.asarray(A)
    N = A.shape[0]

    if np.any(delays_arr < 3):
        raise ValueError(
            "All `delays` must be at least 3 samples, "
            f"got minimum {int(delays_arr.min())}."
        )

    B = np.asarray(B, dtype=float).reshape(N, -1)
    C = np.asarray(C, dtype=float).reshape(-1, N)
    D = np.asarray(D, dtype=float).reshape(C.shape[0], B.shape[1])

    U_blocks = []
    P = np.zeros((N, 0))  # start with 0 columns
    R = np.zeros((0, N))  # start with 0 rows

    for it in range(N):
        # U_j: (delays_arr[it]-2) x (delays_arr[it]-2) with 1's on first
        # superdiagonal. np.diag(np.ones(n), 1) has shape (n+1, n+1), so
        # passing size_Uj = delays_arr[it] - 3 directly (including the
        # size_Uj == 0 case, which yields the (1, 1) zero block needed for the
        # minimum delay of 3 samples) already produces the correct
        # (delays_arr[it]-2, delays_arr[it]-2) block, matching the sizes of the
        # corresponding P_j/R_j blocks.
        size_Uj = delays_arr[it] - 3
        U_j = np.diag(np.ones(size_Uj), 1)
        U_blocks.append(U_j)

        # R_j: (delays_arr[it]-2) x N, last row = 1 at column it
        R_j = np.zeros((delays_arr[it] - 2, N))
        R_j[-1, it] = 1
        R = np.vstack([R, R_j]) if R.size else R_j

        # P_j: N x (delays_arr[it]-2), first column = 1 at row it
        P_j = np.zeros((N, delays_arr[it] - 2))
        if delays_arr[it] - 2 > 0:
            P_j[it, 0] = 1
        P = np.hstack([P, P_j]) if P.size else P_j

    # Block diagonal U
    if U_blocks:
        U = block_diag(*U_blocks)
    else:
        U = np.zeros((0, 0))

    # Construct AA
    top = np.hstack([U, np.zeros_like(R), R])
    middle = np.hstack([P, np.zeros((N, 2 * N))])
    bottom = np.hstack([np.zeros_like(P), A, np.zeros_like(A)])
    AA = np.vstack([top, middle, bottom])

    NN = AA.shape[0]

    bb = np.zeros((NN, B.shape[1]))
    bb[-N:] = B
    cc = np.zeros((C.shape[0], NN))
    cc[:, -2 * N : -N] = C
    return AA, bb, cc, D
