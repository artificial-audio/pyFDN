import numpy as np
from scipy.signal import grpdelay

def mgrpdelay(A):
    """
    Compute group delay of a 3D FIR filter matrix A for each matrix entry.

    Args:
        A (ndarray): FIR matrix of shape (N, M, FIR)

    Returns:
        GD (ndarray): Group delay of shape (N, M, FIR)
        w (ndarray): Frequency array returned by scipy.signal.grpdelay
    """
    N, M, FIR = A.shape
    GD = np.zeros((N, M, FIR))
    w = None

    for i in range(N):
        for j in range(M):
            # grpdelay returns (w, gd), so we swap output
            w_tmp, gd_tmp = grpdelay(A[i, j, :], a=1, fs=2*np.pi)
            GD[i, j, :] = gd_tmp
            if w is None:
                w = w_tmp

    return GD, w
