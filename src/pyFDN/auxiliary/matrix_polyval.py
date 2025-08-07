import numpy as np


def matrix_polyval(P, z):
    """
    Evaluate matrix polynomial at z.
    P: shape (N, M, FIR)
    z: scalar (can be complex)
    Returns:
        Y: shape (N, M)
    """
    degree = P.shape[2]
    exponents = np.arange(degree-1, -1, -1)
    zz = z ** exponents
    zz = zz.reshape((1, 1, degree))
    Y = np.sum(P * zz, axis=2)
    return Y
