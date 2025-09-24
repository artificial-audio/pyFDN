import numpy as np

def outer_sum_approximation(A):
    """
    Minimizes || u + v' - A ||_F with a rank-1 approximation.

    Args:
        A (ndarray): 2D input matrix

    Returns:
        u (ndarray): Column vector of shape (N,)
        v (ndarray): Column vector of shape (M,)
    """
    # Transform to exponential domain
    maxA = np.max(A)
    eA = np.exp(A / maxA)

    # Rank-1 approximation via SVD
    U, S, Vh = np.linalg.svd(eA, full_matrices=False)
    sqrt_singular = np.sqrt(S[0])
    
    eu = U[:, 0] * sqrt_singular
    ev = Vh[0, :] * sqrt_singular

    # Transform back from exp domain
    u = np.log(np.abs(eu)) * maxA
    v = np.log(np.abs(ev)) * maxA

    return u, v
