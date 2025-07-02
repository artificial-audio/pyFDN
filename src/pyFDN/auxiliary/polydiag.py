import numpy as np

def polydiag(p):
    """
    Convert array of polynomials p (shape [N, FIR]) to diagonal polynomial matrix d (shape [N, N, FIR]).
    """
    N, L = p.shape
    d = np.zeros((N, N, L), dtype=p.dtype)
    for it in range(N):
        d[it, it, :] = p[it, :]
    return d
