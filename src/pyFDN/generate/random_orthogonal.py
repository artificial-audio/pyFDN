import numpy as np
from scipy.linalg import qr

def random_orthogonal(n):
    """Generate a random n x n orthogonal matrix."""
    Q, R = qr(np.random.randn(n, n))
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q
