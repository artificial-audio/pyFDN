import numpy as np
from scipy.linalg import qr


def random_orthogonal(n: int) -> np.ndarray:
    """Generate a random orthogonal matrix distributed according to the Haar measure."""

    q, r = np.linalg.qr(np.random.standard_normal((n, n)))
    d = np.sign(np.diag(r))
    d[d == 0] = 1
    return q * d
