from __future__ import annotations

import numpy as np


def random_orthogonal(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random orthogonal matrix distributed according to the Haar measure.

    Draws from ``rng`` when given, otherwise from NumPy's global random state
    (so :func:`numpy.random.seed` makes the result reproducible).
    """

    normal = np.random.standard_normal if rng is None else rng.standard_normal
    q, r = np.linalg.qr(normal((n, n)))
    d = np.sign(np.diag(r))
    d[d == 0] = 1
    return q * d
