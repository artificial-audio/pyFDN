import numpy as np

from pyFDN.generate.random_orthogonal import random_orthogonal


def test_random_orthogonal():
    n = 4
    Q = random_orthogonal(n)
    assert Q.shape == (n, n)
    # Q should be orthogonal: Q.T @ Q = I
    identity_matrix = Q.T @ Q
    np.testing.assert_allclose(identity_matrix, np.eye(n), atol=1e-7)
