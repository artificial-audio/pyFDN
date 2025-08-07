import numpy as np

from pyFDN.auxiliary.matrix_polyval import matrix_polyval


def test_matrix_polyval_basic():
    P = np.ones((2, 2, 3))
    z = 2
    Y = matrix_polyval(P, z)
    assert Y.shape == (2, 2)
