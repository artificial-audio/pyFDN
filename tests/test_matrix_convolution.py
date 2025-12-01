import numpy as np

from pyFDN.auxiliary.math import matrix_convolution


def test_matrix_convolution_basic():
    A = np.ones((2, 2, 2))
    B = np.ones((2, 2, 2))
    C = matrix_convolution(A, B)
    assert C.shape == (2, 2, 3)
