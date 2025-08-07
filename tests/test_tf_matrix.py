import numpy as np

from pyFDN.auxiliary.tf_matrix import TFMatrix


def test_tfmatrix_init_and_at():
    num = np.ones((2, 2, 3))
    den = np.ones((2, 2, 3))
    tfm = TFMatrix(num, den, var='z^-1')
    z = 1.0
    result = tfm.at(z)
    assert result.shape == (2, 2)

def test_tfmatrix_poles():
    num = np.ones((2, 2, 3))
    den = np.zeros((2, 2, 3))
    den[..., 0] = 1
    tfm = TFMatrix(num, den)
    poles = tfm.poles()
    assert isinstance(poles, np.ndarray)
