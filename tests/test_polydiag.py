import numpy as np

from pyFDN.auxiliary.math import polydiag


def test_polydiag_basic():
    p = np.array([[1, 2], [3, 4]])
    d = polydiag(p)
    assert d.shape == (2, 2, 2)
    assert np.all(d[0, 0, :] == [1, 2])
    assert np.all(d[1, 1, :] == [3, 4])
