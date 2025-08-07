import numpy as np
from pyFDN.auxiliary.mag2db import mag2db

def test_mag2db_basic():
    arr = np.array([1, 10, 100])
    result = mag2db(arr)
    expected = 20 * np.log10(arr)
    np.testing.assert_allclose(result, expected)

def test_mag2db_zero():
    arr = np.array([0])
    result = mag2db(arr)
    assert np.isfinite(result) 