import numpy as np

from pyFDN.auxiliary.math import poly_degree
from pyFDN.auxiliary.utils import mag2db


def test_mag2db():
    arr = np.array([1, 10, 100])
    result = mag2db(arr)
    expected = 20 * np.log10(np.maximum(np.abs(arr), np.finfo(float).eps))
    np.testing.assert_allclose(result, expected)


def test_poly_degree_z1():
    poly = np.array([0, 0, 1])
    deg = poly_degree(poly, "z^1")
    assert deg == 0


def test_poly_degree_zm1():
    poly = np.array([1, 0, 0])
    deg = poly_degree(poly, "z^-1")
    assert deg == 0
