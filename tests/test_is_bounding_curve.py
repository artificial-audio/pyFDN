import numpy as np
from pyFDN.auxiliary.is_bounding_curve import is_bounding_curve

def test_is_bounding_curve_upper():
    x = np.linspace(0, 1, 5)
    y = np.array([1, 2, 3, 4, 5])
    x_curve = x
    y_curve = y + 1
    all_bounded, is_bounded = is_bounding_curve(x, y, x_curve, y_curve, 'upper')
    assert all_bounded
    assert np.all(is_bounded)

def test_is_bounding_curve_lower():
    x = np.linspace(0, 1, 5)
    y = np.array([1, 2, 3, 4, 5])
    x_curve = x
    y_curve = y - 1
    all_bounded, is_bounded = is_bounding_curve(x, y, x_curve, y_curve, 'lower')
    assert all_bounded
    assert np.all(is_bounded) 