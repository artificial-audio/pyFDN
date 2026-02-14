"""Tests for auxiliary.math module."""

import numpy as np
import pytest

from pyFDN import matrix_convolution
from pyFDN import matrix_polyval
from pyFDN import negpolyder
from pyFDN import outer_sum_approximation
from pyFDN import poly_degree
from pyFDN import polyder_rational
from pyFDN import polydiag


# ============================================================================
# Matrix Convolution Tests
# ============================================================================

def test_matrix_convolution_basic():
    A = np.ones((2, 2, 2))
    B = np.ones((2, 2, 2))
    C = matrix_convolution(A, B)
    assert C.shape == (2, 2, 3)


# ============================================================================
# Matrix Polyval Tests
# ============================================================================

def test_matrix_polyval_basic():
    P = np.ones((2, 2, 3))
    z = 2
    Y = matrix_polyval(P, z)
    assert Y.shape == (2, 2)


# ============================================================================
# Poly Degree Tests
# ============================================================================

def test_poly_degree_z1():
    poly = np.array([0, 0, 1])
    deg = poly_degree(poly, "z^1")
    assert deg == 0


def test_poly_degree_zm1():
    poly = np.array([1, 0, 0])
    deg = poly_degree(poly, "z^-1")
    assert deg == 0


# ============================================================================
# Polydiag Tests
# ============================================================================

def test_polydiag_basic():
    p = np.array([[1, 2], [3, 4]])
    d = polydiag(p)
    assert d.shape == (2, 2, 2)
    assert np.all(d[0, 0, :] == [1, 2])
    assert np.all(d[1, 1, :] == [3, 4])


# ============================================================================
# Outer Sum Approximation Tests
# ============================================================================

def test_outer_sum_approximation_handles_zero_matrix():
    u, v = outer_sum_approximation(np.zeros((3, 4)))
    assert np.array_equal(u, np.zeros(3))
    assert np.array_equal(v, np.zeros(4))


# ============================================================================
# Polyder Rational Tests
# ============================================================================

def test_polyder_rational_matches_finite_difference():
    b = np.array([1.0, -0.5, 0.25])
    a = np.array([1.0, -0.2])
    q_pos, p_pos = polyder_rational(b, a)

    z = 0.8
    eps = 1e-6

    def rational(val: float) -> float:
        return np.polyval(b, val) / np.polyval(a, val)

    forward = rational(z + eps)
    center = rational(z - eps)
    finite_diff = (forward - center) / (2 * eps)

    analytic = np.polyval(q_pos, z) / np.polyval(p_pos, z)
    assert pytest.approx(finite_diff, rel=1e-5) == analytic


def test_negpolyder_preserves_length_when_requested():
    b = np.array([1.0, -0.5, 0.25])
    a = np.array([1.0, -0.2])
    with pytest.raises(ValueError):
        negpolyder(b, a, dont_truncate=True)
