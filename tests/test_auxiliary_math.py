"""Tests for auxiliary.math module."""

import numpy as np
import pytest

from pyFDN.auxiliary.math import (
    adj_poly,
    adjugate,
    det_polynomial,
    general_char_poly,
    loop_tf,
    matrix_convolution,
    matrix_polyval,
    negpolyder,
    outer_sum_approximation,
    poly_degree,
    polyder_rational,
    polydiag,
)

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


def test_poly_degree_zm1():
    # [1, 0, 0] in z^{-1} ordering: only z^0 term is nonzero, degree = 0
    poly = np.array([1, 0, 0])
    deg = poly_degree(poly)
    assert deg == 0


def test_poly_degree_zm1_nonzero_last():
    # [0, 0, 1] in z^{-1} ordering: z^{-2} term, degree = 2
    poly = np.array([0, 0, 1])
    deg = poly_degree(poly)
    assert deg == 2


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


# ============================================================================
# det_polynomial Tests
# ============================================================================


def test_det_polynomial_scalar_matrix():
    # Constant scalar matrix: det of 2x2 identity as polynomial matrix
    A = np.zeros((2, 2, 1))
    A[0, 0, 0] = 1.0
    A[1, 1, 0] = 1.0
    result = det_polynomial(A)
    np.testing.assert_allclose(result, [1.0], atol=1e-10)


def test_det_polynomial_diagonal_delay():
    # diag([z^{-1}, z^{-2}]): det = z^{-3} => coefficients [0, 0, 0, 1]
    A = np.zeros((2, 2, 3))
    A[0, 0, 1] = 1.0  # z^{-1}
    A[1, 1, 2] = 1.0  # z^{-2}
    result = det_polynomial(A)
    expected = np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_det_polynomial_known_2x2():
    # [[1 + z^{-1}, z^{-1}], [0, 1]]: det = 1 + z^{-1}
    A = np.zeros((2, 2, 2))
    A[0, 0, 0] = 1.0
    A[0, 0, 1] = 1.0
    A[0, 1, 1] = 1.0
    A[1, 1, 0] = 1.0
    result = det_polynomial(A)
    np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-10)


def test_det_polynomial_agrees_with_linalg_det_for_constant():
    # For a constant matrix (degree 0), det_polynomial must match np.linalg.det
    rng = np.random.default_rng(0)
    M = rng.standard_normal((3, 3))
    A = M[:, :, np.newaxis]  # shape (3, 3, 1)
    result = det_polynomial(A)
    np.testing.assert_allclose(result[0], np.linalg.det(M), rtol=1e-10)


# ============================================================================
# general_char_poly Tests
# ============================================================================


def test_general_char_poly_scalar_identity_single_delay():
    # A=0 (no feedback), single delay d: GCP = 1 - 0 = 1 (constant)
    delays = np.array([3])
    A = np.array([[0.0]])
    p = general_char_poly(delays, A)
    expected = np.zeros(4)
    expected[0] = 1.0
    np.testing.assert_allclose(p, expected, atol=1e-12)


def test_general_char_poly_scalar_two_delays():
    # Known result: for N=2, A=diag(a1,a2), delays=[d1,d2]:
    # GCP = 1 - a1*z^{-d1} - a2*z^{-d2} + (a1*a2 - a2*a1)*z^{-(d1+d2)}
    #      = 1 - a1*z^{-d1} - a2*z^{-d2}  (off-diagonal det is zero for diagonal A)
    delays = np.array([2, 3])
    a1, a2 = 0.5, 0.3
    A = np.diag([a1, a2])
    p = general_char_poly(delays, A)
    expected = np.zeros(6)
    expected[0] = 1.0
    expected[2] = -a1
    expected[3] = -a2
    expected[5] = a1 * a2  # det of full 2x2 diagonal submatrix
    np.testing.assert_allclose(p, expected, atol=1e-12)


def test_general_char_poly_polynomial_A_matches_scalar():
    # A polynomial A with only a degree-0 term must match scalar A path
    delays = np.array([2, 3])
    rng = np.random.default_rng(1)
    A_scalar = rng.standard_normal((2, 2)) * 0.3
    A_poly = A_scalar[:, :, np.newaxis]  # shape (2, 2, 1)
    p_scalar = general_char_poly(delays, A_scalar)
    p_poly = general_char_poly(delays, A_poly)
    min_len = min(len(p_scalar), len(p_poly))
    np.testing.assert_allclose(p_poly[:min_len], p_scalar[:min_len], atol=1e-8)


# ============================================================================
# Adjugate Tests
# ============================================================================


def test_adjugate_satisfies_cofactor_identity():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4))
    adj = adjugate(A)
    np.testing.assert_allclose(adj @ A, np.linalg.det(A) * np.eye(4), atol=1e-12)
    np.testing.assert_allclose(A @ adj, np.linalg.det(A) * np.eye(4), atol=1e-12)


def test_adjugate_singular_matrix():
    # rank-1 matrix: A @ adj(A) = det(A) I = 0, but adj(A) itself is nonzero
    # only for n == 2; for n >= 3 all cofactors of a rank-1 matrix vanish
    rng = np.random.default_rng(1)
    A = np.outer(rng.standard_normal(3), rng.standard_normal(3))
    adj = adjugate(A)
    np.testing.assert_allclose(A @ adj, np.zeros((3, 3)), atol=1e-12)
    np.testing.assert_allclose(adj, np.zeros((3, 3)), atol=1e-12)


def test_adjugate_2x2_closed_form():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.array([[4.0, -2.0], [-3.0, 1.0]])
    np.testing.assert_allclose(adjugate(A), expected, atol=1e-12)


def test_adjugate_complex_matrix():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    adj = adjugate(A)
    np.testing.assert_allclose(adj @ A, np.linalg.det(A) * np.eye(3), atol=1e-12)


# ============================================================================
# Loop TF and adj_poly Tests
# ============================================================================


def test_loop_tf_static_matrix_structure():
    delays = np.array([2, 3])
    A = np.array([[0.1, 0.2], [0.3, 0.4]])
    P = loop_tf(delays, A)
    # P(z) = diag(z^m) - A in z^1 convention, last slice = z^0
    assert P.shape == (2, 2, 4)
    np.testing.assert_allclose(P[:, :, -1], -A)
    assert P[0, 0, 3 - 2] == 1.0  # z^2 coefficient of entry (0, 0)
    assert P[1, 1, 3 - 3] == 1.0  # z^3 coefficient of entry (1, 1)


def test_adj_poly_z1_pointwise_identity():
    # adj(P)(z) @ P(z) = det(P(z)) I at any evaluation point
    rng = np.random.default_rng(3)
    delays = np.array([3, 5, 4])
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    P = loop_tf(delays, Q)
    adj = adj_poly(P, "z^1")

    for z in [0.95 * np.exp(1j * 0.7), 1.1 * np.exp(-1j * 2.1)]:
        Pz = matrix_polyval(P, z)
        adj_z = matrix_polyval(adj, z)
        np.testing.assert_allclose(adj_z @ Pz, np.linalg.det(Pz) * np.eye(3), atol=1e-9)


def test_adj_poly_zm1_pointwise_identity():
    rng = np.random.default_rng(4)
    B = rng.standard_normal((3, 3, 4))
    adj = adj_poly(B, "z^-1")

    zi = 1.0 / (0.9 * np.exp(1j * 1.3))
    Bz = sum(B[:, :, k] * zi**k for k in range(B.shape[2]))
    adj_z = sum(adj[:, :, k] * zi**k for k in range(adj.shape[2]))
    np.testing.assert_allclose(adj_z @ Bz, np.linalg.det(Bz) * np.eye(3), atol=1e-9)


def test_adj_poly_rejects_unknown_convention():
    with pytest.raises(ValueError):
        adj_poly(np.zeros((2, 2, 3)), "z^2")
