"""Tests for generator utilities."""

import numpy as np
import pytest

from pyFDN.auxiliary.math import matrix_convolution
from pyFDN.generate.anderson_matrix import anderson_matrix
from pyFDN.generate.complete_orthogonal import complete_orthogonal
from pyFDN.generate.construct_cascaded_paraunitary_matrix import (
    construct_cascaded_paraunitary_matrix,
)
from pyFDN.generate.construct_velvet_feedback_matrix import (
    construct_velvet_feedback_matrix,
)
from pyFDN.generate.degree_one_lossless import degree_one_lossless
from pyFDN.generate.fdn_matrix_gallery import fdn_matrix_gallery
from pyFDN.generate.householder_matrix import householder_matrix
from pyFDN.generate.is_almost_zero import is_almost_zero
from pyFDN.generate.nearest_orthogonal import nearest_orthogonal
from pyFDN.generate.nearest_sign_agnostic_orthogonal import (
    nearest_sign_agnostic_orthogonal,
)
from pyFDN.generate.random_matrix_shift import random_matrix_shift
from pyFDN.generate.random_orthogonal import random_orthogonal
from pyFDN.generate.shift_matrix import shift_matrix
from pyFDN.generate.shift_matrix_distribute import shift_matrix_distribute


@pytest.fixture()
def deterministic_rng(monkeypatch):
    generator = np.random.default_rng(42)
    monkeypatch.setattr(np.random, "default_rng", lambda: generator)


def test_random_orthogonal_produces_unitary_matrix():
    mat = random_orthogonal(5)
    identity = mat.T @ mat
    np.testing.assert_allclose(identity, np.eye(5), atol=1e-12)


def test_random_orthogonal():
    n = 4
    Q = random_orthogonal(n)
    assert Q.shape == (n, n)
    # Q should be orthogonal: Q.T @ Q = I
    identity_matrix = Q.T @ Q
    np.testing.assert_allclose(identity_matrix, np.eye(n), atol=1e-7)


def test_shift_matrix_supports_left_and_right():
    tensor = np.arange(12).reshape(2, 2, 3)
    shifted_left = shift_matrix(tensor, np.array([1, 0]), "left")
    shifted_right = shift_matrix(tensor, np.array([0, 1]), "right")

    assert shifted_left.shape[-1] >= tensor.shape[-1]
    assert shifted_right.shape[-1] >= tensor.shape[-1]
    np.testing.assert_allclose(shifted_left[1, :, : tensor.shape[2]], tensor[1, :, :])
    np.testing.assert_allclose(shifted_right[:, 0, : tensor.shape[2]], tensor[:, 0, :])


def test_shift_matrix_distribute_respects_pulse_size(deterministic_rng):
    tensor = np.ones((3, 3, 2))
    shifts = shift_matrix_distribute(tensor, sparsity=0.5, pulse_size=4)
    np.testing.assert_equal(shifts.shape, (3,))
    np.testing.assert_equal(shifts % 4, 0)


def test_random_matrix_shift_returns_consistent_lengths(deterministic_rng):
    tensor = np.ones((3, 3, 2))
    shifted, shifted_rev, left, right = random_matrix_shift(5, tensor, tensor)

    np.testing.assert_equal(shifted.shape[0:2], tensor.shape[0:2])
    np.testing.assert_(shifted.shape[2] >= tensor.shape[2])
    assert shifted_rev is not None
    np.testing.assert_(shifted_rev.shape[2] >= tensor.shape[2])
    np.testing.assert_equal(left.size, tensor.shape[0])
    np.testing.assert_equal(right.size, tensor.shape[0])
    np.testing.assert_(np.all(left >= 0))
    np.testing.assert_(np.all(right >= 0))


def test_construct_cascaded_paraunitary_matrix_is_inverse(monkeypatch):
    monkeypatch.setattr(
        "pyFDN.generate.shift_matrix_distribute.shift_matrix_distribute",
        lambda *args, **kwargs: np.zeros(args[0].shape[0], dtype=int),
    )
    matrix, rev = construct_cascaded_paraunitary_matrix(4, 0, matrix_type="Hadamard")
    product = matrix_convolution(matrix, rev)
    identity = np.zeros_like(product)
    identity[:, :, 0] = np.eye(4)
    np.testing.assert_allclose(product, identity)


def test_construct_velvet_feedback_matches_wrapper(monkeypatch):
    monkeypatch.setattr(
        "pyFDN.generate.construct_velvet_feedback_matrix.construct_cascaded_paraunitary_matrix",
        lambda *args, **kwargs: (np.ones((2, 2, 1)), np.ones((2, 2, 1))),
    )
    matrix, rev = construct_velvet_feedback_matrix(2, 1, 0.5)
    np.testing.assert_equal(matrix, 1.0)
    np.testing.assert_equal(rev, 1.0)


def test_filter_matrix_gallery_types_are_paraunitary():
    import pyFDN

    np.random.seed(0)
    n = 4
    types = pyFDN.filter_matrix_gallery()
    assert types == ["RandomDense", "Velvet", "FromElementals"]
    for mtype in types:
        mat = pyFDN.filter_matrix_gallery(n, mtype, num_stages=2)
        assert mat.ndim == 3 and mat.shape[:2] == (n, n)
        is_pu, _, _ = pyFDN.is_paraunitary(mat.transpose(2, 0, 1))
        assert is_pu, mtype

    # stage_matrix_type is honored for the cascaded types
    mat_rnd = pyFDN.filter_matrix_gallery(
        n, "Velvet", num_stages=2, stage_matrix_type="random"
    )
    is_pu, _, _ = pyFDN.is_paraunitary(mat_rnd.transpose(2, 0, 1))
    assert is_pu

    with pytest.raises(ValueError, match="Unknown matrix_type"):
        pyFDN.filter_matrix_gallery(n, "nope")
    with pytest.raises(ValueError, match="N must be provided"):
        pyFDN.filter_matrix_gallery(matrix_type="Velvet")


def test_is_almost_zero_true_and_false():
    np.testing.assert_(is_almost_zero(np.array([1e-13])))
    np.testing.assert_(not is_almost_zero(np.array([1e-8])))


# ---------------------------------------------------------------------------
# householder_matrix
# ---------------------------------------------------------------------------


def test_householder_matrix_is_orthogonal():
    H = householder_matrix(np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(H @ H.T, np.eye(4), atol=1e-12)


def test_householder_matrix_is_symmetric():
    H = householder_matrix(np.array([1.0, 0.0, 1.0]))
    np.testing.assert_allclose(H, H.T, atol=1e-12)


def test_householder_matrix_is_involutory():
    """H @ H = I (self-inverse)."""
    H = householder_matrix(np.random.randn(5))
    np.testing.assert_allclose(H @ H, np.eye(5), atol=1e-12)


# ---------------------------------------------------------------------------
# nearest_orthogonal
# ---------------------------------------------------------------------------


def test_nearest_orthogonal_is_orthogonal():
    A = np.random.randn(5, 5)
    B = nearest_orthogonal(A)
    np.testing.assert_allclose(B @ B.T, np.eye(5), atol=1e-10)


def test_nearest_orthogonal_identity_preserved():
    """Nearest orthogonal of I is I."""
    B = nearest_orthogonal(np.eye(4))
    np.testing.assert_allclose(np.abs(B), np.eye(4), atol=1e-12)


# ---------------------------------------------------------------------------
# nearest_sign_agnostic_orthogonal
# ---------------------------------------------------------------------------


def test_nearest_sign_agnostic_orthogonal_is_orthogonal():
    np.random.seed(0)
    A = np.abs(np.random.randn(4, 4))
    U = nearest_sign_agnostic_orthogonal(A, max_trials=200)
    np.testing.assert_allclose(U @ U.T, np.eye(4), atol=1e-8)


# ---------------------------------------------------------------------------
# complete_orthogonal
# ---------------------------------------------------------------------------


def test_complete_orthogonal_v_is_orthogonal():
    """V = [[A, b], [c, d]] must be orthogonal when A has exactly num_io sv < 1."""
    rng = np.random.default_rng(42)
    N = 4
    U = np.linalg.qr(rng.standard_normal((N, N)))[0]
    V_mat = np.linalg.qr(rng.standard_normal((N, N)))[0]
    sigma = np.array([1.0, 1.0, 1.0, 0.6])
    A = U @ np.diag(sigma) @ V_mat.T
    _, _, _, V_full = complete_orthogonal(A, 1)
    np.testing.assert_allclose(V_full @ V_full.T, np.eye(N + 1), atol=1e-10)


def test_complete_orthogonal_multi_io():
    rng = np.random.default_rng(7)
    N = 5
    U = np.linalg.qr(rng.standard_normal((N, N)))[0]
    Vm = np.linalg.qr(rng.standard_normal((N, N)))[0]
    sigma = np.array([1.0, 1.0, 1.0, 0.7, 0.5])
    A = U @ np.diag(sigma) @ Vm.T
    _, _, _, V_full = complete_orthogonal(A, 2)
    np.testing.assert_allclose(V_full @ V_full.T, np.eye(N + 2), atol=1e-10)


# ---------------------------------------------------------------------------
# degree_one_lossless
# ---------------------------------------------------------------------------


def test_degree_one_lossless_shape():
    V = degree_one_lossless(np.array([1.0, 0.0, 0.0, 0.0]))
    assert V.shape == (4, 4, 2)


def test_degree_one_lossless_z0_plus_z1_equals_identity():
    v = np.random.randn(5)
    V = degree_one_lossless(v)
    np.testing.assert_allclose(V[:, :, 0] + V[:, :, 1], np.eye(5), atol=1e-12)


def test_degree_one_lossless_z1_is_rank1():
    v = np.random.randn(4)
    V = degree_one_lossless(v)
    assert np.linalg.matrix_rank(V[:, :, 1]) == 1


# ---------------------------------------------------------------------------
# fdn_matrix_gallery
# ---------------------------------------------------------------------------


def test_fdn_matrix_gallery_returns_type_list():
    types = fdn_matrix_gallery()
    assert isinstance(types, list)
    assert "Hadamard" in types
    assert "orthogonal" in types


@pytest.mark.parametrize("matrix_type", ["orthogonal", "Householder", "circulant"])
def test_fdn_matrix_gallery_orthogonal_types(matrix_type):
    A = fdn_matrix_gallery(4, matrix_type)
    assert isinstance(A, np.ndarray)
    np.testing.assert_allclose(A @ A.T, np.eye(4), atol=1e-8)


def test_fdn_matrix_gallery_hadamard():
    A = fdn_matrix_gallery(8, "Hadamard")
    assert isinstance(A, np.ndarray)
    np.testing.assert_allclose(A @ A.T, np.eye(8), atol=1e-10)


def test_fdn_matrix_gallery_parallel():
    assert np.allclose(fdn_matrix_gallery(4, "parallel"), np.eye(4))


def test_fdn_matrix_gallery_unknown_type_raises():
    with pytest.raises(ValueError):
        fdn_matrix_gallery(4, "unknown_type_xyz")


# ---------------------------------------------------------------------------
# anderson_matrix
# ---------------------------------------------------------------------------


def test_anderson_matrix_shape():
    AM = anderson_matrix(8)
    assert AM.shape == (8, 8)


def test_anderson_matrix_is_orthogonal():
    AM = anderson_matrix(8)
    np.testing.assert_allclose(AM @ AM.T, np.eye(8), atol=1e-10)


def test_anderson_matrix_custom_block():
    AM = anderson_matrix(12, K=4, matrix_type="orthogonal")
    assert AM.shape == (12, 12)
    np.testing.assert_allclose(AM @ AM.T, np.eye(12), atol=1e-8)


def test_anderson_matrix_invalid_k_raises():
    with pytest.raises(ValueError, match="divisible"):
        anderson_matrix(9, K=4)
