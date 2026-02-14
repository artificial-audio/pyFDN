"""Tests for generator utilities."""

import numpy as np
import pytest

from pyFDN import matrix_convolution
from pyFDN import construct_cascaded_paraunitary_matrix
from pyFDN import construct_velvet_feedback_matrix
from pyFDN import is_almost_zero
from pyFDN import random_matrix_shift
from pyFDN import random_orthogonal
from pyFDN import shift_matrix
from pyFDN import shift_matrix_distribute


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
    np.testing.assert_allclose(shifted_left[1, :, :tensor.shape[2]], tensor[1, :, :])
    np.testing.assert_allclose(shifted_right[:, 0, :tensor.shape[2]], tensor[:, 0, :])


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


def test_is_almost_zero_true_and_false():
    np.testing.assert_(is_almost_zero(np.array([1e-13])))
    np.testing.assert_(not is_almost_zero(np.array([1e-8])))
