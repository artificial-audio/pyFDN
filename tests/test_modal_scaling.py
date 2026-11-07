"""Scalable pairing and factored-response checks with independent small oracles."""

import numpy as np
import pytest

from pyFDN.auxiliary.modal import RankOneResidues, validate_modal_response
from pyFDN.auxiliary.poles import reduce_conjugate_pairs, residue_at_pole


def test_pairing_avoids_global_hungarian(monkeypatch):
    def forbidden(*_):
        raise AssertionError("Resolved conjugates must not use a pairwise assignment")

    monkeypatch.setattr("pyFDN.auxiliary.poles.linear_sum_assignment", forbidden)
    roots = 0.99 * np.exp(2j * np.pi * np.arange(4096) / 4096)
    np.random.default_rng(71).shuffle(roots)
    poles, pairs, unpaired = reduce_conjugate_pairs(roots, strict=True)
    assert np.sum(1 + pairs) == 4096
    assert unpaired.size == 0
    assert np.sum(~pairs) == 2


def test_unbalanced_large_set_preserves_pairs_and_reports_orphan():
    upper = 0.99 * np.exp(1j * np.linspace(0.001, np.pi - 0.001, 2048))
    roots = np.concatenate((upper, upper.conj(), [1.2 + 0.3j]))
    with pytest.warns(UserWarning, match="1 unpaired"):
        poles, pairs, orphan = reduce_conjugate_pairs(roots)
    assert np.sum(1 + pairs) == 4096
    np.testing.assert_array_equal(orphan, [1.2 + 0.3j])
    with pytest.raises(ValueError, match="1 unpaired"):
        reduce_conjugate_pairs(roots, strict=True)


def test_one_sided_cluster_reports_unpaired_without_assignment():
    with pytest.warns(UserWarning, match="1024 unpaired"):
        reduced, _, unpaired = reduce_conjugate_pairs(np.full(1024, 1j))
    assert reduced.size == 0 and unpaired.size == 1024


def test_local_assignment_prefers_legal_pairs_over_lower_total_distance():
    t = 1e-8
    p1, p2 = 0.5 + 1j + 0.2j * t, 0.5 + 0.8 * t + 1j
    roots = np.array([p1, p2, p1.conjugate(), 0.5 - 0.8 * t - 1j])
    # Unrestricted minimum distance prefers an exact match plus an illegal pair.
    _, pairs, unpaired = reduce_conjugate_pairs(roots, strict=True)
    assert np.sum(1 + pairs) == 4 and unpaired.size == 0


def test_pairing_handles_real_sort_crossing_with_local_assignments(monkeypatch):
    from scipy.optimize import linear_sum_assignment

    shapes = []

    def local(cost):
        shapes.append(cost.shape)
        return linear_sum_assignment(cost)

    monkeypatch.setattr("pyFDN.auxiliary.poles.linear_sum_assignment", local)
    roots = np.array([0.5 + 1j, 0.5 + 1.1j, 0.5 + 1e-10 - 1j, 0.5 - 1e-10 - 1.1j])
    poles, pairs, orphan = reduce_conjugate_pairs(roots, strict=True)
    assert np.sum(1 + pairs) == 4 and orphan.size == 0
    assert shapes and all(shape == (1, 1) for shape in shapes)
    np.testing.assert_allclose(poles, roots[:2])


def test_ambiguous_components_use_bounded_assignment(monkeypatch):
    from scipy.optimize import linear_sum_assignment

    shapes = []

    def local(cost):
        shapes.append(cost.shape)
        return linear_sum_assignment(cost)

    monkeypatch.setattr("pyFDN.auxiliary.poles.linear_sum_assignment", local)
    balanced = np.array([0.5 + 1j, 0.5 + 1e-9 + 1j, 0.5 - 1j, 0.5 + 2e-9 - 1j])
    _, balanced_pairs, _ = reduce_conjugate_pairs(balanced, strict=True)
    assert np.sum(1 + balanced_pairs) == 4 and shapes == [(2, 2)]
    shapes.clear()
    roots = np.array([0.5 + 1j, 0.5 + 1e-9 + 1j, 0.5 - 1j, 0.5 + 2e-9 - 1j, 2 + 0.5j])
    with pytest.warns(UserWarning, match="1 unpaired"):
        _, pairs, _ = reduce_conjugate_pairs(roots)
    assert np.sum(1 + pairs) == 4 and shapes == [(2, 2)]
    with pytest.raises(ValueError, match="512-pole"):
        reduce_conjugate_pairs(np.array([1j] * 513 + [-1j] * 512), strict=True)


@pytest.mark.parametrize("size", [0, 1, 20])
def test_reals_and_exact_multiplicities(size):
    roots = np.array([0.2] * size + [0.4 + 1j] * size + [0.4 - 1j] * size)
    reduced, pairs, unpaired = reduce_conjugate_pairs(roots, strict=True)
    assert len(reduced) == 2 * size and unpaired.size == 0
    assert np.sum(1 + pairs) == 3 * size


def test_factored_response_matches_dense_oracle_at_same_grid():
    rng = np.random.default_rng(9)
    poles = np.array([0.3 + 0.2j, 0.5 + 0.1j, 0.7])
    pairs = np.array([True, True, False])
    outputs = rng.normal(size=(3, 7)) + 1j * rng.normal(size=(3, 7))
    inputs = rng.normal(size=(3, 7)) + 1j * rng.normal(size=(3, 7))
    weights = np.array([0.2 + 0.3j, -0.4 + 0.1j, 0.5])
    outputs[-1] = outputs[-1].real
    inputs[-1] = inputs[-1].real
    dense = np.einsum("ki,kj,k->kij", outputs, inputs, weights)

    def probe(z):
        value = np.sum(dense / (z - poles)[:, None, None], axis=0)
        return value + np.sum(
            dense[pairs].conj() / (z - poles[pairs].conj())[:, None, None], axis=0
        )

    expected = validate_modal_response(poles, dense, pairs, probe, 5, 1e-12)
    actual = validate_modal_response(
        poles, RankOneResidues(outputs, inputs, weights), pairs, probe, 5, 1e-12
    )
    assert (
        expected["points"] == actual["points"]
        and expected["radius"] == actual["radius"]
    )
    assert actual["relativeL2"] < 1e-14 and actual["normalizedPeak"] < 1e-14
    with pytest.raises(RuntimeError, match="All-port"):
        validate_modal_response(
            poles, RankOneResidues(outputs, inputs, 2 * weights), pairs, probe, 5, 1e-8
        )


def test_factored_response_rejects_nonfinite_weights():
    with pytest.raises(RuntimeError, match="nonfinite"):
        validate_modal_response(
            np.array([0.5]),
            RankOneResidues(np.ones((1, 2)), np.ones((1, 2)), np.array([np.nan])),
            np.array([False]),
            lambda z: np.eye(2),
            1,
            1e-8,
        )


def test_saved_null_vectors_do_not_retain_full_svd_matrices():
    denominator, numerator, right, left = residue_at_pole(
        np.diag([0.1, 0.2, 0]), np.eye(3), np.eye(3), np.eye(3)
    )
    assert right.flags.owndata and left.flags.owndata
    np.testing.assert_allclose(numerator / denominator, np.diag([0, 0, 1]))


def test_factored_response_cannot_broadcast_a_missing_output_port():
    with pytest.raises(RuntimeError, match="shape"):
        validate_modal_response(
            np.array([0.5]),
            RankOneResidues(np.ones((1, 1)), np.array([[1.0, 2.0]]), np.ones(1)),
            np.array([False]),
            lambda z: np.array([[1.0, 2.0], [1.0, 2.0]]) / (z - 0.5),
            1,
            1e-8,
        )
