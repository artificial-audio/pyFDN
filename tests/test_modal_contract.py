"""Small regression cases for shared modal result and synthesis contracts."""

import numpy as np
import pytest

from pyFDN.translate.dss_to_impz import dss_to_impz
from pyFDN.translate.dss_to_pr import dss_to_pr
from pyFDN.translate.pr_to_impz import pr_to_impz


def _eai(**options):
    return dss_to_pr(
        [3, 5],
        [[0.4, 0.1], [-0.1, 0.3]],
        np.eye(2),
        np.eye(2),
        np.zeros((2, 2)),
        mode="eai",
        **options,
    )


@pytest.mark.parametrize("iterations", [0, -1, 1.5])
def test_invalid_eai_iterations(iterations):
    with pytest.raises(ValueError, match="maximum_iterations"):
        _eai(maximum_iterations=iterations)


def test_eai_rejects_empty_result_even_with_approximation():
    for approximate in [False, True]:
        with pytest.raises(RuntimeError, match="0/8 poles"):
            _eai(maximum_iterations=1, allow_approximate=approximate)


def test_inaccurate_partial_eai_result_cannot_be_opted_into():
    with pytest.raises(RuntimeError, match="allow_approximate=True"):
        _eai(deflation_type="none")
    with pytest.raises(RuntimeError, match="All-port modal response validation failed"):
        _eai(deflation_type="none", allow_approximate=True)


@pytest.mark.parametrize("mode", ["eig", "roots", "eai", "lindsey_fox"])
def test_complete_result_metadata_and_residue_convention(mode):
    delays = np.array([3, 5])
    a = np.array([[0.4, 0.1], [-0.1, 0.3]])
    r, p, d, pairs, meta = dss_to_pr(
        delays,
        a,
        np.eye(2),
        np.eye(2),
        np.zeros((2, 2)),
        mode=mode,
    )
    assert meta["complete"] and not meta["approximate"]
    assert meta["expectedPoles"] == meta["returnedPoles"] == 8
    assert meta["solver"] == mode
    assert (meta["responseValidation"] is not None) == (mode == "lindsey_fox")
    z = 1.2 + 0.2j
    modal = np.sum(r / (z - p[:, None, None]), axis=0)
    modal += np.sum(r[pairs].conj() / (z - p[pairs, None, None].conj()), axis=0)
    np.testing.assert_allclose(
        modal + d, np.linalg.inv(np.diag(z**delays) - a), atol=1e-11
    )


@pytest.mark.parametrize("mode", ["fast", "lowMemory"])
def test_zero_pole_is_a_unit_delay(mode):
    actual = pr_to_impz(np.ones((1, 1, 1)), [0j], [[0.25]], [False], 4, mode=mode)
    np.testing.assert_array_equal(actual[:, 0, 0], [0.25, 1, 0, 0])
    np.testing.assert_array_equal(
        pr_to_impz(np.ones((1, 1, 1)), [0j], [[0.25]], [False], 1, mode=mode),
        [[[0.25]]],
    )


def test_roots_preserves_a_simple_zero_pole():
    args = ([1, 1], np.diag([0, 0.4]), np.eye(2), np.eye(2), np.zeros((2, 2)))
    result = dss_to_pr(*args, mode="roots")
    assert np.any(result[1] == 0)
    np.testing.assert_allclose(
        pr_to_impz(*result[:4], 16), dss_to_impz(*args, 16), atol=1e-12
    )


@pytest.mark.parametrize("mode", ["eig", "roots"])
def test_unsupported_repeated_zero_poles_raise(mode):
    with pytest.raises(ValueError, match="Multiple.*simple residues"):
        dss_to_pr([3], [[0]], [[1]], [[1]], [[0]], mode=mode)


def test_least_squares_residue_fit_supports_zero_pole():
    from pyFDN.translate.impz_to_res import impz_to_res

    residues, _, _ = impz_to_res(
        np.array([0, 1, 0, 0]), np.array([0j]), np.array([False])
    )
    np.testing.assert_allclose(residues, [1])


def test_repeated_matrix_modes_are_not_returned_as_rank_one_residues():
    with pytest.raises(ValueError, match="Multiple null vectors"):
        dss_to_pr([3, 3], 0.4 * np.eye(2), np.eye(2), np.eye(2), np.zeros((2, 2)))
