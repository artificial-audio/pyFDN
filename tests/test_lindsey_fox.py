"""Numerical acceptance tests for the experimental FFT modal backend."""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from pyFDN.auxiliary.lindsey_fox import lindsey_fox_roots
from pyFDN.auxiliary.math import general_char_poly_fft
from pyFDN.translate.dss_to_impz import dss_to_impz
from pyFDN.translate.dss_to_pr import dss_to_pr
from pyFDN.translate.pr_to_impz import pr_to_impz


@pytest.mark.parametrize("gain", [0.2, 0.95, 1.02])
@pytest.mark.parametrize("delays", [[3, 4, 5], [31, 37, 43]])
def test_modal_poles_residues_and_ir_match_eig(gain, delays):
    rng = np.random.default_rng(53)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    a = gain * q
    b = rng.normal(size=(3, 2))
    c = rng.normal(size=(2, 3))
    d = np.full((2, 2), 0.1)
    reference = dss_to_pr(delays, a, b, c, d, mode="eig")
    residues, poles, direct, pairs, meta = dss_to_pr(
        delays, a, b, c, d, mode="lindsey_fox"
    )
    assert np.sum(1 + pairs) == sum(delays)
    assert meta["nonPairedPoles"].size == 0
    row, col = linear_sum_assignment(np.abs(reference[1][:, None] - poles))
    np.testing.assert_allclose(poles[col], reference[1][row], atol=1e-10, rtol=0)
    np.testing.assert_allclose(residues[col], reference[0][row], atol=1e-9, rtol=1e-9)
    np.testing.assert_array_equal(direct, d)
    ir = pr_to_impz(residues, poles, direct, pairs, 512)
    np.testing.assert_allclose(
        ir, dss_to_impz(delays, a, b, c, d, 512), atol=1e-8, rtol=1e-8
    )


def test_fft_search_without_companion_solver(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("FFT backend must not call a companion/eigenvalue solver")

    monkeypatch.setattr(np, "roots", forbidden)
    monkeypatch.setattr(np.linalg, "eigvals", forbidden)
    monkeypatch.setattr(np.linalg, "eig", forbidden)
    # z^127 - 0.7: roots have known radius and angles, including a real root.
    coefficients = np.zeros(128)
    coefficients[[0, -1]] = [-0.7, 1]
    radius = 0.7 ** (1 / 127)
    roots = lindsey_fox_roots(coefficients, radius, radius)
    expected = radius * np.exp(2j * np.pi * np.arange(127) / 127)
    row, col = linear_sum_assignment(np.abs(expected[:, None] - roots))
    np.testing.assert_allclose(roots[col], expected[row], atol=1e-12, rtol=0)


def test_repeated_roots_fail_explicitly():
    with pytest.raises(RuntimeError, match="completeness/reconstruction"):
        lindsey_fox_roots(np.array([1.0, -2.0, 1.0]), 0.9, 1.1)


def test_linear_polynomial():
    np.testing.assert_allclose(lindsey_fox_roots(np.array([-0.5, 1]), 0.5, 0.5), [0.5])


@pytest.mark.parametrize("n", [4, 8])
def test_unequal_line_losses(n):
    rng = np.random.default_rng(37)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    a = q @ np.diag(np.linspace(0.4, 0.98, n))
    delays = 2 * np.arange(n) + 3
    b = rng.normal(size=(n, 1))
    c = rng.normal(size=(1, n))
    result = dss_to_pr(delays, a, b, c, [[0]], mode="lindsey_fox")
    assert np.sum(1 + result[3]) == delays.sum()
    np.testing.assert_allclose(
        pr_to_impz(*result[:4], 512),
        dss_to_impz(delays, a, b, c, [[0]], 512),
        atol=1e-9,
        rtol=1e-9,
    )


@pytest.mark.parametrize("delays", [[0, 2], [-1, 2], [1.5, 2]])
def test_invalid_delays(delays):
    with pytest.raises(ValueError, match="positive integer delays"):
        dss_to_pr(
            delays,
            np.eye(2),
            np.ones((2, 1)),
            np.ones((1, 2)),
            [[0]],
            mode="lindsey_fox",
        )


def test_singular_feedback_rejected():
    with pytest.raises(ValueError, match="nonsingular"):
        dss_to_pr(
            [2, 3],
            np.zeros((2, 2)),
            np.ones((2, 1)),
            np.ones((1, 2)),
            [[0]],
            mode="lindsey_fox",
        )


def test_modal_backend_constructs_polynomial_once_without_principal_minors(monkeypatch):
    from importlib import import_module

    module = import_module("pyFDN.translate.dss_to_pr")
    calls = []

    def construct(delays, a):
        calls.append(1)
        return general_char_poly_fft(delays, a)

    def forbidden(*args, **kwargs):
        raise AssertionError("Modal FFT backend must not enumerate principal minors")

    monkeypatch.setattr(module, "general_char_poly_fft", construct)
    monkeypatch.setattr("pyFDN.auxiliary.math.combinations", forbidden)
    a = np.array([[0.6, 0.1], [-0.2, 0.7]])
    result = dss_to_pr(
        [17, 23], a, np.ones((2, 1)), np.ones((1, 2)), [[0]], mode="lindsey_fox"
    )
    assert len(calls) == 1
    assert np.sum(1 + result[3]) == 40
