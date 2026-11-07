"""Independent evaluation, reconstruction, and large-root numerical oracles."""

from time import monotonic

import numpy as np
import pytest

from pyFDN.auxiliary._fft_polynomial import FFTPolynomial
from pyFDN.auxiliary.lindsey_fox import _unfactor, _unique_roots, lindsey_fox_roots


def test_fft_derivative_ratios_match_analytic_monomial():
    n = 8192
    coefficients = np.zeros(n + 1)
    coefficients[[0, -1]] = [-0.9, 1]
    rng = np.random.default_rng(73)
    angles = rng.uniform(-np.pi, np.pi, 2048)
    points = np.exp((-0.1 + 0.05 * rng.uniform(-1, 1, 2048)) / n + 1j * angles)
    engine = FFTPolynomial(coefficients, 1 << 16, np.inf)
    p, dp, ddp, _, _ = engine.evaluate(points)
    truth = points**n - 0.9
    expected_dp = n * points ** (n - 1)
    expected_ddp = n * (n - 1) * points ** (n - 2)
    np.testing.assert_allclose(dp / p, expected_dp / truth, rtol=2e-8, atol=1e-7)
    np.testing.assert_allclose(ddp / p, expected_ddp / truth, rtol=2e-8, atol=1e-2)


def test_fft_dense_polynomial_ratios_match_horner():
    rng = np.random.default_rng(83)
    n = 4096
    coefficients = rng.normal(size=n + 1) + 1j * rng.normal(size=n + 1)
    points = np.exp(
        rng.uniform(-0.4, 0.4, 2048) / n + 1j * rng.uniform(-np.pi, np.pi, 2048)
    )
    p, dp, ddp, _, _ = FFTPolynomial(coefficients, 1 << 15, np.inf).evaluate(points)
    truth = np.polynomial.polynomial.polyval(points, coefficients)
    d = np.polynomial.polynomial.polyder(coefficients)
    d2 = np.polynomial.polynomial.polyder(d)
    np.testing.assert_allclose(
        dp / p,
        np.polynomial.polynomial.polyval(points, d) / truth,
        rtol=2e-9,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        ddp / p,
        np.polynomial.polynomial.polyval(points, d2) / truth,
        rtol=2e-9,
        atol=0.05,
    )


def test_large_sparse_roots_pass_analytic_oracle_without_companion(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Companion/eigenvalue solvers are not part of LF")

    monkeypatch.setattr(np, "roots", forbidden)
    monkeypatch.setattr(np.linalg, "eigvals", forbidden)
    n = 8192
    coefficients = np.zeros(n + 1)
    coefficients[[0, -1]] = [-0.9, 1]
    radius = 0.9 ** (1 / n)
    roots = lindsey_fox_roots(coefficients, radius, radius, max_seconds=30)
    index = (
        np.rint(np.mod(np.angle(roots), 2 * np.pi) * n / (2 * np.pi)).astype(int) % n
    )
    assert np.unique(index).size == n
    expected = radius * np.exp(2j * np.pi * index / n)
    np.testing.assert_allclose(roots, expected, rtol=0, atol=1e-12)


def test_fft_reconstruction_large_known_roots():
    n = 16384
    roots = 0.9 ** (1 / n) * np.exp(2j * np.pi * np.arange(n) / n)
    expected = np.zeros(n + 1)
    expected[[0, -1]] = [-0.9, 1]
    np.testing.assert_allclose(_unfactor(roots), expected, atol=1e-9, rtol=1e-9)


def test_spatial_uniqueness_matches_greedy_order():
    roots = np.array(
        [0.5 + 1j, 0.5 + 1j + 1e-10, 0.5 + 1j + 2e-8, 1 - 0.2j, 1 - 0.2j + 1e-11]
    )
    expected = []
    for root in roots[np.argsort(np.angle(roots))]:
        if not expected or min(abs(root - np.array(expected))) > 1e-8:
            expected.append(root)
    np.testing.assert_array_equal(_unique_roots(roots), expected)


def test_direct_chunks_and_fft_circles_observe_deadline():
    engine = FFTPolynomial(np.ones(4097), 1 << 15, monotonic() - 1)
    with pytest.raises(RuntimeError, match="time budget"):
        engine.evaluate(np.ones(10, dtype=complex))
    with pytest.raises(RuntimeError, match="time budget"):
        engine.evaluate(np.ones(2048, dtype=complex))


def test_curvature_seeds_recover_missing_close_partner():
    from pyFDN.auxiliary.lindsey_fox import _complete

    pairs = 2048
    phase = 0.1
    coefficients = np.zeros(2 * pairs + 1, dtype=complex)
    coefficients[[0, pairs, -1]] = [1, -2 * np.cos(phase), 1]
    angles = (2 * np.pi * np.arange(pairs)[:, None] + np.array([-phase, phase])) / pairs
    expected = np.exp(1j * angles.ravel())
    partial = expected[1:]
    engine = FFTPolynomial(coefficients, 1 << 15, np.inf)
    recovered = _complete(partial, engine, 2 * pairs)
    assert recovered.size == expected.size
    assert min(abs(recovered - expected[0])) < 1e-12
    np.testing.assert_allclose(_unfactor(recovered), coefficients, atol=1e-8, rtol=0)


def test_fft_verification_allowance_covers_analytic_values():
    n = 4096
    coefficients = np.zeros(n + 1, dtype=complex)
    coefficients[[0, -1]] = [-0.9, 1]
    rng = np.random.default_rng(419)
    points = np.exp(
        rng.uniform(-1, 1, 2048) / n + 1j * rng.uniform(-np.pi, np.pi, 2048)
    )
    value, _, _, allowance, scale = FFTPolynomial(
        coefficients, 1 << 15, np.inf
    ).evaluate(points, derivatives=False)
    anchor = np.rint(n * np.log(abs(points)))
    normalization = np.maximum(0.9, np.exp(anchor))
    truth = (points**n - 0.9) / normalization
    assert np.all(abs(value - truth) <= allowance)
    actual_scale = (abs(points) ** n + 0.9) / normalization
    assert np.all(scale <= actual_scale * (1 + 1e-14))
