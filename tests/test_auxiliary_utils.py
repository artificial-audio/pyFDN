"""Tests for auxiliary.utils module."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyFDN.auxiliary.acoustics import (
    first_order_absorption,
    one_pole_absorption,
    rt_to_slope,
    slope_to_rt,
)
from pyFDN.auxiliary.delay import ms_to_smp
from pyFDN.auxiliary.math import negpolyder, outer_sum_approximation, polyder_rational
from pyFDN.auxiliary.utils import (
    ensure_3d,
    hertz_to_unit,
    is_bounding_curve,
    last_nonzero_indices,
    lin_to_db,
    max_corr,
    pole_boundaries,
)

# ============================================================================
# Conversion Utility Tests
# ============================================================================


def test_ms_to_smp_round_trip_simple_values():
    fs = 48_000
    times_ms = [0.0, 0.5, 1.0, 10.0]
    expected = np.array([0, 24, 48, 480])
    assert np.array_equal(ms_to_smp(times_ms, fs), expected)


def test_hertz_to_unit_maps_to_nyquist():
    fs = 48_000
    hz = np.array([0.0, fs / 2])
    normalized = hertz_to_unit(hz, fs)
    assert normalized[0] == 0.0
    assert normalized[1] == 1.0


def test_rt_slope_inverse_relationship():
    fs = 48_000
    rt = np.array([0.4, 1.2, 2.5])
    slope = rt_to_slope(rt, fs)
    recovered = slope_to_rt(slope, fs)
    assert np.allclose(recovered, rt)


# ============================================================================
# lin_to_db Tests
# ============================================================================


def test_lin_to_db_basic():
    arr = np.array([1, 10, 100])
    result = lin_to_db(arr)
    expected = 20 * np.log10(arr)
    np.testing.assert_allclose(result, expected)


def test_lin_to_db_zero():
    arr = np.array([0])
    result = lin_to_db(arr)
    assert np.isfinite(result)


def test_lin_to_db_from_poly_degree():
    arr = np.array([1, 10, 100])
    result = lin_to_db(arr)
    expected = 20 * np.log10(np.maximum(np.abs(arr), np.finfo(float).eps))
    np.testing.assert_allclose(result, expected)


# ============================================================================
# One-Pole Absorption Tests
# ============================================================================


def test_one_pole_absorption_shapes_are_correct():
    delays = np.array([10.0, 20.0, 30.0])
    sos = one_pole_absorption(1.2, 0.8, delays, 44100.0)
    assert sos.shape == (1, 6, delays.size)
    assert np.all(sos[0, 3, :] == 1.0)


def test_first_order_absorption_matches_rt_targets():
    fs = 48000.0
    rt_dc, rt_ny = 1.2, 0.8
    delays = np.array([100.0, 130.0, 250.0])
    sos = first_order_absorption(rt_dc, rt_ny, delays, fs, crossover_frequency=4000.0)

    assert sos.shape == (1, 6, delays.size)
    s = sos[0]  # (6, N): rows [b0, b1, b2, a0, a1, a2]
    assert np.all(s[3, :] == 1.0)
    assert np.all(s[[2, 5], :] == 0.0)  # first-order: b2 = a2 = 0
    assert np.all(np.abs(s[4, :]) < 1.0)  # stable pole

    # gain at DC (z=1) and Nyquist (z=-1) must match the target decay per delay
    h_dc = (s[0] + s[1]) / (s[3] + s[4])
    h_ny = (s[0] - s[1]) / (s[3] - s[4])
    np.testing.assert_allclose(h_dc, 10 ** (delays * (-60.0 / (rt_dc * fs)) / 20.0))
    np.testing.assert_allclose(h_ny, 10 ** (delays * (-60.0 / (rt_ny * fs)) / 20.0))


def test_first_order_absorption_clamps_high_crossover():
    fs = 48000.0
    delays = np.array([100.0, 130.0])
    clamped = first_order_absorption(1.0, 0.5, delays, fs, crossover_frequency=fs / 3)
    limit = first_order_absorption(1.0, 0.5, delays, fs, crossover_frequency=fs / 5)
    np.testing.assert_allclose(clamped, limit)


# ============================================================================
# Helper Utility Tests
# ============================================================================


def test_ensure_3d_promotes_2d_arrays():
    mat = np.array([[1.0, 2.0]])
    promoted = ensure_3d(mat)
    assert promoted.shape == (1, 2, 1)
    assert promoted[..., 0].tolist() == mat.tolist()


def test_ensure_3d_rejects_invalid_rank():
    with pytest.raises(ValueError):
        ensure_3d(np.array([1.0]))


def test_last_nonzero_indices_reports_positions():
    tensor = np.array(
        [
            [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]],
            [[0.0, 3.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    indices = last_nonzero_indices(tensor)
    assert indices.shape == (2, 2)
    assert indices[0, 0] == 3
    assert indices[1, 0] == 2
    assert np.all(indices[:, 1] == 0)


# ============================================================================
# Bounding Curve Tests
# ============================================================================


def test_is_bounding_curve_upper():
    x = np.linspace(0, 1, 5)
    y = np.array([1, 2, 3, 4, 5])
    x_curve = x
    y_curve = y + 1
    all_bounded, is_bounded = is_bounding_curve(x, y, x_curve, y_curve, "upper")
    assert all_bounded
    assert np.all(is_bounded)


def test_is_bounding_curve_lower():
    x = np.linspace(0, 1, 5)
    y = np.array([1, 2, 3, 4, 5])
    x_curve = x
    y_curve = y - 1
    all_bounded, is_bounded = is_bounding_curve(x, y, x_curve, y_curve, "lower")
    assert all_bounded
    assert np.all(is_bounded)


# ============================================================================
# Pole Boundaries Tests
# ============================================================================


def test_pole_boundaries_basic():
    delays = np.array([1, 2])
    b = np.ones((2, 1, 4))
    a = np.ones((2, 1, 4))
    absorption = SimpleNamespace(b=b, a=a)
    feedback_matrix = np.ones((2, 2, 4))
    fs = 48000
    MinCurve, MaxCurve, f = pole_boundaries(
        delays, absorption, feedback_matrix, fs, nfft=8
    )
    assert MinCurve.shape == (8,)
    assert MaxCurve.shape == (8,)
    assert f.shape == (8,)
    # regression: freqz unpacking once swapped (w, h), making f complex
    assert not np.iscomplexobj(f)
    assert np.all(np.diff(f) > 0)
    assert np.all(MinCurve >= 0) and np.all(MaxCurve >= MinCurve - 1e-12)


# ============================================================================
# Math Utility Tests (from test_aux_utils.py)
# ============================================================================


def test_outer_sum_approximation_handles_zero_matrix():
    u, v = outer_sum_approximation(np.zeros((3, 4)))
    assert np.array_equal(u, np.zeros(3))
    assert np.array_equal(v, np.zeros(4))


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
# max_corr Tests
# ============================================================================


def test_max_corr_diagonal_and_symmetry():
    rng = np.random.default_rng(0)
    signals = rng.standard_normal((2, 2, 256))
    M = max_corr(signals)
    assert M.shape == (4, 4)
    np.testing.assert_allclose(np.diag(M), np.ones(4), atol=1e-12)
    np.testing.assert_allclose(M, M.T, atol=1e-12)


def test_max_corr_shifted_negated_copy():
    # a delayed, negated copy has maximum correlation -1 at the matching lag
    rng = np.random.default_rng(1)
    x = rng.standard_normal(50)
    signals = np.zeros((1, 2, 80))
    signals[0, 0, :50] = x
    signals[0, 1, 10:60] = -x
    M = max_corr(signals)
    assert M[0, 1] == pytest.approx(-1.0)


def test_max_corr_unfolds_column_major():
    # signal k corresponds to entry (k % N1, k // N1), as in MATLAB maxCorr.m
    signals = np.zeros((2, 2, 16))
    signals[1, 0, 3] = 1.0  # column-major index 1
    signals[0, 1, 7] = 1.0  # column-major index 2
    M = max_corr(signals)
    assert M[1, 2] == pytest.approx(1.0)
    assert M[0, 3] == pytest.approx(0.0)


def test_max_corr_zero_signal_yields_zero_row():
    signals = np.zeros((1, 2, 32))
    signals[0, 0, 0] = 1.0
    M = max_corr(signals)
    assert M[0, 1] == 0.0
    assert M[1, 1] == 0.0
