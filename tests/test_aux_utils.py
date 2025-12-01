"""Utility-level tests for auxiliary helpers."""

import numpy as np
import pytest

from pyFDN.auxiliary.utils import hertz2unit
from pyFDN.auxiliary.delay import ms2smp
from pyFDN.auxiliary.math import negpolyder
from pyFDN.auxiliary.acoustics import one_pole_absorption
from pyFDN.auxiliary.math import outer_sum_approximation
from pyFDN.auxiliary.math import polyder_rational
from pyFDN.auxiliary.acoustics import rt60_to_slope
from pyFDN.auxiliary.acoustics import slope_to_rt60


def test_ms2smp_round_trip_simple_values():
    fs = 48_000
    times_ms = [0.0, 0.5, 1.0, 10.0]
    expected = np.array([0, 24, 48, 480])
    assert np.array_equal(ms2smp(times_ms, fs), expected)


def test_hertz2unit_maps_to_nyquist():
    fs = 48_000
    hz = np.array([0.0, fs / 2])
    normalized = hertz2unit(hz, fs)
    assert normalized[0] == 0.0
    assert normalized[1] == 1.0


def test_rt60_slope_inverse_relationship():
    fs = 48_000
    rt60 = np.array([0.4, 1.2, 2.5])
    slope = rt60_to_slope(rt60, fs)
    recovered = slope_to_rt60(slope, fs)
    assert np.allclose(recovered, rt60)


def test_one_pole_absorption_shapes_are_correct():
    delays = np.array([10.0, 20.0, 30.0])
    sos = one_pole_absorption(1.2, 0.8, delays, 44100.0)
    assert sos.shape == (6, delays.size)
    assert np.all(sos[3, :] == 1.0)


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
