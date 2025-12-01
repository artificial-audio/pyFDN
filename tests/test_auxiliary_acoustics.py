"""Tests for auxiliary.acoustics module."""

import numpy as np
import pytest

from pyFDN.auxiliary.acoustics import absorption_filters
from pyFDN.auxiliary.acoustics import absorption_to_t60
from pyFDN.auxiliary.acoustics import rt60_to_slope
from pyFDN.auxiliary.utils import db2mag


# ============================================================================
# Absorption Filter Tests
# ============================================================================

@pytest.mark.parametrize(
    "target_rt, delays",
    [
        (np.array([[1.2, 0.8], [1.2, 0.8]]), np.array([12, 24])),
        (np.array([[0.6, 0.6], [0.6, 0.6]]), np.array([5, 5])),
    ],
)
def test_absorption_filters_zero_order_matches_closed_form(target_rt, delays):
    fs = 48_000
    filter_order = 0
    coeffs = absorption_filters(np.array([0.0, fs / 2]), target_rt, filter_order, delays, fs)

    expected = db2mag(delays * rt60_to_slope(target_rt[0, :], fs))[:, None]
    assert coeffs.shape == (delays.size, filter_order + 1)
    assert np.allclose(coeffs, expected)


def test_absorption_filters_round_trip_with_analysis():
    fs = 48_000
    frequency = np.array([0.0, 1_000.0, 4_000.0, fs / 2])
    target_t60 = np.array(
        [
            [1.0, 0.8],
            [0.9, 0.75],
            [0.8, 0.6],
            [0.8, 0.6],
        ]
    )
    delays = np.array([32, 48])
    filter_order = 8

    coeffs = absorption_filters(frequency, target_t60, filter_order, delays, fs)
    t60, freq = absorption_to_t60(coeffs, delays, 2 ** 12, fs)

    assert freq.shape[0] == t60.shape[0]
    # Compare at sample frequencies
    for idx, f in enumerate(frequency[:-1]):
        closest = np.argmin(np.abs(freq - f))
        assert np.allclose(t60[closest, :], target_t60[idx, :], atol=0.15)


def test_absorption_to_t60_handles_single_channel():
    fs = 48_000
    delays = np.array([24])
    coeffs = np.ones((1, 3)) * 0.5
    t60, freq = absorption_to_t60(coeffs, delays, 512, fs)

    assert freq.ndim == 1
    assert t60.shape[1] == coeffs.shape[0]
    assert np.all(np.isfinite(t60))
