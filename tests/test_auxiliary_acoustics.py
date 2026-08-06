"""Tests for auxiliary.acoustics module."""

import numpy as np
import pytest

from pyFDN.auxiliary.acoustics import absorption_filters, absorption_to_rt, rt_to_slope
from pyFDN.auxiliary.utils import db_to_lin

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
    coeffs = absorption_filters(
        np.array([0.0, fs / 2]), target_rt, filter_order, delays, fs
    )

    expected = db_to_lin(delays * rt_to_slope(target_rt[0, :], fs))[:, None]
    assert coeffs.shape == (delays.size, filter_order + 1)
    assert np.allclose(coeffs, expected)


def test_absorption_filters_round_trip_with_analysis():
    fs = 48_000
    frequency = np.array([0.0, 1_000.0, 4_000.0, fs / 2])
    target_rt = np.array(
        [
            [1.0, 0.8],
            [0.9, 0.75],
            [0.8, 0.6],
            [0.8, 0.6],
        ]
    )
    delays = np.array([32, 48])
    filter_order = 8

    coeffs = absorption_filters(frequency, target_rt, filter_order, delays, fs)
    rt, freq = absorption_to_rt(coeffs, delays, 2**12, fs)

    assert freq.shape[0] == rt.shape[0]
    # Compare at sample frequencies
    for idx, f in enumerate(frequency[:-1]):
        closest = np.argmin(np.abs(freq - f))
        assert np.allclose(rt[closest, :], target_rt[idx, :], atol=0.15)


def test_absorption_to_rt_handles_single_channel():
    fs = 48_000
    delays = np.array([24])
    coeffs = np.ones((1, 3)) * 0.5
    rt, freq = absorption_to_rt(coeffs, delays, 512, fs)

    assert freq.ndim == 1
    assert rt.shape[1] == coeffs.shape[0]
    assert np.all(np.isfinite(rt))


def test_octave_bands_edges_and_nyquist():
    from pyFDN.auxiliary.acoustics import octave_bands

    bands, f_centre = octave_bands()
    np.testing.assert_allclose(f_centre, 1000.0 * 2.0 ** np.arange(-4.0, 4.0))
    np.testing.assert_allclose(bands[:, 0], f_centre / np.sqrt(2))
    np.testing.assert_allclose(bands[:, 1], f_centre * np.sqrt(2))

    # bands reaching Nyquist are dropped
    _, f_centre_16k = octave_bands(fs=16000)
    assert f_centre_16k.tolist() == f_centre[:-1].tolist()


def test_octave_band_filterbank_passes_its_own_band():
    from scipy.signal import sosfreqz

    from pyFDN.auxiliary.acoustics import octave_band_filterbank, octave_bands

    fs = 48000
    bands, f_centre = octave_bands(fs=fs)
    sos_bank = octave_band_filterbank(bands, fs)

    assert len(sos_bank) == len(f_centre)
    for sos, centre, (lower, upper) in zip(sos_bank, f_centre, bands, strict=True):
        w, h = sosfreqz(sos, worN=[centre, lower / 4, upper * 4], fs=fs)
        gain_db = 20 * np.log10(np.abs(h))
        assert gain_db[0] > -3.1  # passband: within 3 dB at the centre frequency
        assert np.all(gain_db[1:] < -40)  # stopband: two octaves outside the band


def test_estimate_rt_bands_rejects_band_above_nyquist():
    from pyFDN.auxiliary.acoustics import octave_band_filterbank

    with pytest.raises(ValueError):
        octave_band_filterbank(np.array([[30000.0, 40000.0]]), fs=48000)


def test_slope_amplitude_to_level_round_trip():
    from pyFDN.auxiliary.acoustics import slope_amplitude_to_level

    fs = 48000
    rt = np.array([1.5, 0.5])
    level = np.array([0.3, 0.7])
    # energy of an exponentially decaying envelope L * 10 ** (-3 t / T)
    t = np.arange(4 * fs) / fs
    energy = np.array(
        [
            np.sum((li * 10 ** (-3 * t / ti)) ** 2)
            for li, ti in zip(level, rt, strict=True)
        ]
    )

    np.testing.assert_allclose(
        slope_amplitude_to_level(energy, rt, fs), level, rtol=1e-3
    )


def test_slope_amplitude_to_level_inactive_slope_is_zero():
    from pyFDN.auxiliary.acoustics import slope_amplitude_to_level

    # multi-slope estimators mark inactive slopes with T = 0 and A = 0
    level = slope_amplitude_to_level(
        np.array([[1.0, 0.0]]), np.array([[2.0, 0.0]]), 48000
    )
    assert level.shape == (1, 2)
    assert level[0, 1] == 0.0
    assert level[0, 0] > 0.0


def test_estimate_initial_level_bands_synthetic_decay():
    from pyFDN.auxiliary.acoustics import (
        estimate_initial_level_bands,
        estimate_rt_bands,
        octave_bands,
    )

    fs = 48000
    rt_true = 1.5
    level_true = 0.3
    rng = np.random.default_rng(0)
    n = np.arange(2 * fs)
    ir = rng.standard_normal(len(n)) * level_true * 10 ** (-3 * n / (rt_true * fs))

    rt, f_centre = estimate_rt_bands(ir, fs)
    level, f_centre_level = estimate_initial_level_bands(ir, rt, fs)

    np.testing.assert_allclose(f_centre_level, f_centre)
    np.testing.assert_allclose(rt, rt_true, rtol=0.1)
    # white noise: each band holds the fraction of broadband energy given by
    # its bandwidth, so the expected band level is level_true * sqrt(bw / (fs/2))
    bands, _ = octave_bands(fs=fs)
    expected = level_true * np.sqrt((bands[:, 1] - bands[:, 0]) / (fs / 2))
    np.testing.assert_allclose(level, expected, rtol=0.2)


def test_estimate_initial_level_bands_rt_length_mismatch():
    from pyFDN.auxiliary.acoustics import estimate_initial_level_bands

    with pytest.raises(ValueError):
        estimate_initial_level_bands(np.random.randn(48000), np.ones(3), 48000)
