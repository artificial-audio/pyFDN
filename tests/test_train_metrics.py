"""Torch-free tests for pyFDN.train metrics and data types.

These must run without torch/flamo installed, so this module imports only the
NumPy metrics and the plain dataclasses.
"""

import numpy as np
import pytest

from pyFDN.train import (
    Trainable,
    TrainLog,
    edc_l1,
    magnitude_response,
    mr_stft_distance,
    octave_colouration,
    spectral_flatness,
)
from pyFDN.train.objectives import output_domain


def _impulse(n=2048):
    x = np.zeros(n)
    x[0] = 1.0
    return x


def _tone(n=2048, fs=48000.0, f=1000.0):
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * f * t)


# --- spectral_flatness -----------------------------------------------------


def test_spectral_flatness_impulse_is_flat():
    # An impulse has a perfectly flat magnitude spectrum -> flatness == 1.
    assert spectral_flatness(_impulse()) == pytest.approx(1.0, abs=1e-9)


def test_spectral_flatness_tone_is_peaky():
    assert spectral_flatness(_tone()) < 0.1


def test_spectral_flatness_in_unit_interval():
    rng = np.random.default_rng(0)
    for sig in (_impulse(), _tone(), rng.standard_normal(2048)):
        value = spectral_flatness(sig)
        assert 0.0 <= value <= 1.0 + 1e-9


def test_spectral_flatness_white_above_tone():
    rng = np.random.default_rng(1)
    assert spectral_flatness(rng.standard_normal(4096)) > spectral_flatness(_tone())


# --- magnitude_response ----------------------------------------------------


def test_magnitude_response_shape_and_impulse_is_unit():
    mag = magnitude_response(_impulse(1024))
    assert mag.shape == (1024 // 2 + 1,)
    np.testing.assert_allclose(mag, 1.0, atol=1e-9)


# --- octave_colouration ----------------------------------------------------


def test_octave_colouration_shape_and_zero_mean():
    dev = octave_colouration(_impulse(), fs=48000.0, n=6)
    assert dev.shape == (6,)
    # An impulse is spectrally flat -> per-band deviations near zero.
    assert np.nanmax(np.abs(dev)) < 1e-6


# --- edc_l1 ----------------------------------------------------------------


def test_edc_l1_zero_for_identical():
    rng = np.random.default_rng(2)
    ir = rng.standard_normal(2048) * np.exp(-np.arange(2048) / 400.0)
    assert edc_l1(ir, ir) == pytest.approx(0.0, abs=1e-9)


def test_edc_l1_positive_for_different_decays():
    n = 4096
    fast = np.random.default_rng(3).standard_normal(n) * np.exp(-np.arange(n) / 100.0)
    slow = np.random.default_rng(4).standard_normal(n) * np.exp(-np.arange(n) / 800.0)
    assert edc_l1(fast, slow) > 1.0


# --- mr_stft_distance ------------------------------------------------------


def test_mr_stft_distance_zero_for_identical():
    rng = np.random.default_rng(5)
    ir = rng.standard_normal(4096)
    assert mr_stft_distance(ir, ir) == pytest.approx(0.0, abs=1e-9)


def test_mr_stft_distance_positive_for_different():
    rng = np.random.default_rng(6)
    assert mr_stft_distance(rng.standard_normal(4096), rng.standard_normal(4096)) > 0.0


# --- dataclasses -----------------------------------------------------------


def test_trainable_defaults():
    t = Trainable()
    assert t.feedback and t.input_gain and t.output_gain
    assert not t.direct


def test_output_domain_per_mode():
    assert output_domain("colorless") == "magnitude"
    assert output_domain("match_spectrogram") == "time"
    assert output_domain("match_mel_spectrogram") == "time"


def test_output_domain_rejects_unknown_mode():
    # decay is not a training mode -- it is a build property (rt= / with_decay)
    with pytest.raises(ValueError, match="unknown training mode"):
        output_domain("decay")


def test_trainlog_defaults():
    log = TrainLog()
    assert log.train_loss == [] and log.loss_log == {}
    assert log.steps_run == 0 and log.stopped_early is False
