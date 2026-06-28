"""Tests for pyFDN.metric.quality (no-reference RIR quality proxies).

Each metric is exercised on synthetic signals whose ground-truth artifact level
is known by construction (clean anchor vs. obviously-broken anchor), checking
both the expected magnitude and the monotonic ordering between clean and broken.
"""

import numpy as np
import pytest

from pyFDN.generate.random_orthogonal import random_orthogonal
from pyFDN.metric.quality import (
    decay_fit_residual,
    envelope_autocorrelation_peak,
    modal_excitation_magnitudes,
    modal_excitation_peak,
    modal_excitation_spread,
    normalized_echo_density,
    signal_autocorrelation_peak,
    spectral_flatness,
    spectral_magnitude_spread,
    tail_kurtosis,
)

FS = 48_000


# ----------------------------------------------------------------------------
# Synthetic signal helpers
# ----------------------------------------------------------------------------
def _decaying_noise(rng, fs, t60, length_s, level=1.0):
    """Gaussian noise with a clean single-rate exponential decay (-60 dB at t60)."""
    n = np.arange(int(length_s * fs))
    env = 10.0 ** (-3.0 * n / (t60 * fs))  # amplitude envelope
    return rng.standard_normal(n.size) * env * level


def _decaying_sine(fs, freq, t60, length_s):
    """A single decaying sinusoid (one strongly excited mode -> colored)."""
    n = np.arange(int(length_s * fs))
    env = 10.0 ** (-3.0 * n / (t60 * fs))
    return np.sin(2.0 * np.pi * freq * n / fs) * env


def _velvet_noise(rng, fs, length_s, density):
    """Sparse +/-1 impulses at a given density (low echo density / sputtery)."""
    n = int(length_s * fs)
    sig = np.zeros(n)
    n_imp = int(density * n)
    idx = rng.choice(n, size=n_imp, replace=False)
    sig[idx] = rng.choice([-1.0, 1.0], size=n_imp)
    return sig


# ============================================================================
# Spectral flatness  (coloration; geometric/arithmetic mean of power spectrum)
# ============================================================================
def test_spectral_flatness_white_noise_is_high():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(FS)
    # White noise periodogram -> flatness concentrates near exp(-gamma) ~ 0.56.
    assert spectral_flatness(x) > 0.45


def test_spectral_flatness_pure_tone_is_near_zero():
    n = np.arange(FS)
    bin_aligned = 1000  # integer number of cycles -> energy in one bin
    x = np.sin(2.0 * np.pi * bin_aligned * n / FS)
    assert spectral_flatness(x) < 0.01


def test_spectral_flatness_monotonic_white_gt_colored_gt_tone():
    rng = np.random.default_rng(1)
    white = rng.standard_normal(FS)
    colored = _decaying_sine(FS, 1000.0, t60=0.5, length_s=1.0)
    n = np.arange(FS)
    tone = np.sin(2.0 * np.pi * 1000 * n / FS)
    assert spectral_flatness(white) > spectral_flatness(colored)
    assert spectral_flatness(colored) > spectral_flatness(tone)


# ============================================================================
# Spectral magnitude spread  (coloration; std of magnitude response in dB)
# ============================================================================
def test_spectral_magnitude_spread_resonant_exceeds_white():
    rng = np.random.default_rng(2)
    white = rng.standard_normal(FS)
    resonant = _decaying_sine(FS, 1000.0, t60=0.8, length_s=1.0)
    assert spectral_magnitude_spread(resonant) > spectral_magnitude_spread(white)


def test_spectral_magnitude_spread_white_is_moderate():
    rng = np.random.default_rng(3)
    white = rng.standard_normal(FS)
    # 20*log10|X| for white Gaussian noise has a fixed std of ~5.6 dB.
    assert 3.0 < spectral_magnitude_spread(white) < 8.0


# ============================================================================
# Tail kurtosis  (non-Gaussianity; excess kurtosis of decay-compensated tail)
# ============================================================================
def test_tail_kurtosis_stationary_gaussian_near_zero():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(200_000)
    assert abs(tail_kurtosis(x, FS)) < 0.2


def test_tail_kurtosis_sparse_tail_is_large():
    rng = np.random.default_rng(5)
    x = _velvet_noise(rng, FS, length_s=1.0, density=0.02)
    assert tail_kurtosis(x, FS, normalize_decay=False) > 5.0


def test_tail_kurtosis_decay_compensation_reduces_kurtosis():
    rng = np.random.default_rng(6)
    x = _decaying_noise(rng, FS, t60=0.5, length_s=2.0)
    raw = tail_kurtosis(x, FS, normalize_decay=False)
    compensated = tail_kurtosis(x, FS, normalize_decay=True)
    assert compensated < raw - 1.0
    assert abs(compensated) < 0.6


# ============================================================================
# Envelope autocorrelation peak  (flutter / periodicity)
# ============================================================================
def test_envelope_autocorrelation_diffuse_is_low():
    # Must hold across realizations -- a single-seed pass hides spurious peaks.
    for seed in range(8):
        x = _decaying_noise(np.random.default_rng(seed), FS, t60=1.0, length_s=2.0)
        assert envelope_autocorrelation_peak(x, FS) < 0.3, f"seed {seed}"


def test_envelope_autocorrelation_periodic_is_high_with_correct_period():
    rng = np.random.default_rng(8)
    n = np.arange(2 * FS)
    period_s = 0.01  # 100 Hz flutter
    decay = 10.0 ** (-3.0 * n / (2.0 * FS))
    mod = 1.0 + 0.9 * np.cos(2.0 * np.pi * n / (period_s * FS))
    x = rng.standard_normal(n.size) * decay * mod
    peak, period = envelope_autocorrelation_peak(x, FS, return_period=True)
    assert peak > 0.5
    assert abs(period - period_s) < 0.0015


def test_envelope_autocorrelation_detects_am_with_bandpass():
    # Carrier band-pass is off by default; it can be enabled for the flutter band.
    rng = np.random.default_rng(11)
    n = np.arange(2 * FS)
    mod = 1.0 + 0.9 * np.cos(2.0 * np.pi * n / (0.01 * FS))
    x = rng.standard_normal(n.size) * 10.0 ** (-3.0 * n / (2.0 * FS)) * mod
    assert envelope_autocorrelation_peak(x, FS, bandpass=True) > 0.5


def test_envelope_autocorrelation_periodic_exceeds_diffuse():
    rng = np.random.default_rng(9)
    diffuse = _decaying_noise(rng, FS, t60=1.0, length_s=2.0)
    n = np.arange(2 * FS)
    mod = 1.0 + 0.9 * np.cos(2.0 * np.pi * n / (0.01 * FS))
    periodic = rng.standard_normal(n.size) * 10.0 ** (-3.0 * n / (2.0 * FS)) * mod
    assert envelope_autocorrelation_peak(periodic, FS) > envelope_autocorrelation_peak(
        diffuse, FS
    )


# ============================================================================
# Decay fit residual  (irregular decay; RMS dB deviation from single-slope fit)
# ============================================================================
def test_decay_fit_residual_clean_exponential_is_small():
    rng = np.random.default_rng(10)
    x = _decaying_noise(rng, FS, t60=1.5, length_s=2.0)
    assert decay_fit_residual(x, FS) < 1.0


def test_decay_fit_residual_double_slope_exceeds_clean():
    rng = np.random.default_rng(11)
    clean = _decaying_noise(rng, FS, t60=1.5, length_s=2.0)
    # Two-rate energy decay whose knee sits inside the -5..-35 dB fit range:
    # a steep early term plus a weaker, much slower tail.
    n = np.arange(3 * FS)
    energy = np.exp(-n / (0.1 * FS)) + 0.03 * np.exp(-n / (1.5 * FS))
    double = np.random.default_rng(12).standard_normal(n.size) * np.sqrt(energy)
    res_double = decay_fit_residual(double, FS)
    assert res_double > decay_fit_residual(clean, FS)
    assert res_double > 1.0


# ============================================================================
# Normalized echo density  (texture; Abel & Huang NED summary)
# ============================================================================
def test_normalized_echo_density_diffuse_near_one():
    rng = np.random.default_rng(13)
    x = rng.standard_normal(FS)
    assert 0.85 < normalized_echo_density(x, FS) < 1.15


def test_normalized_echo_density_sparse_below_one():
    rng = np.random.default_rng(14)
    x = _velvet_noise(rng, FS, length_s=1.0, density=0.05)
    assert normalized_echo_density(x, FS) < 0.6


def test_normalized_echo_density_diffuse_exceeds_sparse():
    rng = np.random.default_rng(15)
    diffuse = rng.standard_normal(FS)
    sparse = _velvet_noise(rng, FS, length_s=1.0, density=0.05)
    assert normalized_echo_density(diffuse, FS) > normalized_echo_density(sparse, FS)


# ============================================================================
# Signal-domain autocorrelation peak  (waveform periodicity / tonal ringing)
# ============================================================================
def test_signal_autocorrelation_diffuse_is_low():
    rng = np.random.default_rng(20)
    x = _decaying_noise(rng, FS, t60=1.0, length_s=1.0)
    assert signal_autocorrelation_peak(x, FS) < 0.3


def test_signal_autocorrelation_tonal_is_high_with_period():
    n = np.arange(FS)
    tonal = np.sin(2.0 * np.pi * 1000.0 * n / FS) * 10.0 ** (-3.0 * n / (0.5 * FS))
    peak, period = signal_autocorrelation_peak(tonal, FS, return_period=True)
    assert peak > 0.5
    assert abs(period - 0.001) < 0.0002  # 1 kHz -> 1 ms period


def test_signal_autocorrelation_tonal_exceeds_diffuse():
    rng = np.random.default_rng(21)
    diffuse = _decaying_noise(rng, FS, t60=1.0, length_s=1.0)
    n = np.arange(FS)
    tonal = np.sin(2.0 * np.pi * 1000.0 * n / FS) * 10.0 ** (-3.0 * n / (0.5 * FS))
    assert signal_autocorrelation_peak(tonal, FS) > signal_autocorrelation_peak(
        diffuse, FS
    )


def _flutter_train(fs, repetition_ms, fc=900.0, ping_ms=3.0, t60=1.5, length_s=2.0):
    """A band-limited ping repeated every ``repetition_ms`` (a flutter tail)."""
    m = int(ping_ms * fs / 1000)
    ping = np.sin(2.0 * np.pi * fc * np.arange(m) / fs) * np.hanning(m)
    n = int(length_s * fs)
    x = np.zeros(n)
    step = int(repetition_ms * fs / 1000)
    for i in range(0, n - m, step):
        x[i : i + m] += ping * 10.0 ** (-3.0 * i / (t60 * fs))
    return x


def test_signal_autocorrelation_detects_flutter_repetition():
    x = _flutter_train(FS, repetition_ms=20.0)
    peak, period = signal_autocorrelation_peak(x, FS, return_period=True)
    assert peak > 0.5
    assert abs(period - 0.02) < 0.001  # 20 ms repetition time


def test_signal_autocorrelation_bandpass_rejects_out_of_band_tone():
    n = np.arange(FS)
    tone = np.sin(2.0 * np.pi * 6000.0 * n / FS) * 10.0 ** (-3.0 * n / (0.5 * FS))
    # 6 kHz is above the 0.25-2 kHz flutter band -> rejected.
    assert signal_autocorrelation_peak(tone, FS) < 0.3
    # disabling the bandpass exposes the periodicity again.
    assert signal_autocorrelation_peak(tone, FS, bandpass=False) > 0.5


# ============================================================================
# Analytical modal excitation  (coloration from the FDN model, not an RIR)
# ============================================================================
def _flamo_fdn(delays, feedback, b=None):
    """A FLAMO single-in single-out FDN model from a feedback matrix."""
    pytest.importorskip("flamo")
    torch = pytest.importorskip("torch")
    from pyFDN.translate.dss_to_flamo import dss_to_flamo

    n = len(delays)
    b_gain = np.eye(n, 1) if b is None else np.asarray(b, dtype=float)
    return dss_to_flamo(
        A=np.asarray(feedback, dtype=float),
        B=b_gain,
        C=np.eye(1, n),
        D=np.zeros((1, 1)),
        m=np.asarray(delays, dtype=int),
        Fs=1.0,
        nfft=4096,
        shell=False,
        dtype=torch.float64,
    )


def test_modal_excitation_unmixed_exceeds_mixed():
    np.random.seed(0)
    delays = [7, 11, 13, 17]
    mixed = _flamo_fdn(delays, 0.9 * random_orthogonal(4))  # unitary mixing
    unmixed = _flamo_fdn(delays, 0.9 * np.eye(4))  # no mixing -> colored
    assert modal_excitation_spread(unmixed) > modal_excitation_spread(mixed)
    assert modal_excitation_peak(unmixed) > modal_excitation_peak(mixed)


def test_modal_excitation_matches_dss_reference():
    # The model-based metric must equal a direct pole-residue computation.
    from pyFDN.translate.dss_to_pr import dss_to_pr

    delays = np.array([4, 5, 6], dtype=int)
    a = np.array([[0.25, -0.1, 0.05], [0.15, 0.3, -0.2], [0.1, 0.05, 0.2]])
    model = _flamo_fdn(delays, a)

    mags_model = np.sort(modal_excitation_magnitudes(model))
    res, _p, _d, conj, _m = dss_to_pr(
        delays, a, np.eye(3, 1), np.eye(1, 3), np.zeros((1, 1)), mode="roots"
    )
    mag = np.sqrt(np.sum(np.abs(res) ** 2, axis=(1, 2)))
    mags_ref = np.sort(20.0 * np.log10(np.repeat(mag, np.where(conj, 2, 1)) + 1e-12))

    assert mags_model.size == mags_ref.size == int(np.sum(delays))
    np.testing.assert_allclose(mags_model, mags_ref, atol=1e-6)


def test_modal_excitation_summaries_scale_invariant():
    np.random.seed(2)
    delays = [7, 11, 13, 17]
    feedback = 0.9 * random_orthogonal(4)
    base = _flamo_fdn(delays, feedback)
    scaled = _flamo_fdn(delays, feedback, b=10.0 * np.eye(4, 1))  # 20 dB louder input
    assert abs(modal_excitation_spread(base) - modal_excitation_spread(scaled)) < 1e-4
    assert abs(modal_excitation_peak(base) - modal_excitation_peak(scaled)) < 1e-4


# ============================================================================
# Input validation
# ============================================================================
def test_metrics_reject_multichannel_input():
    stereo = np.zeros((FS, 2))
    with pytest.raises(ValueError):
        spectral_flatness(stereo)
