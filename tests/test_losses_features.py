"""Tests for the composable Feature classes in pyFDN.train.losses.features.

Covers SpectrogramFeature and PhaseSpectrogramFeature
"""

import math

import pytest

pytest.importorskip("torch")

import torch

from pyFDN.train.losses.features import PhaseSpectrogramFeature, SpectrogramFeature

# --- SpectrogramFeature -------------------------------------------------------


def _manual_spectrogram(
    h: torch.Tensor, nfft: tuple[int, ...], overlap: float
) -> torch.Tensor:
    """The same Welch spectrogram, computed one channel at a time with plain
    indexing (no stacking/reshaping), to use as a ground truth to compare against."""
    n_out, n_in = h.shape[1], h.shape[2]
    max_freq = max(n // 2 + 1 for n in nfft)
    out = torch.empty(len(nfft), n_out, n_in, max_freq, dtype=h.dtype)
    for i, n in enumerate(nfft):
        hop = max(1, int(n * (1.0 - overlap)))
        window = torch.hann_window(n, dtype=h.dtype)
        for o in range(n_out):
            for c in range(n_in):
                stft = torch.stft(
                    h[:, o, c],
                    n_fft=n,
                    hop_length=hop,
                    window=window,
                    center=False,
                    return_complex=True,
                ).abs()
                smoothed = (stft**2).mean(dim=-1).sqrt()
                level = smoothed.mean().clamp_min(torch.finfo(smoothed.dtype).tiny)
                normalized = smoothed / level
                out[i, o, c, : normalized.shape[0]] = normalized
                out[i, o, c, normalized.shape[0] :] = 1.0
    return out


def test_spectrogram_feature_preserves_per_channel_identity():
    """For a multi-channel (MIMO) response, each channel's own spectrogram
    must end up at its own (out, in) index in the output, not mixed with
    another channel's."""
    torch.manual_seed(0)
    h = torch.randn(128, 2, 2)
    nfft = (16, 32)
    overlap = 0.5

    feature = SpectrogramFeature(nfft=nfft, overlap=overlap)
    out = feature(h, fs=48000.0)
    expected = _manual_spectrogram(h, nfft, overlap)

    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_spectrogram_feature_pads_the_shorter_window_with_ones():
    """A smaller window has fewer frequency bins than the largest one; the
    missing bins are padded with 1.0 (= "flat"), so they don't skew a
    flatness loss built on this feature."""
    h = torch.randn(64, 1, 1)
    feature = SpectrogramFeature(nfft=(8, 16), overlap=0.5)
    out = feature(h, fs=48000.0)

    short_freq = 8 // 2 + 1  # real bins from the n=8 window
    long_freq = 16 // 2 + 1  # total width after padding to the n=16 window
    padding = out[0, 0, 0, short_freq:long_freq]
    assert torch.allclose(padding, torch.ones_like(padding))


def test_spectrogram_feature_normalizes_each_window_to_a_unit_mean():
    """Each window's spectrum is divided by its own mean, so its real
    (unpadded) bins must average to exactly 1 -- regardless of what the
    input signal actually contains."""
    h = torch.randn(128, 1, 1)
    nfft = (16, 32)
    feature = SpectrogramFeature(nfft=nfft, overlap=0.75)
    out = feature(h, fs=48000.0)

    for i, n in enumerate(nfft):
        n_freq = n // 2 + 1
        mean = out[i, ..., :n_freq].mean(dim=-1)
        torch.testing.assert_close(mean, torch.ones_like(mean), atol=1e-5, rtol=1e-5)


def test_spectrogram_feature_group_counts_match_each_windows_element_count():
    """The feature records how many real (unpadded) elements each window
    contributed, in `_group_counts`. MeanPerGroup relies on this to weight
    each scale correctly, so it must match the true per-window element
    count exactly."""
    h = torch.randn(64, 2, 3)
    nfft = (8, 16)
    feature = SpectrogramFeature(nfft=nfft, overlap=0.5)
    feature(h, fs=48000.0)

    expected = [(n // 2 + 1) * 2 * 3 for n in nfft]
    assert feature._group_counts == expected


def test_spectrogram_feature_rejects_a_window_longer_than_the_response():
    """A window longer than the response itself can't be computed and must
    raise, rather than silently doing something wrong."""
    h = torch.randn(32, 1, 1)
    feature = SpectrogramFeature(nfft=(64,))
    with pytest.raises(ValueError, match="longer than the response"):
        feature(h, fs=48000.0)


# --- PhaseSpectrogramFeature ---------------------------------------------------


def _stft_frames(n_samples: int, n_fft: int, hop: int) -> int:
    """How many STFT frames torch.stft(..., center=False) produces for these settings."""
    x = torch.zeros(n_samples)
    window = torch.hann_window(n_fft)
    return torch.stft(
        x, n_fft=n_fft, hop_length=hop, window=window, center=False, return_complex=True
    ).shape[-1]


def test_phase_spectrogram_feature_shape_spans_every_window_and_the_widest_axes():
    """The output holds one entry per nfft window, padded out to the widest
    frequency axis and the most frames across all the windows."""
    h = torch.randn(64, 2, 3)
    nfft, overlap = (8, 16), 0.5
    feature = PhaseSpectrogramFeature(nfft=nfft, overlap=overlap)
    out = feature(h, fs=48000.0)

    max_freq = 16 // 2 + 1
    frames = [_stft_frames(64, n, max(1, int(n * (1 - overlap)))) for n in nfft]
    assert out.shape == (2, 2, 3, max_freq, max(frames))


def test_phase_spectrogram_feature_matches_direct_stft_phase():
    """In the simplest possible case -- one channel, one window, no padding
    -- the output must equal torch.angle(torch.stft(h)) exactly.
    Regression test for the same reshape bug as
    test_spectrogram_feature_preserves_per_channel_identity, but here it
    used to corrupt even a single channel by entangling the frequency and
    frame axes."""
    torch.manual_seed(1)
    fs, n_fft, overlap = 48000.0, 256, 0.5
    h = torch.randn(2048, 1, 1)

    feature = PhaseSpectrogramFeature(nfft=(n_fft,), overlap=overlap)
    out = feature(h, fs=fs)

    hop = max(1, int(n_fft * (1.0 - overlap)))
    window = torch.hann_window(n_fft)
    expected = torch.angle(
        torch.stft(
            h[:, 0, 0],
            n_fft=n_fft,
            hop_length=hop,
            window=window,
            center=False,
            return_complex=True,
        )
    )

    torch.testing.assert_close(out[0, 0, 0], expected, atol=1e-5, rtol=1e-5)


def test_phase_spectrogram_feature_values_are_wrapped():
    """Phase is always returned in (-pi, pi], the range CircularDistance
    assumes when comparing two phase spectrograms."""
    h = torch.randn(64, 2, 2)
    feature = PhaseSpectrogramFeature(nfft=(16,), overlap=0.5)
    out = feature(h, fs=48000.0)
    assert torch.all(out > -math.pi - 1e-6)
    assert torch.all(out <= math.pi + 1e-6)


def test_phase_spectrogram_feature_group_counts_match_each_windows_element_count():
    """Same idea as the SpectrogramFeature version: `_group_counts` must
    equal the true number of real (unpadded) elements per window, including
    the frame axis this feature has and SpectrogramFeature doesn't."""
    h = torch.randn(64, 2, 3)
    nfft, overlap = (8, 16), 0.5
    feature = PhaseSpectrogramFeature(nfft=nfft, overlap=overlap)
    feature(h, fs=48000.0)

    expected = [
        (n // 2 + 1) * 2 * 3 * _stft_frames(64, n, max(1, int(n * (1 - overlap))))
        for n in nfft
    ]
    assert feature._group_counts == expected


def test_phase_spectrogram_feature_rejects_a_window_longer_than_the_response():
    """Same guard as SpectrogramFeature: a window longer than the response
    must raise instead of computing something meaningless."""
    h = torch.randn(32, 1, 1)
    feature = PhaseSpectrogramFeature(nfft=(64,))
    with pytest.raises(ValueError, match="longer than the response"):
        feature(h, fs=48000.0)


def test_phase_spectrogram_feature_sees_a_quarter_cycle_phase_shift():
    """A cosine and a sine at the same frequency are pi/2 apart in phase at
    every bin; reading that same pi/2 gap back at one arbitrary bin is a
    quick sanity check that this feature is really reading phase (not
    magnitude, and not something constant)."""
    fs, n_samples, n_fft = 48000.0, 2048, 256
    freq_bin = 8
    f = freq_bin * fs / n_fft
    t = torch.arange(n_samples, dtype=torch.float64) / fs
    cos_ir = torch.cos(2 * math.pi * f * t).float().reshape(-1, 1, 1)
    sin_ir = torch.sin(2 * math.pi * f * t).float().reshape(-1, 1, 1)

    feature = PhaseSpectrogramFeature(nfft=(n_fft,), overlap=0.5)
    cos_phase = feature(cos_ir, fs=fs)[0, 0, 0, freq_bin, 0]
    sin_phase = feature(sin_ir, fs=fs)[0, 0, 0, freq_bin, 0]

    diff = cos_phase - sin_phase
    wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
    assert wrapped.abs().item() == pytest.approx(math.pi / 2, abs=0.05)


# --- Match and the losses built on it -------------------------------------------

import numpy as np  # noqa: E402

import pyFDN.train.losses as losses  # noqa: E402
from pyFDN.train.losses.base import Sum as LossSum  # noqa: E402
from pyFDN.train.losses.distances import FlatnessRatioDistance  # noqa: E402
from pyFDN.train.losses.features import (  # noqa: E402
    OCTAVE_EDGES,
    EnergyDecayCurve,
    Phase,
)
from pyFDN.train.response import Response  # noqa: E402


def _decaying(n=4096, n_out=2, n_in=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    envelope = torch.exp(-torch.arange(n, dtype=torch.float64) / 800.0)
    return (
        torch.randn(n, n_out, n_in, generator=g, dtype=torch.float64)
        * envelope[:, None, None]
    )


def test_losses_sum_is_still_the_loss_composition():
    """The Sum reduction must not shadow the composite loss in the package."""
    assert losses.Sum is LossSum


def test_spectral_flatness_fits_the_requested_target():
    """The target flatness is honoured: the loss is (F - target)^2."""
    h = _decaying()
    flatness = FlatnessRatioDistance.flatness(torch.fft.rfft(h, dim=0).abs())
    response = Response(h=h, fs=48000.0)
    for target in (1.0, 0.5, 0.0):
        expected = ((flatness - target) ** 2).mean()
        value = losses.SpectralFlatness(target=target)(response)
        torch.testing.assert_close(value, expected)


def test_spectral_flatness_rejects_a_target_outside_zero_one():
    with pytest.raises(ValueError, match="target_flatness"):
        losses.SpectralFlatness(target=1.5)


def test_spectral_flatness_does_not_score_silence_as_flat():
    """A silent response has flatness 0, so the flat target costs 1, not 0."""
    silent = Response(h=torch.zeros(1024, 1, 1, dtype=torch.float64), fs=48000.0)
    assert float(losses.SpectralFlatness()(silent)) == pytest.approx(1.0)


def test_phase_sum_is_the_phase_of_the_summed_response():
    """channels="sum" takes the angle of the summed spectra, not a sum of angles."""
    h = _decaying(n=512)
    phase = Phase(channels="sum")(h, 48000.0)
    expected = torch.angle(torch.fft.rfft(h.sum(dim=1), dim=0))
    torch.testing.assert_close(phase, expected)
    torch.testing.assert_close(Phase(channels="mean")(h, 48000.0), expected)


def test_match_computes_the_reference_feature_once_per_response_shape():
    calls = []

    class Counting(losses.Waveform):
        def __call__(self, h, fs):
            calls.append(h.shape)
            return h

    h = _decaying(n=256)
    loss = losses.Match(target=h.numpy(), feature=Counting())
    response = Response(h=h, fs=48000.0)
    for _ in range(3):
        loss(response)
    # three predictions, one reference
    assert len(calls) == 4
    # a response of another length rebuilds the reference
    loss(Response(h=_decaying(n=512), fs=48000.0))
    assert len(calls) == 6


def test_energy_decay_curve_feature_defaults_to_octave_bands():
    feature = EnergyDecayCurve(window=1024)
    assert feature.bands == OCTAVE_EDGES
    out = feature(_decaying(n=8192, n_out=1, n_in=1), 48000.0)
    assert out.shape[1] == len(OCTAVE_EDGES) - 1


def test_match_energy_decay_is_rms_over_the_masked_entries():
    """Only reference entries above floor_db count, averaged over those alone."""
    h, target = _decaying(n=16384, seed=1), _decaying(n=16384, seed=2)
    loss = losses.MatchEnergyDecay(target.numpy(), window=1024, floor_db=-30.0)
    response = Response(h=h, fs=48000.0)
    feature = EnergyDecayCurve(window=1024, bands=loss.bands)
    pred, ref = feature(h, 48000.0), feature(target, 48000.0)
    mask = ref > -30.0
    expected = ((pred[mask] - ref[mask]) ** 2).mean().sqrt()
    torch.testing.assert_close(loss(response), expected)


@pytest.mark.parametrize(
    "make_loss",
    [
        lambda t: losses.MatchPhase(t),
        lambda t: losses.MatchPhaseSpectrogram(t, nfft=(256, 512)),
        lambda t: losses.MatchMelMagnitude(t, n_mels=32),
        lambda t: losses.SpectralFlatness(),
    ],
)
@pytest.mark.parametrize("silent", ["prediction", "target"])
def test_new_losses_have_finite_gradients_on_silent_inputs(make_loss, silent):
    h, target = _decaying(n=2048), _decaying(n=2048, seed=3)
    if silent == "prediction":
        h = torch.zeros_like(h)
    else:
        target = torch.zeros_like(target)
    h = h.clone().requires_grad_()
    value = make_loss(np.asarray(target))(Response(h=h, fs=48000.0))
    value.backward()
    assert torch.isfinite(value)
    assert torch.isfinite(h.grad).all()
