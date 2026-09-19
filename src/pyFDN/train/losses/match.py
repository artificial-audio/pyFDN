from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

import torch

from ._targets import _CachedTarget
from .base import ResponseLoss

if TYPE_CHECKING:
    from pyFDN.train.response import Response


# --- Reductions ---

class Reduction(ABC):
    """Base class for aggregation strategies."""

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class Mean(Reduction):
    """Arithmetic mean reduction."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean()


class Sum(Reduction):
    """Sum reduction."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


# --- Distances ---

class Distance(ABC):
    """Base class for pairwise element comparison metrics."""

    @abstractmethod
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...


class SquaredError(Distance):
    """L2: Elementwise squared difference: (pred - target)^2."""

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target) ** 2


class AbsoluteError(Distance):
    """L1: Elementwise absolute difference: |pred - target|."""

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).abs()


class CircularDistance(Distance):
    r"""Circular angular metric: 1 - cos(pred - target) bounded in [0, 2]."""

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.cos(pred - target)


class AsymmetricPowerDistance(Distance):
    r"""Asymmetric penalty: peaks raised to ``peak_power``, dips always squared.

    For flatness measures, peaks ring audibly while dips are inaudible, so peaks
    are penalized harder. Expects magnitude spectra (non-negative values).

    Parameters
    ----------
    peak_power : float
        Exponent for positive deviations (peaks). Must be >= 2. Dips always use power 2.
    """

    def __init__(self, peak_power: float = 4.0) -> None:
        self.peak_power = float(peak_power)
        if self.peak_power < 2.0:
            raise ValueError(f"peak_power must be at least 2; got {self.peak_power}")

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalize prediction to target's RMS per-channel
        rms = (target**2).mean(dim=0, keepdim=True).sqrt()
        eps = torch.finfo(target.dtype).tiny
        normalized = pred / rms.clamp_min(eps)
        deviation = normalized - 1.0

        peaks = deviation.clamp_min(0.0) ** self.peak_power
        dips = deviation.clamp_max(0.0) ** 2
        return peaks + dips


class FlatnessRatioDistance(Distance):
    r"""Spectral flatness: geometric mean / arithmetic mean per frequency.

    Used to measure flatness of a spectrum. Returns MSE between the target
    flatness (default 1.0) and the pred flatness, which is the ratio of
    geometric to arithmetic mean normalized by target's ratio.

    This distance assumes pred and target are both magnitude spectra.
    """

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(pred.dtype).tiny
        # Flatness = exp(mean(log(x))) / mean(x)
        pred_geom = torch.exp(torch.log(pred.clamp_min(eps)).mean(dim=0, keepdim=True))
        pred_arith = pred.mean(dim=0, keepdim=True).clamp_min(eps)
        pred_flatness = pred_geom / pred_arith

        target_geom = torch.exp(torch.log(target.clamp_min(eps)).mean(dim=0, keepdim=True))
        target_arith = target.mean(dim=0, keepdim=True).clamp_min(eps)
        target_flatness = target_geom / target_arith

        return (pred_flatness - target_flatness) ** 2


class ConstantTargetDistance(Distance):
    """Compare pred to a fixed constant target value (ignores ref_feat).

    Useful for losses like FlatMagnitude that fit to a constant rather than
    a reference IR. The target parameter in Match is ignored; instead pred is
    compared to the constant value provided at construction.
    """

    def __init__(self, target_value: float) -> None:
        self.target_value = float(target_value)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ignore target, compare pred to constant
        return (pred - self.target_value) ** 2


class OnesTargetDistance(Distance):
    """Compare pred to a tensor of all ones (ignores ref_feat).

    Useful for losses like FlatSpectrogram that fit normalized spectra to flat
    (where flat = ones after normalization).
    """

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ignore target, compare pred (which is normalized) to ones
        return (pred - 1.0) ** 2


# --- Features ---

class Feature(ABC):
    """Base class for extracting a representation from an impulse response."""

    @abstractmethod
    def __call__(self, h: torch.Tensor, fs: float) -> torch.Tensor: ...


class Waveform(Feature):
    """Raw time-domain waveform feature."""

    def __call__(self, h: torch.Tensor, fs: float) -> torch.Tensor:
        return h


class Magnitude(Feature):
    """Frequency-domain magnitude spectrum via rfft."""

    def __init__(self, channels: str = "none") -> None:
        if channels not in ("sum", "mean", "none"):
            raise ValueError(f"channels must be 'sum', 'mean', or 'none'; got {channels!r}")
        self.channels = channels

    def __call__(self, h: torch.Tensor, fs: float) -> torch.Tensor:
        mag = torch.fft.rfft(h, dim=0).abs()
        if self.channels == "sum":
            return mag.sum(dim=1)
        if self.channels == "mean":
            return mag.mean(dim=1)
        return mag


class Phase(Feature):
    """Frequency-domain phase spectrum via rfft."""

    def __init__(self, channels: str = "none") -> None:
        if channels not in ("sum", "mean", "none"):
            raise ValueError(f"channels must be 'sum', 'mean', or 'none'; got {channels!r}")
        self.channels = channels

    def __call__(self, h: torch.Tensor, fs: float) -> torch.Tensor:
        phase = torch.angle(torch.fft.rfft(h, dim=0))
        if self.channels == "sum":
            return phase.sum(dim=1)
        if self.channels == "mean":
            return phase.mean(dim=1)
        return phase


class MelMagnitude(Feature):
    """Mel-scaled magnitude spectrum via rfft + triangular filterbank.

    Both the time-domain waveform and frequency-domain magnitude are computed
    via FFT and reduced to ``n_mels`` mel bands using ``torchaudio``'s
    triangular mel filterbank, weighting the fit towards low frequencies where
    mel spacing is finer.

    Parameters
    ----------
    n_mels : int
        Number of mel bands.
    f_min, f_max : float, optional
        Mel filterbank frequency range in Hz. ``f_max`` defaults to Nyquist.
    channels : {"sum", "mean", "none"}
        How the output channels are combined. ``"none"`` (default) keeps each
        input/output pair separate.
    """

    def __init__(
        self,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float | None = None,
        channels: str = "none",
    ) -> None:
        if channels not in ("sum", "mean", "none"):
            raise ValueError(f"channels must be 'sum', 'mean', or 'none'; got {channels!r}")
        self.n_mels = int(n_mels)
        self.f_min = float(f_min)
        self.f_max = f_max
        self.channels = channels
        self._mel_scale: Any = None
        self._key: tuple[Any, ...] | None = None

    def _build_mel_scale(self, n_freq: int, fs: float, device: Any, dtype: Any) -> Any:
        """Build and cache MelScale filterbank for the given configuration."""
        from torchaudio.transforms import MelScale

        return MelScale(
            n_mels=self.n_mels,
            sample_rate=int(fs),
            f_min=self.f_min,
            f_max=self.f_max if self.f_max is not None else fs / 2.0,
            n_stft=n_freq,
        ).to(device=device, dtype=dtype)

    def _apply_mel_scale(
        self, magnitude: torch.Tensor, fs: float, device: Any, dtype: Any
    ) -> torch.Tensor:
        """Apply mel filterbank to magnitude spectrum, caching the transform."""
        n_freq = magnitude.shape[0]
        key = (n_freq, fs, device, dtype)
        if self._mel_scale is None or self._key != key:
            self._key = key
            self._mel_scale = self._build_mel_scale(n_freq, fs, device, dtype)

        # MelScale expects (..., freq, time); ours is freq-first with no time axis,
        # so freq moves to -2 and a dummy time of 1 is appended.
        x = magnitude.permute(1, 2, 0).unsqueeze(-1)  # (n_out, n_in, freq, 1)
        mel_mag = self._mel_scale(x).squeeze(-1).permute(2, 0, 1)  # (n_mels, n_out, n_in)

        # Apply channel reduction if requested
        if self.channels == "sum":
            return mel_mag.sum(dim=1)
        if self.channels == "mean":
            return mel_mag.mean(dim=1)
        return mel_mag

    def __call__(self, h: torch.Tensor, fs: float) -> torch.Tensor:
        # Compute magnitude spectrum
        mag = torch.fft.rfft(h, dim=0).abs()  # (n_freq, n_out, n_in)
        return self._apply_mel_scale(mag, fs, h.device, h.dtype)


class SpectrogramFeature(Feature):
    """Multi-resolution Welch spectrogram (smoothed magnitude across windows).

    For each window size in ``nfft``, computes the STFT magnitude, averages over
    frames (Welch estimate), normalizes by per-frequency mean, and returns. This
    feature is resolution-independent (unlike single-FFT magnitude).

    Parameters
    ----------
    nfft : tuple of int
        STFT window sizes, each no longer than the model's nfft.
    overlap : float
        Fractional overlap between frames (0.75 -> hop of quarter window).
    """

    def __init__(
        self,
        nfft: tuple[int, ...] = (256, 512, 1024, 2048),
        overlap: float = 0.75,
    ) -> None:
        self.nfft = tuple(int(n) for n in nfft)
        self.overlap = float(overlap)
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1); got {self.overlap}")
        self._windows: dict[Any, Any] = {}

    def _window(self, n: int, device: Any, dtype: Any) -> Any:
        """Hann window for size ``n``, cached per device/dtype."""
        key = (n, device, dtype)
        if key not in self._windows:
            self._windows[key] = torch.hann_window(n, device=device, dtype=dtype)
        return self._windows[key]

    def __call__(self, h: torch.Tensor, fs: float) -> torch.Tensor:
        """Return (n_nfft, n_out, n_in) tensor stacking normalized Welch spectra."""
        # Reshape for batch STFT: (n_samples, n_out, n_in) -> (n_out * n_in, n_samples)
        signal = h.permute(1, 2, 0).reshape(-1, h.shape[0])

        spectrograms = []
        for n in self.nfft:
            if n > h.shape[0]:
                raise ValueError(
                    f"STFT window {n} is longer than the response "
                    f"({h.shape[0]} samples)"
                )
            stft = torch.stft(
                signal,
                n_fft=n,
                hop_length=max(1, int(n * (1.0 - self.overlap))),
                window=self._window(n, h.device, h.dtype),
                center=False,
                return_complex=True,
            ).abs()
            # Welch: average power over time, back to magnitude
            smoothed = (stft**2).mean(dim=-1).sqrt()  # (batch, freq)
            # Normalize by frequency mean
            level = smoothed.mean(dim=-1, keepdim=True).clamp_min(torch.finfo(smoothed.dtype).tiny)
            normalized = smoothed / level  # (batch, freq)
            spectrograms.append(normalized)

        # Stack across windows, reshape back to (n_nfft, n_out, n_in)
        stacked = torch.stack(spectrograms, dim=0)  # (n_nfft, batch, freq)
        return stacked.permute(0, 2, 1).reshape(len(self.nfft), h.shape[1], h.shape[2])


class PhaseSpectrogramFeature(Feature):
    """Multi-resolution STFT phase (circular distance ready).

    For each window size, extracts STFT phase and stacks across windows.

    Parameters
    ----------
    nfft : tuple of int
        STFT window sizes.
    overlap : float
        Fractional overlap between frames.
    """

    def __init__(
        self,
        nfft: tuple[int, ...] = (256, 512, 1024, 2048),
        overlap: float = 0.75,
    ) -> None:
        self.nfft = tuple(int(n) for n in nfft)
        self.overlap = float(overlap)
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1); got {self.overlap}")
        self._windows: dict[Any, Any] = {}

    def _window(self, n: int, device: Any, dtype: Any) -> Any:
        """Hann window for size ``n``, cached per device/dtype."""
        key = (n, device, dtype)
        if key not in self._windows:
            self._windows[key] = torch.hann_window(n, device=device, dtype=dtype)
        return self._windows[key]

    def __call__(self, h: torch.Tensor, fs: float) -> torch.Tensor:
        """Return (n_nfft, n_out, n_in, n_frames, n_freq) phase tensor."""
        # Reshape for batch STFT: (n_samples, n_out, n_in) -> (n_out * n_in, n_samples)
        signal = h.permute(1, 2, 0).reshape(-1, h.shape[0])

        phases = []
        for n in self.nfft:
            if n > h.shape[0]:
                raise ValueError(
                    f"STFT window {n} is longer than the response ({h.shape[0]} samples)"
                )
            stft = torch.stft(
                signal,
                n_fft=n,
                hop_length=max(1, int(n * (1.0 - self.overlap))),
                window=self._window(n, h.device, h.dtype),
                center=False,
                return_complex=True,
            )
            phase = torch.angle(stft)  # (batch, freq, frames)
            phases.append(phase)

        # Stack: (n_nfft, batch, freq, frames)
        stacked = torch.stack(phases, dim=0)
        # Reshape: (n_nfft, n_out, n_in, freq, frames)
        return stacked.permute(0, 3, 1).reshape(len(self.nfft), h.shape[1], h.shape[2],
                                                stacked.shape[2], stacked.shape[3])


# --- Composed Match class ---

class Match(ResponseLoss):
    r"""Compares a model's impulse response to a target using three steps:

        1. **Feature**: Extracts representations from both the model response and the
        target (e.g., raw waveform, magnitude spectrum, or phase).
        2. **Distance**: Computes the differences between the extracted
        features (e.g., squared error or circular phase difference).
        3. **Reduction**: Aggregates the elementwise differences into a scalar loss
        tensor (e.g., mean or sum).

        The reference target is aligned in length, shape, device, and dtype 
        using ``_CachedTarget``.

        Parameters
        ----------
        target : Any
            Reference impulse response (NumPy array, PyTorch tensor, or list).
        feature : Feature or callable
            Extracts the representation to compare: ``(h, fs) -> torch.Tensor``.
        distance : Distance or callable, default SquaredError()
            Calculates error between two features: ``(pred, target) -> torch.Tensor``.
        reduction : Reduction or callable, default Mean()
            Turns the error tensor into a single scalar: ``(diff) -> torch.Tensor``.
    """

    def __init__(
        self,
        target: Any,
        feature: Feature | Callable[[torch.Tensor, float], torch.Tensor],
        distance: Distance | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = SquaredError(),
        reduction: Reduction | Callable[[torch.Tensor], torch.Tensor] = Mean(),
    ) -> None:
        self._target = _CachedTarget(target)
        self.feature = feature
        self.distance = distance
        self.reduction = reduction

    def __call__(self, response: Response) -> torch.Tensor:
        ref_h = self._target(response)
        pred_feat = self.feature(response.h, response.fs)
        ref_feat = self.feature(ref_h, response.fs)

        diff = self.distance(pred_feat, ref_feat)
        return self.reduction(diff)