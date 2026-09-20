from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


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
        n_freq = magnitude.shape[0]
        key = (n_freq, fs, device, dtype)
        if self._mel_scale is None or self._key != key:
            self._key = key
            self._mel_scale = self._build_mel_scale(n_freq, fs, device, dtype)
        x = magnitude.permute(1, 2, 0).unsqueeze(-1)
        mel_mag = self._mel_scale(x).squeeze(-1).permute(2, 0, 1)
        if self.channels == "sum":
            return mel_mag.sum(dim=1)
        if self.channels == "mean":
            return mel_mag.mean(dim=1)
        return mel_mag

    def __call__(self, h: torch.Tensor, fs: float) -> torch.Tensor:
        mag = torch.fft.rfft(h, dim=0).abs()
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
        key = (n, device, dtype)
        if key not in self._windows:
            self._windows[key] = torch.hann_window(n, device=device, dtype=dtype)
        return self._windows[key]

    def __call__(self, h: torch.Tensor, fs: float) -> torch.Tensor:
        """Return (n_nfft, n_out, n_in, max_freq) tensor stacking normalized Welch spectra."""
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
            smoothed = (stft**2).mean(dim=-1).sqrt()
            level = smoothed.mean(dim=-1, keepdim=True).clamp_min(torch.finfo(smoothed.dtype).tiny)
            normalized = smoothed / level
            spectrograms.append(normalized)

        max_freq = max(s.shape[1] for s in spectrograms)
        padded = [torch.nn.functional.pad(s, (0, max_freq - s.shape[1]), value=1.0)
                  for s in spectrograms]
        stacked = torch.stack(padded, dim=0)
        result = stacked.reshape(len(self.nfft), h.shape[1], h.shape[2], max_freq)
        self._group_counts = [s.shape[0] * s.shape[1] for s in spectrograms]
        return result


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
        key = (n, device, dtype)
        if key not in self._windows:
            self._windows[key] = torch.hann_window(n, device=device, dtype=dtype)
        return self._windows[key]

    def __call__(self, h: torch.Tensor, fs: float) -> torch.Tensor:
        """Return (n_nfft, n_out, n_in, max_freq, max_frames) phase tensor."""
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
            phase = torch.angle(stft)
            phases.append(phase)

        max_freq = max(p.shape[1] for p in phases)
        max_frames = max(p.shape[2] for p in phases)
        padded = [
            torch.nn.functional.pad(p, (0, max_frames - p.shape[2], 0, max_freq - p.shape[1]))
            for p in phases
        ]
        stacked = torch.stack(padded, dim=0)
        result = stacked.reshape(len(self.nfft), h.shape[1], h.shape[2], max_freq, max_frames)
        self._group_counts = [p.shape[0] * p.shape[1] * p.shape[2] for p in phases]
        return result
