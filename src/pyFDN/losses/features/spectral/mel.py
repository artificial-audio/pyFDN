import torch
import torch.nn as nn
import torchaudio.transforms as T
from typing import Optional, Callable, Dict, Any


class MelSpectrogramFeature(nn.Module):
    """
    Differentiable Mel-Spectrogram feature transform wrapping torchaudio.transforms.MelSpectrogram.

    Input:
        ir (torch.Tensor): Audio / RIR tensor of shape (..., time).
    Output:
        torch.Tensor: Mel-spectrogram of shape (..., n_mels, frames).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Callable[..., torch.Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[Dict[str, Any]] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        log: bool = False,
        eps: float = 1e-7,
    ) -> None:
        """
        Args:
            sample_rate: Sample rate of audio signal.
            n_fft: Size of FFT, creates n_fft // 2 + 1 bins.
            win_length: Window size (defaults to n_fft).
            hop_length: Length of hop between STFT windows (defaults to win_length // 2).
            f_min: Minimum frequency cutoff.
            f_max: Maximum frequency cutoff (defaults to sample_rate // 2).
            pad: Two-sided padding of signal.
            n_mels: Number of Mel filterbanks.
            window_fn: Function to create a window tensor.
            power: Exponent for magnitude spectrogram (> 0, e.g., 1: magnitude, 2: power).
            normalized: Whether to normalize by magnitude after STFT.
            wkwargs: Optional dictionary of keyword arguments for window function.
            center: Whether to pad waveform on both sides so frame t is centered at time t.
            pad_mode: Controls padding method when center is True.
            norm: If 'slaney', normalizes triangular mel weights by band width.
            mel_scale: Scale formulation to use: 'htk' or 'slaney'.
            log: If True, applies logarithmic compression: log(Mel + eps).
            eps: Epsilon to prevent log(0) and avoid NaN gradients.
        """
        super().__init__()
        self.log = log
        self.eps = eps

        self.transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            pad=pad,
            n_mels=n_mels,
            window_fn=window_fn,
            power=power,
            normalized=normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            norm=norm,
            mel_scale=mel_scale,
        )

    def forward(self, ir: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ir (torch.Tensor): Audio waveform tensor of shape (..., time).

        Returns:
            torch.Tensor: Mel spectrogram of shape (..., n_mels, frames).
        """
        mel = self.transform(ir)

        if self.log:
            mel = torch.log(mel + self.eps)

        return mel