"""Differentiable RIR feature extractors.

Extract time-domain impulse responses (IRs) into feature representations for
use in training objectives and loss functions. All features are fully
differentiable PyTorch operations suitable for backpropagation-based training.

Features:
- STFT magnitude spectrogram
- STFT phase spectrogram
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def stft_magnitude(
    ir: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int | None = None,
    window: str = "hann",
) -> torch.Tensor:
    r"""Compute STFT magnitude spectrogram from time-domain IR.

    Converts impulse response(s) to STFT magnitude representation, optionally
    normalized to [0, 1] per frequency bin.

    Parameters
    ----------
    ir : torch.Tensor
        Time-domain impulse response, shape ``(n_samples,)`` or
        ``(batch, n_samples)`` or ``(batch, n_samples, n_channels)``.
    n_fft : int, default 2048
        FFT size.
    hop_length : int, optional
        Hop length; defaults to ``n_fft // 4``.
    window : str, default "hann"
        Window function name ('hann', 'hamming', 'blackman', etc.).
    Returns
    -------
    mag : torch.Tensor
        STFT magnitude spectrogram, shape ``(n_freqs, n_frames)`` or
        ``(batch, n_freqs, n_frames)`` or ``(batch, n_channels, n_freqs, n_frames)``,
        where ``n_freqs = n_fft // 2 + 1``.

    Examples
    --------
    >>> ir = torch.randn(1, 48000)  # batch of 1, 48 kHz @ 1 s
    >>> mag = stft_magnitude(ir, n_fft=2048)
    >>> mag.shape
    torch.Size([1, 1025, 95])  # (batch, freq, time)
    """
    if hop_length is None:
        hop_length = n_fft // 4

    original_shape = ir.shape
    if ir.ndim == 1:
        ir = ir.unsqueeze(0)  
    else:
        raise ValueError(f"Expected 1D input, got shape {original_shape}")

    # Compute STFT
    spec = torch.stft(
        ir,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=ir.device, dtype=ir.dtype)
        if window == "hann"
        else torch.hamming_window(n_fft, device=ir.device, dtype=ir.dtype)
        if window == "hamming"
        else torch.blackman_window(n_fft, device=ir.device, dtype=ir.dtype),
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    # Magnitude (batch, n_freqs, n_frames)
    mag = torch.abs(spec)


    mag = mag.squeeze(0)  # (1, n_freqs, n_frames) -> (n_freqs, n_frames)

    return mag


def stft_phase(
    ir: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int | None = None,
    window: str = "hann",
) -> torch.Tensor:
    r"""Compute STFT phase spectrogram from time-domain IR.

    Parameters
    ----------
    ir : torch.Tensor
        Time-domain impulse response, shape ``(n_samples,)`` or
        ``(batch, n_samples)`` or ``(batch, n_samples, n_channels)``.
    n_fft : int, default 2048
        FFT size.
    hop_length : int, optional
        Hop length; defaults to ``n_fft // 4``.
    window : str, default "hann"
        Window function name.

    Returns
    -------
    phase : torch.Tensor
        STFT phase in radians, shape same as :func:`stft_magnitude`.

    See Also
    --------
    stft_magnitude : Corresponding magnitude representation.
    """
    if hop_length is None:
        hop_length = n_fft // 4

    original_shape = ir.shape
    if ir.ndim == 1:
        ir = ir.unsqueeze(0)
    else:
        raise ValueError(f"Expected 1D input, got shape {original_shape}")

    spec = torch.stft(
        ir,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=ir.device, dtype=ir.dtype)
        if window == "hann"
        else torch.hamming_window(n_fft, device=ir.device, dtype=ir.dtype)
        if window == "hamming"
        else torch.blackman_window(n_fft, device=ir.device, dtype=ir.dtype),
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    phase = torch.angle(spec)

    phase = phase.squeeze(0)

    return phase


