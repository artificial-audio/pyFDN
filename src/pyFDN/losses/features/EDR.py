"""Differentiable RIR energy decay relief feature extractor.

Extract time-domain impulse responses (IRs) into an energy decay relief (EDR)
representation for use in training objectives and loss functions. The feature
is implemented with differentiable PyTorch operations suitable for
backpropagation-based training.
"""

from __future__ import annotations

import torch


def energy_decay_relief(
    ir: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int | None = None,
    window: str = "hann",
    eps: float = 1e-12,
    db: bool = True,
) -> torch.Tensor:
    r"""Compute energy decay relief (EDR) from time-domain IR.

    Computes a time-frequency energy representation using the STFT and applies
    backward Schroeder integration over time for each frequency bin.

    Parameters
    ----------
    ir : torch.Tensor
        Time-domain impulse response, shape ``(n_samples,)``.
    n_fft : int, default 2048
        FFT size.
    hop_length : int, optional
        Hop length; defaults to ``n_fft // 4``.
    window : str, default "hann"
        Window function name ('hann', 'hamming', 'blackman', etc.).
    eps : float, default 1e-12
        Small value used for numerical stability.
    db : bool, default True
        If ``True``, return EDR in dB normalized to 0 dB at the maximum energy.

    Returns
    -------
    edr : torch.Tensor
        Energy decay relief, shape ``(n_freqs, n_frames)``, where
        ``n_freqs = n_fft // 2 + 1``.

    Examples
    --------
    >>> ir = torch.randn(48000)
    >>> edr = energy_decay_relief(ir, n_fft=2048)
    >>> edr.shape
    torch.Size([1025, 95])
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

    energy = torch.abs(spec).pow(2)
    edr = torch.flip(torch.cumsum(torch.flip(energy, dims=(-1,)), dim=-1), dims=(-1,))

    if db:
        edr = 10.0 * torch.log10(torch.clamp(edr, min=eps))
        edr = edr - torch.amax(edr, dim=(-2, -1), keepdim=True)

    edr = edr.squeeze(0)

    return edr
