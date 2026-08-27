
import torch
import torch.nn.functional as F


def spectral_flatness(
    ir: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int | None = None,
    window: str = "hann",
    eps: float = 1e-10,
) -> torch.Tensor:
    r"""Compute spectral flatness from time-domain IR.

    Spectral flatness measures the "flatness" of the magnitude spectrum,
    computed as the ratio of geometric mean to arithmetic mean of the
    magnitude spectrum. Higher values indicate a flatter spectrum.

    .. math::

        \text{Flatness} = \frac{\text{GM}(|S|)}{\text{AM}(|S|)}

    where :math:`\text{GM}` is the geometric mean and :math:`\text{AM}` is
    the arithmetic mean of the magnitude spectrum :math:`|S|`.

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
    eps : float, default 1e-10
        Small epsilon value to avoid log(0) and numerical issues.

    Returns
    -------
    flatness : torch.Tensor
        Spectral flatness per frame, shape ``(n_frames,)``, where
        ``n_frames`` depends on IR length and hop length.

    Examples
    --------
    >>> ir = torch.randn(48000)  # 48 kHz @ 1 s
    >>> flatness = spectral_flatness(ir, n_fft=2048)
    >>> flatness.shape
    torch.Size([95])  # (time,)
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

    # Magnitude (batch=1, n_freqs, n_frames)
    mag = torch.abs(spec)

    # Compute geometric mean: exp(mean(log(x)))
    # Add eps to avoid log(0)
    log_mag = torch.log(mag + eps)
    geometric_mean = torch.exp(torch.mean(log_mag, dim=1))  # (1, n_frames)

    # Compute arithmetic mean
    arithmetic_mean = torch.mean(mag, dim=1)  # (1, n_frames)

    # Spectral flatness
    flatness = geometric_mean / (arithmetic_mean + eps)

    flatness = flatness.squeeze(0)  # (1, n_frames) -> (n_frames,)

    return flatness

