"""Spatial features impulse response."""

from __future__ import annotations

import torch

def mimo_rir_eigenvalues_per_frequency(
    ir: torch.Tensor,
    n_fft: int = 2048,
    dim: int = 0,
) -> torch.Tensor:
    r"""Compute eigenvalues of a MIMO transfer matrix at each frequency bin.

    The impulse response is transformed to the frequency domain using
    :func:`torch.fft.rfft`, and the eigenvalues of the resulting transfer
    matrix are computed independently at each frequency bin.

    Parameters
    ----------
    ir : torch.Tensor
        MIMO impulse response with shape ``(n_samples, n_out, n_in)``.
        The system must be square, i.e. ``n_out == n_in``.
    n_fft : int, default 2048
        DFT length.
    dim : int, default 0
        Temporal dimension of the impulse response.

    Returns
    -------
    torch.Tensor
        Complex eigenvalues with shape
        ``(n_freqs, n_out)``, where ``n_freqs = n_fft // 2 + 1``.

    Raises
    ------
    ValueError
        If ``ir`` is not 3D or the MIMO system is not square.
    """
    if not isinstance(ir, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(ir).__name__}")
    
    if ir.ndim != 3:
        raise ValueError(
            f"Expected a 3D tensor of shape (n_samples, n_out, n_in), "
            f"got shape {tuple(ir.shape)}."
        )

    n_out, n_in = ir.shape[1:]
    if n_out != n_in:
        raise ValueError(
            f"Expected a square MIMO system, got {n_out}x{n_in}."
        )

    h_freq = torch.fft.rfft(ir, n=n_fft, dim=dim)

    return torch.linalg.eigvals(h_freq)