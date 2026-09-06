"""Temporal feature representations and decay metrics."""

from __future__ import annotations

import torch


def energy_decay_curve(
    ir: torch.Tensor,
    dim: int = 0,
    eps: float = 1e-12,
    db: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    r"""Compute the Energy Decay Curve (EDC) via backward Schroeder integration.

        Evaluates the continuous-energy decay profile across discrete time samples:
    .. math::

        \text{EDC}[n] = \sum_{m=n}^{L-1} h^2[m]

    Parameters
    ----------
    ir : torch.Tensor
        Discrete-time impulse response. The time axis must be the first
        dimension (``dim=0``), such as ``(n_samples,)``, ``(n_samples, channels)``,
        or ``(n_samples, n_out, n_in)``.
    dim : int, default 0
        Temporal dimension.
    eps : float, default 1e-12
        Numerical stability constant.
    db : bool, default True
        If ``True``, return the EDC in decibels.
    normalize : bool, default True
        If ``True``, normalize the EDC by its initial energy.

    Returns
    -------
    torch.Tensor
        Energy decay curve with the same shape as ``ir``.
    """
    if not isinstance(ir, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(ir).__name__}")

    if not ir.is_floating_point():
        raise TypeError(f"Expected real floating-point tensor, got {ir.dtype}")

    energy = ir.pow(2)

    # Schroeder backward integration
    edc = torch.flip(
        torch.cumsum(
            torch.flip(energy, dims=[dim]),
            dim=dim,
        ),
        dims=[dim],
    )

    if normalize:
        initial_energy = edc.select(dim, 0).unsqueeze(dim)
        edc = edc / initial_energy.clamp_min(eps)

    if db:
        edc = 10.0 * torch.log10(edc.clamp_min(eps))

    return edc
