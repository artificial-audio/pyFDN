"""Asymmetric magnitude-response loss .

This loss penalizes only the parts of the magnitude response that lie
*above* its mean, leaving deviations below the mean unpenalized.

.. math::
    L = \\| |H(e^{j\\omega})| - \\mathrm{mean}(|H(e^{j\\omega})|) \\|

where :math:`\\|x\\| = x^2` for :math:`x > 0` and :math:`0` otherwise.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def asymmetric_loss(
    h: torch.Tensor,
    eps: float = 1e-12,
    db: bool = True,
    dim: int = -1,
) -> torch.Tensor:
    r"""Compute the asymmetric loss from a frequency response.

    Only penalizes magnitude excursions above the mean magnitude

    Parameters
    ----------
    h : torch.Tensor
        Complex or real-valued frequency response
        :math:`H(e^{j\\omega})`, shape ``(..., n_freqs)``. If complex,
        its magnitude is used.
    eps : float, default 1e-12
        Small value used for numerical stability when computing dB.
    db : bool, default True
        If ``True``, operate on the magnitude in dB
        (``20 * log10(|H|)``) before computing the mean and deviation.
    dim : int, default -1
        Frequency dimension over which the mean is computed.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss value (mean over all remaining dimensions).

    Examples
    --------
    >>> h = torch.randn(1025, dtype=torch.cfloat)
    >>> loss = asymmetric_loss(h)
    >>> loss.shape
    torch.Size([])
    """
    if torch.is_complex(h):
        mag = torch.abs(h)
    else:
        mag = h

    if db:
        mag = 20.0 * torch.log10(torch.clamp(mag, min=eps))

    mean_mag = torch.mean(mag, dim=dim, keepdim=True)
    deviation = mag - mean_mag

    # Only penalize positive deviations (above-mean energy / peaks).
    positive_deviation = torch.clamp(deviation, min=0.0)
    loss = positive_deviation.pow(2)

    return loss.mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric magnitude-response loss module.

    Wraps :func:`asymmetric_loss` as an ``nn.Module`` for use
    in training pipelines.

    Parameters
    ----------
    eps : float, default 1e-12
        Small value used for numerical stability when computing dB.
    db : bool, default True
        If ``True``, operate on the magnitude in dB.
    dim : int, default -1
        Frequency dimension over which the mean is computed.
    """

    def __init__(self, eps: float = 1e-12, db: bool = True, dim: int = -1) -> None:
        super().__init__()
        self.eps = eps
        self.db = db
        self.dim = dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return asymmetric_loss(h, eps=self.eps, db=self.db, dim=self.dim)
