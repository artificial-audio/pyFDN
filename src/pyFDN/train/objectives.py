"""Map a training *mode* (and optional target) to the ``(input, target)``
tensors, loss criteria, and output domain that :func:`pyFDN.train_fdn` uses.

==========================  ====================================  ============
mode                        loss                                  target
==========================  ====================================  ============
``"colorless"``             magnitude MSE vs a flat response      none
``"match_spectrogram"``     multi-resolution STFT (``mss``)       impulse resp.
``"match_mel_spectrogram"`` mel multi-resolution STFT             impulse resp.
==========================  ====================================  ============

``colorless`` is single-input/single-output (its loss sums the model's outputs).
The matching modes take a time-domain IR target; a 3-D ``(n_samples, n_out,
n_in)`` target fits a full MIMO system. Every mode also adds a feedback-matrix
sparsity penalty (weight ``sparsity_alpha``) that only acts when the feedback
matrix is trainable.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np

Objective = Literal[
    "colorless",
    "match_spectrogram",
    "match_mel_spectrogram",
]

# (criterion, alpha, requires_model) tuples for the trainer's register_criterion.
Criterion = tuple[Any, float, bool]

# mode -> output domain the model is measured in ("time" or "magnitude").
# "colorless" fits a flat magnitude target; the matching modes fit a time-domain
# impulse-response target directly.
_MODES: dict[str, str] = {
    "colorless": "magnitude",
    "match_spectrogram": "time",
    "match_mel_spectrogram": "time",
}

# "colorless" fits a synthetic flat target and needs no user data.
_NEEDS_TARGET = frozenset(_MODES) - {"colorless"}


def output_domain(mode: str) -> str:
    """The model output layer a mode measures in (``"time"`` or ``"magnitude"``)."""
    if mode not in _MODES:
        raise ValueError(
            f"unknown training mode {mode!r}; choose from {sorted(_MODES)}"
        )
    return _MODES[mode]


def colorless_sparsity_loss() -> Any:
    """Sparsity penalty on the FDN feedback matrix (the colorless penalty of
    *Optimizing Tiny Colorless FDNs*, Dal Santo et al.).

    Returns an ``nn.Module`` with the ``(y_pred, y_target, model)`` signature used
    for ``requires_model=True`` criteria.
    """
    import torch.nn as nn

    from pyFDN.auxiliary.flamo_graph import feedback_matrix_module

    class _ColorlessSparsity(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._module: Any = None

        def forward(self, y_pred: Any, y_target: Any, model: Any) -> Any:
            if self._module is None:
                self._module = feedback_matrix_module(model)
            module = self._module
            a = module.map(module.param)
            n = a.shape[-1]
            root_n = float(np.sqrt(n))
            # 0 when |A| is maximally dense (good mixing), 1 when fully sparse.
            return -(a.abs().sum() - n * root_n) / (n * (root_n - 1.0))

    return _ColorlessSparsity()


def build_objective(
    mode: str,
    *,
    target: Any,
    criteria: list[Criterion] | None,
    sparsity_alpha: float,
    mss_nfft: tuple[int, ...],
    fs: float,
    nfft: int,
    n_in: int,
    n_out: int,
    device: Any,
    dtype: Any,
) -> tuple[Any, Any, list[Criterion], str]:
    """Return ``(input, target, criteria, output_domain)`` for a training ``mode``.

    Uses the mode's default criteria (primary loss + sparsity) unless ``criteria``
    overrides them.
    """
    if mode not in _MODES:
        raise ValueError(
            f"unknown training mode {mode!r}; choose from {sorted(_MODES)}"
        )
    domain = _MODES[mode]
    needs_target = mode in _NEEDS_TARGET
    if needs_target and target is None:
        raise ValueError(f"mode {mode!r} requires target=")
    if domain == "magnitude" and n_out != 1:
        warnings.warn(
            f"{mode!r} is a SISO magnitude objective but the model has "
            f"n_out={n_out} outputs; their magnitudes are summed into one and "
            "fit together. Build with a single output for a well-posed fit.",
            stacklevel=3,
        )
    inp, tgt = _input_target(
        target, nfft, n_in, n_out, device, dtype, flat=not needs_target
    )
    if criteria is not None:
        return inp, tgt, criteria, domain

    crit: list[Criterion] = [
        (_primary_loss(mode, nfft, mss_nfft, fs, device), 1.0, False)
    ]
    if sparsity_alpha > 0:
        crit.append((colorless_sparsity_loss(), float(sparsity_alpha), True))
    return inp, tgt, crit, domain


def _input_target(
    target: Any,
    nfft: int,
    n_in: int,
    n_out: int,
    device: Any,
    dtype: Any,
    *,
    flat: bool,
) -> tuple[Any, Any]:
    """The fixed ``(input, target)`` tensors a mode fits.

    Single-impulse excitation for SISO/SIMO; a 3-D ``(n_samples, n_out, n_in)``
    target excites each input on its own batch row to fit the full MIMO matrix.
    """
    import torch

    if flat:
        # colorless: one impulse on all inputs, flat single-channel target.
        impulse = torch.zeros((1, nfft, n_in), device=device, dtype=dtype)
        impulse[:, 0, :] = 1.0
        tgt = torch.ones((1, nfft // 2 + 1, 1), device=device, dtype=dtype)
        return impulse, tgt

    arr = np.asarray(target, dtype=np.float64)

    if arr.ndim == 3:
        # MIMO: target is the full IR matrix (n_samples, n_out, n_in). Excite each
        # input on its own batch row (identity at t=0) so the model's batched
        # output (n_in, nfft, n_out) is the transfer matrix H[out, in].
        if arr.shape[1:] != (n_out, n_in):
            raise ValueError(
                f"MIMO target must have shape (n_samples, n_out={n_out}, "
                f"n_in={n_in}); got {arr.shape}"
            )
        impulse = torch.zeros((n_in, nfft, n_in), device=device, dtype=dtype)
        for i in range(n_in):
            impulse[i, 0, i] = 1.0
        ir = np.zeros((n_in, nfft, n_out))
        length = min(arr.shape[0], nfft)
        # arr[t, j, i] -> tgt[i, t, j]: input on the batch axis, output as channel.
        ir[:, :length, :] = np.transpose(arr[:length], (2, 0, 1))
        return impulse, torch.as_tensor(ir, device=device, dtype=dtype)

    # SISO / SIMO: one impulse on all inputs (batch 1), target one column per
    # output; a mono target is broadcast across the outputs.
    impulse = torch.zeros((1, nfft, n_in), device=device, dtype=dtype)
    impulse[:, 0, :] = 1.0
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] != n_out:
        arr = np.repeat(arr[:, :1], n_out, axis=1)
    ir = np.zeros((nfft, n_out))
    length = min(arr.shape[0], nfft)
    ir[:length, :] = arr[:length, :]

    return impulse, torch.as_tensor(ir[None], device=device, dtype=dtype)


def _primary_loss(
    mode: str, nfft: int, mss_nfft: tuple[int, ...], fs: float, device: Any
) -> Any:
    if mode == "colorless":
        from flamo.optimize.loss import mse_loss

        return mse_loss(nfft=nfft, device=device)
    elif mode == "match_spectrogram":
        from flamo.optimize.loss import mss_loss

        return mss_loss(nfft=list(mss_nfft), sample_rate=int(fs), device=device)
    elif mode == "match_mel_spectrogram":
        from flamo.optimize.loss import mel_mss_loss

        return mel_mss_loss(nfft=list(mss_nfft), sample_rate=int(fs), device=device)
