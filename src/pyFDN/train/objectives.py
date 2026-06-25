"""Objective setup for :func:`pyFDN.train_fdn`: map a training *mode* (plus
optional target data) to the ``(input, target)`` tensor pair, loss criteria, and
the model output domain to measure in.

The mode alone fixes the loss and the output domain. The matching modes take
their ``target`` as a single time-domain impulse response and compare it in the
time domain, so the user never has to pre-transform the reference.

==========================  ====================================  ============
mode                        loss                                  target
==========================  ====================================  ============
``"colorless"``             magnitude MSE vs a flat response      none
``"match_spectrogram"``     multi-resolution STFT (``mss``)       impulse resp.
``"match_mel_spectrogram"`` mel multi-resolution STFT             impulse resp.
==========================  ====================================  ============

``colorless`` uses flamo's ``mse_loss``, which sums the model's outputs before
the MSE, so the target is a single flat magnitude curve. It is therefore a
**SISO** objective: colorlessness is a property of the FDN core (feedback matrix
+ delays), not of the input/output gains, so multiple ``B``/``C`` columns add no
flatness the core does not already provide and the objective cannot distinguish
them (what a multichannel reverb actually wants from extra channels is
*decorrelation*, a different objective). Build with a single input and output
for ``colorless``. (Matching an arbitrary target *magnitude* bin-for-bin is
ill-posed for a reverb's spiky modal spectrum and is deliberately not offered;
match a real IR with the spectrogram modes instead.)

Every mode also adds a feedback-matrix sparsity penalty
(:func:`colorless_sparsity_loss`, weight ``sparsity_alpha``, default 0.2; 0
disables it) that biases the mixing matrix toward dense, colorless mixing -- it
only bites when the feedback matrix is trainable.

A single fixed ``(input, target)`` pair fully specifies the fit -- one LTI system
is a pure optimization problem, with no held-out data -- so this builds raw
tensors for :class:`pyFDN.train._trainer.EagerTrainer` rather than a flamo
``Dataset``.

flamo/torch are imported lazily inside :func:`build_objective`,
:func:`_input_target` and :func:`colorless_sparsity_loss`, so this module
imports without torch.
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
    """A flamo-compatible sparsity penalty on the FDN feedback matrix.

    Same role and formula as flamo's ``sparsity_loss`` (the colorless penalty of
    *Optimizing Tiny Colorless FDNs*, Dal Santo et al.), but it locates the
    feedback matrix through pyFDN's graph walk
    (:func:`pyFDN.auxiliary.flamo_graph.feedback_matrix_module`), which resolves
    it for both a plain ``Series`` core and a ``Parallel`` core (the FDN summed
    with an always-present direct path). flamo's own ``sparsity_loss`` hard-codes
    attribute paths that assume a ``.mixing_matrix`` on the Parallel branch, so
    it raises once the direct path is always present.

    ``A = map(param)`` is computed **in-graph** so gradients flow. The returned
    ``nn.Module`` has the ``(y_pred, y_target, model)`` signature flamo's
    ``Trainer`` uses for ``requires_model=True`` criteria.
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

    Maps the mode to the ``(input, target)`` tensor pair (with ``target`` shaped
    appropriately) and the full criteria list -- the mode's primary loss plus the
    sparsity regularizer -- unless a caller-supplied ``criteria`` override
    replaces it. flamo/torch are imported here so the module stays torch-free.
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
    """The fixed ``(input, target)`` tensors a mode fits, each batch dim 1."""
    import torch

    # A unit impulse drives the model -- identical for every mode.
    impulse = torch.zeros((1, nfft, n_in), device=device, dtype=dtype)
    impulse[:, 0, :] = 1.0

    if flat:
        tgt = torch.ones((1, nfft // 2 + 1, 1), device=device, dtype=dtype)
        return impulse, tgt

    arr = np.asarray(target, dtype=np.float64)
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
