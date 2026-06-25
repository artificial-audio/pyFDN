"""Train an FDN toward a target.

:func:`train_fdn` fits a model from :func:`pyFDN.build_fdn` (or
:func:`pyFDN.trainable_from_build`) toward a training *mode* (``colorless`` /
``match_spectrogram`` / ``match_mel_spectrogram``),
**in place**, and returns a :class:`TrainLog`. Fitting one FDN is a pure
optimization on a single ``(input, target)`` pair, so it runs
:class:`pyFDN.train._trainer.EagerTrainer` (a direct gradient loop) rather than a
``Dataset``/``DataLoader`` epoch stack. Extraction
(:func:`pyFDN.extract_build`) and scoring (the metrics) are separate, explicit
steps so the user keeps full control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .objectives import Criterion, Objective, build_objective


@dataclass
class TrainLog:
    """Optimization metadata from a training run.

    Attributes
    ----------
    train_loss : list of float
        Total loss at each optimization step.
    loss_log : dict of str to list of float
        Per-criterion loss history, keyed by criterion class name.
    steps_run : int
        Gradient steps actually run (< ``max_steps`` if a plateau stopped it).
    stopped_early : bool
        Whether optimization stopped before ``max_steps``.
    """

    train_loss: list[float] = field(default_factory=list)
    loss_log: dict[str, list[float]] = field(default_factory=dict)
    steps_run: int = 0
    stopped_early: bool = False


def train_fdn(
    model: Any,
    mode: Objective,
    *,
    target: Any = None,
    criteria: list[Criterion] | None = None,
    sparsity_alpha: float = 0.2,
    mss_nfft: tuple[int, ...] = (256, 512, 1024),
    max_steps: int = 2000,
    lr: float = 1e-3,
    optimizer: str = "adam",
    patience: int = 10,
    tol: float = 1e-6,
    device: Any = None,
    dtype: Any = None,
    rng: int | None = None,
    log: bool = False,
    train_dir: str | None = None,
) -> TrainLog:
    """Train ``model`` for ``mode`` (in place) and return a TrainLog.

    The ``mode`` fixes the loss and the model output domain (see
    :mod:`pyFDN.train.objectives`); this builds the ``(input, target)`` pair, sets
    the output layer to match, and runs :class:`~pyFDN.train._trainer.EagerTrainer`
    directly on it. The model is mutated in place; read the result back with
    :func:`pyFDN.extract_build`.

    Parameters
    ----------
    model : flamo Shell
        A trainable model from :func:`pyFDN.build_fdn` / ``trainable_from_build``.
        ``dtype`` should match the model's (default float32 on both).
    mode : str
        ``"colorless"``, ``"match_spectrogram"`` or ``"match_mel_spectrogram"``.
        ``"colorless"`` is a SISO objective (build the model with one input and
        one output): colorlessness lives in the FDN core, so extra ``B``/``C``
        channels add nothing the magnitude loss can use.
    target : np.ndarray, optional
        Reference impulse response the matching modes fit to -- a time-domain IR.
        Unused for ``colorless``; required otherwise.
    criteria : list of (criterion, alpha, requires_model), optional
        Override the default loss list (primary loss + sparsity) with your own
        flamo criteria.
    sparsity_alpha : float
        Weight of the feedback-matrix sparsity penalty added to every mode
        (default 0.2; 0 disables it; only works when the feedback is trainable).
    mss_nfft : tuple of int
        Multi-resolution STFT window sizes for the spectrogram modes.
    max_steps, lr, patience : optimization settings (max gradient steps, learning
        rate, plateau patience in steps).
    optimizer : str
        ``"adam"`` (default) or ``"lbfgs"``; L-BFGS suits the deterministic match
        modes, Adam suits all of them.
    tol : float
        Relative-improvement threshold for the plateau early stop (default 1e-6).
    device, dtype : optional
        Torch device / dtype (default cpu / float32).
    rng : int or None
        Integer seed for ``torch.manual_seed``; ``None`` leaves torch's global
        RNG untouched. Training only seeds torch (its RNG is independent of
        NumPy's), so this is a plain int rather than a NumPy generator.
    log : bool
        If True, flamo logs/checkpoints to ``train_dir`` (created if missing).
        Default False keeps training side-effect-free.
    train_dir : str, optional
        Checkpoint directory (only used when ``log=True``); created if missing.

    Notes
    -----
    Setting the output layer is a deliberate, visible mutation: after a
    ``colorless`` run the model emits ``|H|``; after the spectrogram modes it
    emits a time response.
    """
    import torch

    from ._trainer import EagerTrainer

    dev = "cpu" if device is None else device
    torch_dtype = torch.float32 if dtype is None else dtype

    if rng is not None:
        torch.manual_seed(int(rng))

    nfft = int(model.get_inputLayer().nfft)
    n_in, n_out, fs = _model_info(model)

    inp, tgt, criteria, output_domain = build_objective(
        mode,
        target=target,
        criteria=criteria,
        sparsity_alpha=sparsity_alpha,
        mss_nfft=tuple(mss_nfft),
        fs=fs,
        nfft=nfft,
        n_in=n_in,
        n_out=n_out,
        device=dev,
        dtype=torch_dtype,
    )
    _set_output_domain(model, output_domain, nfft, torch_dtype)

    # Checkpoint only when logging to a directory; EagerTrainer asserts it exists.
    save_checkpoints = log and train_dir is not None
    if save_checkpoints:
        os.makedirs(train_dir, exist_ok=True)
    trainer = EagerTrainer(
        model,
        max_steps=max_steps,
        lr=lr,
        optimizer=optimizer,
        patience=patience,
        tol=tol,
        device=dev,
        log=log,
        train_dir=train_dir,
        save_checkpoints=save_checkpoints,
    )
    for criterion, alpha, requires_model in criteria:
        trainer.register_criterion(criterion, alpha, requires_model)
    history = trainer.optimize(inp, tgt)

    train_loss = [float(x) for x in history.get("total", [])]
    steps_run = len(train_loss)
    return TrainLog(
        train_loss=train_loss,
        loss_log={
            k: [float(x) for x in v] for k, v in history.items() if k != "total"
        },
        steps_run=steps_run,
        stopped_early=steps_run < max_steps,
    )


def _model_info(model: Any) -> tuple[int, int, float]:
    """``(n_in, n_out, fs)`` read from the model's gain and delay leaves."""
    from pyFDN.auxiliary.flamo_graph import flamo_model_to_nodes, flamo_nodes_flat

    leaves = [
        n for n in flamo_nodes_flat(flamo_model_to_nodes(model)) if n["type"] == "Leaf"
    ]

    def _gain(name: str) -> Any:
        matches = [n["module"] for n in leaves if n["name"] == name]
        if len(matches) != 1:
            raise ValueError(
                f"model must contain exactly one {name!r} leaf; found {len(matches)}"
            )
        return matches[0]

    delay = next(
        (n["module"] for n in leaves if "delay" in type(n["module"]).__name__.lower()),
        None,
    )
    fs = float(getattr(delay, "fs", 48000.0) or 48000.0)
    return (
        int(_gain("input_gain").param.shape[1]),
        int(_gain("output_gain").param.shape[0]),
        fs,
    )


def _set_output_domain(model: Any, output_domain: str, nfft: int, dtype: Any) -> None:
    """Reset the Shell's output layer to time (iFFT) or magnitude (``|.|``)."""
    import torch
    from flamo.processor import dsp

    if output_domain == "time":
        layer: Any = dsp.iFFT(nfft, dtype=dtype)
    elif output_domain == "magnitude":
        layer = dsp.Transform(transform=torch.abs, dtype=dtype)
    else:
        raise ValueError(
            f"output_domain must be 'time' or 'magnitude', got {output_domain!r}"
        )
    model.set_outputLayer(layer)
