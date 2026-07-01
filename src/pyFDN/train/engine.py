"""Train an FDN toward a target.

:func:`train_fdn` fits a model from :func:`pyFDN.build_fdn` toward a *mode*
(``colorless`` / ``match_spectrogram`` / ``match_mel_spectrogram``) in place and
returns a :class:`TrainLog`. Read the result back with :func:`pyFDN.extract_build`.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any

from .objectives import Criterion, Objective, build_objective


@dataclass
class TrainLog:
    """Per-step loss history and stopping info from a training run.

    Attributes
    ----------
    train_loss : list of float
        Total loss at each step.
    loss_log : dict of str to list of float
        Per-criterion loss history, keyed by criterion class name.
    steps_run : int
        Steps actually run.
    stopped_early : bool
        Whether a plateau stopped it before ``max_steps``.
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
    """Train ``model`` for ``mode`` in place and return a :class:`TrainLog`.

    Read the trained result back with :func:`pyFDN.extract_build`.

    Parameters
    ----------
    model : flamo Shell
        A trainable model from :func:`pyFDN.build_fdn` / ``trainable_from_build``.
    mode : str
        ``"colorless"``, ``"match_spectrogram"`` or ``"match_mel_spectrogram"``.
        ``"colorless"`` is single-input/single-output only.
    target : np.ndarray, optional
        Reference impulse response for the matching modes (unused for
        ``colorless``). Shape ``(n_samples,)`` or ``(n_samples, n_out)``, or a 3-D
        ``(n_samples, n_out, n_in)`` IR matrix to fit a full MIMO system.
    criteria : list of (criterion, alpha, requires_model), optional
        Replace the default loss list (primary loss + sparsity) with your own.
    sparsity_alpha : float
        Weight of the feedback-matrix sparsity penalty (default 0.2; 0 disables).
    mss_nfft : tuple of int
        STFT window sizes for the spectrogram modes.
    max_steps, lr, patience : max gradient steps, learning rate, plateau patience.
    optimizer : str
        ``"adam"`` (default) or ``"lbfgs"``.
    tol : float
        Relative-improvement threshold for the plateau early stop.
    device, dtype : optional
        Torch device / dtype (default cpu / float32).
    rng : int or None
        Integer seed for ``torch.manual_seed``.
    log : bool
        If True, log/checkpoint to ``train_dir``.
    train_dir : str, optional
        Checkpoint directory (used when ``log=True``).
    """
    import torch
    from flamo.optimize.trainer import EagerTrainer

    from pyFDN.auxiliary.flamo import output_layer

    dev = "cpu" if device is None else device
    torch_dtype = torch.float32 if dtype is None else dtype

    if rng is not None:
        torch.manual_seed(int(rng))

    nfft = int(model.get_inputLayer().nfft)
    n_in, n_out, fs = _model_info(model)

    if mode == "colorless" and _loop_filter_is_trainable(model):
        warnings.warn(
            "colorless does not constrain decay, so a trainable in-loop filter "
            "is unconstrained and may drift toward a lossless/unstable loop; use "
            "match_spectrogram / match_mel_spectrogram to fit a target's decay.",
            stacklevel=2,
        )

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
    # Set the Shell's output layer to match the objective (mutates the model).
    model.set_outputLayer(output_layer(output_domain, nfft, torch_dtype))

    save_checkpoints = log and train_dir is not None
    if save_checkpoints:
        assert train_dir is not None  # implied by save_checkpoints
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
        loss_log={k: [float(x) for x in v] for k, v in history.items() if k != "total"},
        steps_run=steps_run,
        stopped_early=steps_run < max_steps,
    )


def _loop_filter_is_trainable(model: Any) -> bool:
    """True if the model's in-loop filter leaf has any ``requires_grad`` param."""
    from pyFDN.auxiliary import flamo_graph as layout
    from pyFDN.auxiliary.flamo_graph import flamo_model_to_nodes, flamo_nodes_flat

    leaves = [
        n for n in flamo_nodes_flat(flamo_model_to_nodes(model)) if n["type"] == "Leaf"
    ]
    modules = [
        n["module"] for n in leaves if n["name"] in layout.LOOP_FILTER_ALIASES
    ]
    return any(
        getattr(p, "requires_grad", False)
        for m in modules
        if hasattr(m, "parameters")
        for p in m.parameters()
    )


def _model_info(model: Any) -> tuple[int, int, float]:
    """``(n_in, n_out, fs)`` read from the model's gain and delay modules."""
    from pyFDN.auxiliary import flamo_graph as layout
    from pyFDN.auxiliary.flamo_graph import flamo_model_to_nodes, flamo_nodes_flat

    leaves = [
        n for n in flamo_nodes_flat(flamo_model_to_nodes(model)) if n["type"] == "Leaf"
    ]

    def _gain(name: str) -> Any:
        matches = [n["module"] for n in leaves if n["name"] == name]
        if len(matches) != 1:
            raise ValueError(
                f"model must contain exactly one {name!r} module; found {len(matches)}"
            )
        return matches[0]

    delay = next((n["module"] for n in leaves if n["name"] == layout.DELAY), None)
    if delay is None:
        raise ValueError("model has no delay module; check build again.")
    fs = float(delay.fs)
    return (
        int(_gain(layout.INPUT_GAIN).param.shape[1]),
        int(_gain(layout.OUTPUT_GAIN).param.shape[0]),
        fs,
    )
