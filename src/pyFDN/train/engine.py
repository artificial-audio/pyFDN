"""Train an FDN toward an objective.

:func:`train_fdn` fits a model from :func:`pyFDN.build_fdn` to a loss built from
:mod:`pyFDN.train.losses`, in place, and returns a :class:`TrainLog`. Read the
result back with :func:`pyFDN.extract_build`.

The engine knows nothing about any particular objective. Its whole job is: run
the model on an impulse, hand the resulting :class:`~pyFDN.train.response.Response`
to each loss term, and let the optimizer do the rest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .losses import Loss
from .response import Response, impulse_excitation, model_fs, require_time_output


@dataclass
class TrainLog:
    """Per-step loss history and stopping info from a training run.

    Attributes
    ----------
    train_loss : list of float
        Total (weighted) loss at each step.
    loss_log : dict of str to list of float
        Per-term loss history, keyed by each term's name and stored
        *unweighted*, so terms stay comparable to their own scale.
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
    loss: Loss,
    *,
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
    """Train ``model`` on ``loss`` in place and return a :class:`TrainLog`.

    Read the trained result back with :func:`pyFDN.extract_build`.

    Parameters
    ----------
    model : flamo Shell
        A trainable model from :func:`pyFDN.build_fdn` / ``trainable_from_build``.
        It must return its impulse response -- every loss is a function of it --
        which every pyFDN shell does by construction.
    loss : Loss
        The objective, e.g.::

            pyFDN.FlatMagnitude() + 0.2 * pyFDN.Sparsity(pyFDN.param(model, "feedback"))

        A loss holds whatever reference data it needs (e.g.
        ``MatchSpectrogram(target)``), so one objective can compare against
        more than one reference.
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

    dev = "cpu" if device is None else device
    torch_dtype = torch.float32 if dtype is None else dtype

    if rng is not None:
        torch.manual_seed(int(rng))

    loss.check(model)

    nfft = int(model.nfft)
    n_in = int(model.input_channels)
    # Every loss reads the impulse response. The trainer does not install that
    # output layer, it requires one: the model's domain is settled where the
    # model is built.
    require_time_output(model)
    excitation = impulse_excitation(n_in, nfft, device=dev, dtype=torch_dtype)

    # Checkpoint only when logging to a directory; EagerTrainer asserts it exists.
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

    factory = _ResponseFactory(model_fs(model))
    for weight, term, name in _named_terms(loss):
        trainer.register_criterion(_criterion(term, name, factory), weight, False)

    history = trainer.optimize(excitation, _unused_target(torch, dev, torch_dtype))

    train_loss = [float(x) for x in history.get("total", [])]
    steps_run = len(train_loss)
    return TrainLog(
        train_loss=train_loss,
        loss_log={k: [float(x) for x in v] for k, v in history.items() if k != "total"},
        steps_run=steps_run,
        stopped_early=steps_run < max_steps,
    )


def _unused_target(torch: Any, device: Any, dtype: Any) -> Any:
    """Placeholder for FLAMO's fixed target tensor.

    ``EagerTrainer`` fits an ``(input, target)`` pair, but here every loss holds
    whatever reference data it needs -- which is what lets one objective compare
    against two different references. Nothing reads this tensor.
    """
    return torch.zeros(0, device=device, dtype=dtype)


class _ResponseFactory:
    """Turns a model output into a :class:`Response`, once per step.

    All terms in one step see the same model output tensor, so a single-entry
    cache keyed on its identity means the ``Response``'s lazily computed
    spectrum is shared by every spectral term instead of recomputed per term.
    """

    def __init__(self, fs: float) -> None:
        self.fs = fs
        self._cached_output: Any = None
        self._cached_response: Response | None = None

    def of(self, output: Any) -> Response:
        if output is self._cached_output and self._cached_response is not None:
            return self._cached_response
        # model output is (n_in, nfft, n_out); Response is (nfft, n_out, n_in).
        response = Response(h=output.permute(1, 2, 0), fs=self.fs)
        self._cached_output, self._cached_response = output, response
        return response


def _named_terms(loss: Loss) -> list[tuple[float, Loss, str]]:
    """``(weight, term, unique name)`` for every leaf of the objective."""
    named: list[tuple[float, Loss, str]] = []
    seen: dict[str, int] = {}
    for weight, term in loss.terms():
        name = term.name
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}#{seen[name]}"
        named.append((weight, term, name))
    return named


def _criterion(term: Loss, name: str, factory: _ResponseFactory) -> Any:
    """Wrap a loss term in the ``nn.Module`` criterion FLAMO's trainer expects.

    The class is built per term because ``EagerTrainer`` keys its loss history
    by the criterion's class name.
    """
    import torch.nn as nn

    class _TermCriterion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.term = term
            self.factory = factory

        def forward(self, y_pred: Any, y_target: Any) -> Any:
            return self.term(self.factory.of(y_pred))

    return type(name, (_TermCriterion,), {})()
