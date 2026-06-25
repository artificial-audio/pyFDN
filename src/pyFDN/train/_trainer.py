"""A lightweight single-system optimizer for fitting one FDN.

``EagerTrainer`` fits a single differentiable system from one fixed
``(input, target)`` tensor pair by running gradient descent directly -- no
``Dataset``/``DataLoader``/``expand``/``split``/``shuffle``/batch machinery.
Fitting one LTI system (e.g. an FDN) is a pure optimization problem: there is no
held-out data and nothing to generalize to, so the dataset stack flamo's
``Trainer`` carries is pure overhead here.

This module prefers an upstream ``flamo.optimize.trainer.EagerTrainer`` when the
installed flamo provides one, and otherwise falls back to the vendored
``_EagerTrainer`` below. The vendored copy is adapted from flamo (MIT, (c) 2024
Gloria Dal Santo, Gian Marco De Bortoli, Sebastian Jiro Schlecht) and trimmed to
pyFDN's needs (no ``masked_mse_loss`` warning -- pyFDN registers only
deterministic criteria), so it has no flamo dependency of its own.

torch is imported lazily inside the methods so this module imports without torch,
keeping ``import pyFDN`` torch-free.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any


class _EagerTrainer:
    """Fit one differentiable system on a fixed ``(input, target)`` pair.

    Criteria are registered exactly as in flamo's ``Trainer`` via
    :meth:`register_criterion`, so existing flamo loss functions (and the
    ``requires_model`` flag) transfer unchanged.

    Parameters
    ----------
    net : nn.Module
        The differentiable system to optimize.
    max_steps : int
        Maximum number of optimization (gradient) steps. Default 1000.
    lr : float
        Learning rate. Default 1e-3.
    optimizer : str
        ``"adam"`` or ``"lbfgs"`` (case-insensitive). Default ``"adam"``.
    step_size, step_factor : int, float
        ``StepLR`` period and decay factor (Adam only).
    tol : float
        Relative-improvement threshold for plateau early stop. Default 1e-6.
    patience : int
        Consecutive non-improving steps before stopping. Default 10.
    min_steps : int
        Warmup: early stopping is suppressed until this many steps have run, so a
        slow/flat start (before the optimizer settles into a basin) does not trip
        the plateau gate. Default 0 (early stopping allowed from the first step).
    log : bool
        Show the progress bar and print timing / plateau messages. Default True.
    train_dir : str, optional
        Directory for checkpoints (required if ``save_checkpoints``).
    save_checkpoints : bool
        Save the ``state_dict`` each step. Default False.
    device : str
        Device for optimization. Default ``"cpu"``.
    """

    def __init__(
        self,
        net: Any,
        max_steps: int = 1000,
        lr: float = 1e-3,
        optimizer: str = "adam",
        step_size: int = 50,
        step_factor: float = 0.1,
        tol: float = 1e-6,
        patience: int = 10,
        min_steps: int = 0,
        log: bool = True,
        train_dir: str | None = None,
        save_checkpoints: bool = False,
        device: str = "cpu",
    ):
        import torch

        self.device = device
        self.net = net.to(device)
        self.max_steps = max_steps
        self.lr = lr
        self.tol = tol
        self.patience = patience
        self.min_steps = min_steps
        self.log = log
        self.train_dir = train_dir
        self.save_checkpoints = save_checkpoints
        self.n_loss = 0
        self.criterion, self.alpha, self.requires_model = [], [], []

        if self.save_checkpoints:
            assert train_dir is not None and os.path.isdir(train_dir), (
                "save_checkpoints=True requires an existing train_dir."
            )

        self.optimizer_name = optimizer.lower()
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=step_factor
            )
        elif self.optimizer_name == "lbfgs":
            self.optimizer = torch.optim.LBFGS(
                self.net.parameters(), lr=self.lr, line_search_fn="strong_wolfe"
            )
            self.scheduler = None
        else:
            raise ValueError(
                f"Unknown optimizer {optimizer!r}. Use 'adam' or 'lbfgs'."
            )

    def register_criterion(
        self, criterion: Any, alpha: float = 1, requires_model: bool = False
    ) -> None:
        """Register a loss function and its weight (mirrors ``Trainer``)."""
        self.criterion.append(criterion.to(self.device))
        self.alpha.append(alpha)
        self.requires_model.append(requires_model)
        self.n_loss += 1

    def move_to_device(self, data: Any) -> Any:
        if isinstance(data, list):
            return [x.to(self.device) for x in data]
        return data.to(self.device)

    def _compute_loss(
        self, estimations: Any, targets: Any, log_dict: dict | None = None
    ) -> Any:
        """Weighted sum of registered criteria; optionally log each into ``log_dict``."""
        loss = 0
        for alpha, criterion, requires_model in zip(
            self.alpha, self.criterion, self.requires_model
        ):
            if requires_model:
                temp = criterion(estimations, targets, self.net)
            else:
                temp = criterion(estimations, targets)
            if log_dict is not None:
                log_dict[criterion.__class__.__name__].append(temp.item())
            loss = loss + alpha * temp
        return loss

    def optimize(self, input: Any, target: Any) -> dict:
        """Optimize on a single fixed ``(input, target)`` pair.

        Returns a loss history dict with key ``"total"`` plus one key per
        criterion class name; all lists have equal length (one entry per step).
        """
        import torch
        from tqdm import trange

        input = self.move_to_device(input)
        target = self.move_to_device(target)

        self.loss_history = {"total": []}
        for c in self.criterion:
            self.loss_history[c.__class__.__name__] = []

        best_loss = float("inf")
        counter = 0
        st = time.time()
        pbar = trange(self.max_steps, desc="Optimizing", disable=not self.log)
        for step in pbar:
            if self.optimizer_name == "adam":
                self.optimizer.zero_grad()
                est = self.net(input)
                loss = self._compute_loss(est, target, self.loss_history)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total = loss.item()
            else:  # lbfgs

                def closure():
                    self.optimizer.zero_grad()
                    est = self.net(input)
                    loss = self._compute_loss(est, target)
                    loss.backward()
                    return loss

                self.optimizer.step(closure)
                with torch.no_grad():
                    est = self.net(input)
                    total = self._compute_loss(est, target, self.loss_history).item()

            self.loss_history["total"].append(total)
            pbar.set_postfix_str(f"loss: {total:.6f}")

            if self.save_checkpoints:
                self.save_model(step)

            # plateau early stopping on relative improvement of the total loss
            if not math.isfinite(total):
                # a non-finite loss (diverged to inf/nan) is not progress -- count
                # it toward the plateau so a dead run stops instead of burning
                # every step.
                improved = False
            elif math.isfinite(best_loss):
                improved = (best_loss - total) / (abs(best_loss) + 1e-12) > self.tol
            else:
                # first finite loss after the inf init -- genuine improvement.
                improved = True
            if total < best_loss:
                best_loss = total
            if improved:
                counter = 0
            else:
                counter += 1
                # Suppress early stopping during warmup so a slow/flat start does
                # not trip the gate before the optimizer settles into a basin.
                if counter >= self.patience and (step + 1) >= self.min_steps:
                    if self.log:
                        print(f"Plateau reached at step {step}.")
                    break

        if self.log:
            print(f"Optimization time: {time.time() - st:.3f}s")
        return self.loss_history

    def save_model(self, step: int) -> None:
        """Save the model parameters to ``train_dir/checkpoints/model_e<step>.pt``."""
        import torch

        dir_path = os.path.join(self.train_dir, "checkpoints")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(
            self.net.state_dict(),
            os.path.join(dir_path, "model_e" + str(step) + ".pt"),
        )


try:  # prefer an upstream EagerTrainer when the installed flamo ships one
    from flamo.optimize.trainer import EagerTrainer
except ImportError:  # distributed flamo without it -> use the vendored copy
    EagerTrainer = _EagerTrainer
