"""Unit tests for pyFDN's vendored ``_EagerTrainer``.

These exercise the vendored fallback directly (``_EagerTrainer``), so they hold
whether or not the installed flamo ships its own ``EagerTrainer``. The active
``EagerTrainer`` symbol (flamo's if available, else the vendored copy) is checked
separately for API parity.
"""

import os

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from pyFDN.train._trainer import EagerTrainer, _EagerTrainer  # noqa: E402


class _Linear(nn.Module):
    """Trivial learnable system y = w * x, w initialised to zeros."""

    def __init__(self, n=3):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return self.w * x


class _ConstLoss(nn.Module):
    """Differentiable but constant loss (zero gradient) for plateau testing."""

    def forward(self, est, target):
        return (est * 0).sum() + 1.0


def test_adam_reduces_loss():
    # step_size huge so the LR holds at 0.1 -- this tests convergence of the
    # loop itself, not the default StepLR decay.
    opt = _EagerTrainer(
        _Linear(), max_steps=2000, lr=0.1, step_size=10_000, patience=50,
        tol=1e-9, log=False,
    )
    opt.register_criterion(nn.MSELoss(), 1)
    history = opt.optimize(torch.ones(1, 3), torch.tensor([[1.0, 2.0, 3.0]]))
    assert history["total"][-1] < history["total"][0]
    assert history["total"][-1] < 1e-3


def test_loss_history_structure():
    opt = _EagerTrainer(
        _Linear(), max_steps=10, lr=0.1, patience=100, tol=0.0, log=False
    )
    opt.register_criterion(nn.MSELoss(), 1)
    history = opt.optimize(torch.ones(1, 3), torch.tensor([[1.0, 2.0, 3.0]]))
    assert "total" in history and "MSELoss" in history
    assert len(history["total"]) == len(history["MSELoss"]) == 10


def test_plateau_early_stop():
    opt = _EagerTrainer(
        _Linear(), max_steps=100, lr=0.1, patience=3, tol=1e-6, log=False
    )
    opt.register_criterion(_ConstLoss(), 1)
    history = opt.optimize(torch.ones(1, 3), torch.zeros(1, 3))
    assert len(history["total"]) < 100  # stopped early
    assert len(history["total"]) <= 5  # ~ patience + 1


def test_unknown_optimizer_raises():
    with pytest.raises(ValueError, match="optimizer"):
        _EagerTrainer(_Linear(), optimizer="sgd")


def test_lbfgs_reduces_loss():
    opt = _EagerTrainer(
        _Linear(), max_steps=50, lr=1.0, optimizer="lbfgs", patience=10,
        tol=1e-12, log=False,
    )
    opt.register_criterion(nn.MSELoss(), 1)
    history = opt.optimize(torch.ones(1, 3), torch.tensor([[1.0, 2.0, 3.0]]))
    assert history["total"][-1] < 1e-4


def test_requires_model_criterion_receives_net():
    seen = {}

    class _UsesModel(nn.Module):
        def forward(self, est, target, model):
            seen["model"] = model
            return (est - target).pow(2).mean()

    net = _Linear()
    opt = _EagerTrainer(net, max_steps=3, lr=0.1, patience=100, tol=0.0, log=False)
    opt.register_criterion(_UsesModel(), 1, requires_model=True)
    opt.optimize(torch.ones(1, 3), torch.tensor([[1.0, 2.0, 3.0]]))
    assert seen["model"] is net


def test_save_checkpoints_writes_files(tmp_path):
    opt = _EagerTrainer(
        _Linear(), max_steps=5, lr=0.1, save_checkpoints=True,
        train_dir=str(tmp_path), patience=100, tol=0.0, log=False,
    )
    opt.register_criterion(nn.MSELoss(), 1)
    opt.optimize(torch.ones(1, 3), torch.tensor([[1.0, 2.0, 3.0]]))
    assert len(os.listdir(tmp_path / "checkpoints")) >= 1


def test_save_checkpoints_requires_train_dir():
    with pytest.raises(AssertionError):
        _EagerTrainer(_Linear(), save_checkpoints=True, train_dir=None)


def test_active_eager_trainer_has_parity_api():
    # The public EagerTrainer alias must expose the constructor/optimize() surface
    # pyFDN relies on (it is always the vendored copy).
    opt = EagerTrainer(_Linear(), max_steps=5, lr=0.1, log=False)
    opt.register_criterion(nn.MSELoss(), 1)
    history = opt.optimize(torch.ones(1, 3), torch.tensor([[1.0, 2.0, 3.0]]))
    assert "total" in history and len(history["total"]) == 5
