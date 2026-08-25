"""The two loss families, and how they compose.

A training objective is a weighted sum of losses, built with ``+`` and ``*``::

    loss = pyFDN.FlatMagnitude() + 0.2 * pyFDN.Sparsity(A)

Losses come in two families, distinguished by what they read:

* :class:`ResponseLoss` -- a function of the model's impulse response
  (:class:`~pyFDN.train.response.Response`).
* :class:`ParameterLoss` -- a function of one model parameter, referenced by a
  :class:`~pyFDN.train.params.ParamRef` it holds.

Both are called with the response, so the trainer needs no special case; a
parameter loss simply ignores it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from pyFDN.train.params import ParamRef
    from pyFDN.train.response import Response


class Loss(ABC):
    """Base class for training losses. Combine with ``+`` and scale with ``*``."""

    @abstractmethod
    def __call__(self, response: Response) -> torch.Tensor:
        """The loss value, as a differentiable scalar tensor."""

    @property
    def name(self) -> str:
        """Short label, used as the key in :attr:`pyFDN.TrainLog.loss_log`."""
        return type(self).__name__

    def terms(self) -> list[tuple[float, Loss]]:
        """Flatten into ``(weight, loss)`` leaves. The trainer registers these."""
        return [(1.0, self)]

    def check(self, model: Any) -> None:
        """Preflight against the model it will train, before the first step.

        Override to reject or warn about a model this loss cannot be fit on --
        raising here beats a silently useless optimization run. Most losses
        place no demands on the model, hence the no-op default.
        """
        return None

    # Annotated `object` so the NotImplemented fallbacks are real code paths:
    # Python needs them to hand an unsupported operand back to the other side.
    def __add__(self, other: object) -> Loss:
        if not isinstance(other, Loss):
            return NotImplemented
        return Sum([*self._as_sum(), *other._as_sum()])

    def __mul__(self, weight: object) -> Loss:
        if not isinstance(weight, int | float):
            return NotImplemented
        return Scaled(float(weight), self)

    __rmul__ = __mul__

    def _as_sum(self) -> list[Loss]:
        return [self]


class ResponseLoss(Loss):
    """A loss on the impulse response. Implement :meth:`__call__`."""


class ParameterLoss(Loss):
    """A loss on one model parameter, referenced by ``ref``.

    Implement :meth:`penalty`; the response argument is ignored.
    """

    def __init__(self, ref: ParamRef) -> None:
        self.ref = ref

    @abstractmethod
    def penalty(self, value: torch.Tensor) -> torch.Tensor:
        """The penalty on the parameter's mapped ``value``."""

    def __call__(self, response: Response) -> torch.Tensor:
        return self.penalty(self.ref.value())

    @property
    def name(self) -> str:
        return f"{type(self).__name__}[{self.ref.name}]"


class Scaled(Loss):
    """``weight * loss``. Built by :meth:`Loss.__mul__`, rarely by hand."""

    def __init__(self, weight: float, loss: Loss) -> None:
        self.weight = float(weight)
        self.loss = loss

    def __call__(self, response: Response) -> torch.Tensor:
        return self.weight * self.loss(response)

    @property
    def name(self) -> str:
        return self.loss.name

    def terms(self) -> list[tuple[float, Loss]]:
        # Fold the weight into the leaves, so the trainer can log each term
        # unweighted and still sum the weighted total.
        return [(self.weight * w, leaf) for w, leaf in self.loss.terms()]

    def check(self, model: Any) -> None:
        self.loss.check(model)

    def __repr__(self) -> str:
        return f"{self.weight} * {self.loss!r}"


class Sum(Loss):
    """A sum of losses. Built by :meth:`Loss.__add__`, rarely by hand."""

    def __init__(self, losses: list[Loss]) -> None:
        self.losses = list(losses)

    def __call__(self, response: Response) -> torch.Tensor:
        total = self.losses[0](response)
        for loss in self.losses[1:]:
            total = total + loss(response)
        return total

    @property
    def name(self) -> str:
        return " + ".join(loss.name for loss in self.losses)

    def terms(self) -> list[tuple[float, Loss]]:
        return [term for loss in self.losses for term in loss.terms()]

    def check(self, model: Any) -> None:
        for loss in self.losses:
            loss.check(model)

    def _as_sum(self) -> list[Loss]:
        return list(self.losses)

    def __repr__(self) -> str:
        return " + ".join(repr(loss) for loss in self.losses)
