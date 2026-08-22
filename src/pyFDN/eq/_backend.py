"""Which array library a value belongs to.

The graphic-EQ design is used twice: in numpy, to design a filter once, and in
torch, as the differentiable ``map`` of a trainable module. The formulas are the
same either way -- the biquad prototypes are elementwise arithmetic on the gain,
and every frequency-dependent coefficient in them is a constant, computed from
``omega`` in plain Python before the gain is touched. So the same source can
serve both backends, provided it reaches for its array functions through the
namespace of its argument rather than importing one.

The five functions used this way -- ``sqrt``, ``ones_like``, ``zeros_like``,
``stack`` and ``concatenate`` -- carry the same name and the same positional
signature in numpy and in torch (``torch.concatenate`` is the array-API spelling
of ``torch.cat``), so no shim beyond this lookup is needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def array_namespace(x: Any) -> Any:
    """The array module ``x`` belongs to: :mod:`torch` for tensors, else numpy.

    torch is imported only when a tensor is actually passed, so the numpy design
    path keeps working in an environment without it.
    """
    if type(x).__module__.split(".", 1)[0] == "torch":
        import torch

        return torch
    return np
