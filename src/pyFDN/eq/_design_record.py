"""Small helpers for optional JSON-compatible EQ design records."""

from __future__ import annotations

from typing import Any

import numpy as np

DesignRecord = dict[str, Any]


def design_value(value: Any) -> Any:
    """Detach an array-like design target and convert it to JSON values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def with_design(value: Any, design: DesignRecord, return_design: bool) -> Any:
    """Return a coefficient value alone or paired with its design record."""
    return (value, design) if return_design else value


__all__ = ["DesignRecord", "design_value", "with_design"]
