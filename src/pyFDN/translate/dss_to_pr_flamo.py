"""FLAMO-only modal decomposition entry point.

This module intentionally exposes a *single* architecture:
FLAMO graph probing with torch autograd (Stage A backend).

Use this when you want a clear, dedicated path for FLAMO-based DSS2PR without
the mixed/manual probing options from ``dss_to_pr``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.translate.dss_to_pr import dss_to_pr


def dss_to_pr_flamo(
    delays: ArrayLike,
    A: Any,
    B: Any,
    C: Any,
    D: Any,
    *,
    inverse_matrix: Any | None = None,
    deflation_type: str = "fullDeflation",
    absorption_filters: Any | None = None,
    reject_unstable_poles: bool = False,
    quality_threshold: float | None = None,
    maximum_iterations: int = 50,
    verbose: bool = True,
    feedback_delay_units: int | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    FLAMO-architecture DSS2PR: autograd probing only.

    This is a thin, explicit wrapper around :func:`pyFDN.dss_to_pr` that fixes
    ``probe_backend="autograd"`` and rejects backend override options.
    """
    if "probe_backend" in kwargs or "probeBackend" in kwargs:
        raise TypeError(
            "dss_to_pr_flamo is autograd-only; do not pass probe_backend/probeBackend."
        )

    return dss_to_pr(
        delays,
        A,
        B,
        C,
        D,
        inverse_matrix=inverse_matrix,
        deflation_type=deflation_type,
        absorption_filters=absorption_filters,
        reject_unstable_poles=reject_unstable_poles,
        quality_threshold=quality_threshold,
        maximum_iterations=maximum_iterations,
        verbose=verbose,
        feedback_delay_units=feedback_delay_units,
        probe_backend="autograd",
        **kwargs,
    )

