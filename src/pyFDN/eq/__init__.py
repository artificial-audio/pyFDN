"""Filter sections and target-to-EQ design functions."""

from __future__ import annotations

from typing import Any

import numpy as np

from .biquads import (
    first_order_shelf_biquad,
    highshelf_biquad,
    lowshelf_biquad,
    one_pole_biquad,
    peaking_biquad,
)
from .design import (
    FilterDesign,
    _shelf_crossover_omega,
    decay_to_first_order_shelf,
    decay_to_geq,
    decay_to_one_pole,
    gain_to_first_order_shelf,
    gain_to_one_pole,
)
from .graphic_eq import (
    BANDWIDTH_R,
    CENTER_FREQUENCIES,
    SHELVING_CROSSOVER,
    design_geq,
    gain_to_geq,
    geq_design_matrix,
)
from .probe_sos import probe_sos

# Compatibility spellings from the original API.
absorption_geq = decay_to_geq
geq_sos = gain_to_geq
first_order_shelf_sos = first_order_shelf_biquad
one_pole_sos = one_pole_biquad


def first_order_absorption(
    rt_dc: Any,
    rt_nyquist: Any,
    delays: Any,
    fs: float,
    crossover_frequency: float | None = None,
) -> Any:
    """Compatibility alias for :func:`decay_to_first_order_shelf`."""
    return decay_to_first_order_shelf(
        rt_dc, rt_nyquist, crossover_frequency, delays, fs
    )


def first_order_shelving_eq(
    db_dc: Any,
    db_nyquist: Any,
    fs: float,
    crossover_frequency: float | None = None,
) -> Any:
    """Compatibility alias for :func:`gain_to_first_order_shelf`."""
    return gain_to_first_order_shelf(
        np.asarray(db_dc, dtype=float).ravel(),
        np.asarray(db_nyquist, dtype=float).ravel(),
        crossover_frequency,
        fs,
    )


def one_pole_absorption(rt_dc: Any, rt_nyquist: Any, delays: Any, fs: float) -> Any:
    """Compatibility alias for :func:`decay_to_one_pole`."""
    return decay_to_one_pole(rt_dc, rt_nyquist, delays, fs)


def bandpass_filter(omega_c: float, gain: Any, Q: float) -> tuple[Any, Any]:
    """Compatibility alias for :func:`peaking_biquad`."""
    return peaking_biquad(omega_c, gain, Q)


def shelving_filter(omega_c: float, gain: Any, filter_type: str) -> tuple[Any, Any]:
    """Compatibility wrapper for the explicit shelf biquad functions."""
    if filter_type == "low":
        return lowshelf_biquad(omega_c, gain)
    if filter_type == "high":
        return highshelf_biquad(omega_c, gain)
    raise ValueError(f"filter_type must be 'low' or 'high', got {filter_type!r}")


def shelf_crossover_omega(fs: float, crossover_frequency: float | None = None) -> float:
    """Compatibility wrapper for the first-order shelf crossover."""
    return _shelf_crossover_omega(fs, crossover_frequency)


__all__ = [
    "BANDWIDTH_R",
    "CENTER_FREQUENCIES",
    "FilterDesign",
    "SHELVING_CROSSOVER",
    "decay_to_first_order_shelf",
    "decay_to_geq",
    "decay_to_one_pole",
    "design_geq",
    "first_order_shelf_biquad",
    "gain_to_first_order_shelf",
    "gain_to_geq",
    "gain_to_one_pole",
    "geq_design_matrix",
    "highshelf_biquad",
    "lowshelf_biquad",
    "one_pole_biquad",
    "peaking_biquad",
    "probe_sos",
    # compatibility
    "absorption_geq",
    "bandpass_filter",
    "first_order_absorption",
    "first_order_shelf_sos",
    "first_order_shelving_eq",
    "geq_sos",
    "one_pole_absorption",
    "one_pole_sos",
    "shelf_crossover_omega",
    "shelving_filter",
]
