"""Map decay targets or gain targets to EQ coefficients.

The public functions are grouped by the quantity a caller knows first:
reverberation time for an attenuation filter, or gain in dB for an output EQ.
The lower-level filter-section formulas live in :mod:`pyFDN.eq.biquads`, while
the graphic-EQ implementation lives in :mod:`pyFDN.eq.graphic_eq`.
"""

from __future__ import annotations

import math
from typing import Any, Literal, get_args

import numpy as np

from ._backend import array_namespace
from ._design_record import design_value, with_design
from .biquads import first_order_shelf_biquad, one_pole_biquad
from .graphic_eq import (
    N_GRAPHIC_EQ_BANDS,
    N_GRAPHIC_EQ_SECTIONS,
    gain_to_geq,
    geq_design_matrix,
)

EQDesign = Literal["graphic_eq", "first_order_shelf", "one_pole"]
EQ_DESIGNS = get_args(EQDesign)


def decay_to_geq(
    rt: Any, delays: Any, fs: float, *, return_design: bool = False
) -> Any:
    """Design attenuation GEQs from ten reverberation times in seconds."""
    gain_db = _decay_to_gain_db(rt, delays, fs, N_GRAPHIC_EQ_BANDS)
    sos = gain_to_geq(gain_db, fs)
    return with_design(
        sos,
        {"type": "graphic_eq", "rt": design_value(rt)},
        return_design,
    )


def _shelf_crossover_omega(fs: float, crossover: float | None) -> float:
    crossover_hz = fs / 8.0 if crossover is None else float(crossover)
    return min(crossover_hz, fs / 5.0) / fs * 2.0 * math.pi


def gain_to_first_order_shelf(
    gain_db: Any,
    gain_db_nyquist: Any,
    crossover: float | None,
    fs: float,
    *,
    return_design: bool = False,
) -> Any:
    """Design a first-order shelf from its endpoint amplitudes in dB."""
    xp = array_namespace(gain_db)
    if xp is np:
        gain_db, gain_db_nyquist = np.broadcast_arrays(
            np.asarray(gain_db, dtype=float),
            np.asarray(gain_db_nyquist, dtype=float),
        )
    sos = first_order_shelf_biquad(
        10.0 ** (gain_db / 20.0),
        10.0 ** (gain_db_nyquist / 20.0),
        _shelf_crossover_omega(fs, crossover),
    )
    return with_design(
        sos,
        {
            "type": "first_order_shelf",
            "gain_db": design_value(gain_db),
            "gain_db_nyquist": design_value(gain_db_nyquist),
            "crossover": crossover,
        },
        return_design,
    )


def decay_to_first_order_shelf(
    rt: Any,
    rt_nyquist: Any,
    rt_crossover: float | None,
    delays: Any,
    fs: float,
    *,
    return_design: bool = False,
) -> Any:
    """Design first-order attenuation shelves from endpoint RTs in seconds."""
    gain_db = _decay_to_gain_db(rt, delays, fs)
    nyquist_db = _decay_to_gain_db(rt_nyquist, delays, fs)
    sos = gain_to_first_order_shelf(gain_db, nyquist_db, rt_crossover, fs)
    return with_design(
        sos,
        {
            "type": "first_order_shelf",
            "rt": design_value(rt),
            "rt_nyquist": design_value(rt_nyquist),
            "rt_crossover": rt_crossover,
        },
        return_design,
    )


def gain_to_one_pole(
    gain_db: Any,
    gain_db_nyquist: Any,
    *,
    return_design: bool = False,
) -> Any:
    """Design a one-pole filter from its endpoint amplitudes in dB."""
    xp = array_namespace(gain_db)
    if xp is np:
        gain_db, gain_db_nyquist = np.broadcast_arrays(
            np.asarray(gain_db, dtype=float),
            np.asarray(gain_db_nyquist, dtype=float),
        )
    sos = one_pole_biquad(
        10.0 ** (gain_db / 20.0),
        10.0 ** (gain_db_nyquist / 20.0),
    )
    return with_design(
        sos,
        {
            "type": "one_pole",
            "gain_db": design_value(gain_db),
            "gain_db_nyquist": design_value(gain_db_nyquist),
        },
        return_design,
    )


def decay_to_one_pole(
    rt: Any,
    rt_nyquist: Any,
    delays: Any,
    fs: float,
    *,
    return_design: bool = False,
) -> Any:
    """Design one-pole attenuation filters from endpoint RTs in seconds."""
    sos = gain_to_one_pole(
        _decay_to_gain_db(rt, delays, fs),
        _decay_to_gain_db(rt_nyquist, delays, fs),
    )
    return with_design(
        sos,
        {
            "type": "one_pole",
            "rt": design_value(rt),
            "rt_nyquist": design_value(rt_nyquist),
        },
        return_design,
    )


def _decay_to_gain_db(
    rt: Any,
    delays: Any,
    fs: float,
    n_parameters: int | None = None,
) -> Any:
    xp = array_namespace(rt)
    if xp is np:
        rt = np.asarray(rt, dtype=float)
        delays = np.asarray(delays, dtype=float).ravel()
    elif array_namespace(delays) is np:
        delays = xp.as_tensor(delays, dtype=rt.dtype, device=rt.device).flatten()

    if n_parameters is not None:
        if rt.ndim == 0 or rt.shape[0] != n_parameters:
            got = 0 if rt.ndim == 0 else rt.shape[0]
            raise ValueError(f"expected {n_parameters} reverberation times, got {got}")
        if rt.ndim == 1:
            rt = rt[:, None]
        elif rt.ndim != 2:
            raise ValueError(f"rt must be 1- or 2-dimensional, got shape {rt.shape}")
        if rt.shape[1] not in (1, delays.shape[0]):
            raise ValueError(
                "a per-line RT must have one column per delay: "
                f"expected {delays.shape[0]}, got {rt.shape[1]}"
            )
    return -60.0 * delays / (rt * float(fs))


def _design_parameter_count(design: EQDesign) -> int:
    _validate_filter_design(design)
    return N_GRAPHIC_EQ_BANDS if design == "graphic_eq" else 2


def _design_section_count(design: EQDesign) -> int:
    _validate_filter_design(design)
    return N_GRAPHIC_EQ_SECTIONS if design == "graphic_eq" else 1


def _design_buffers(design: EQDesign, fs: float) -> dict[str, np.ndarray]:
    _validate_filter_design(design)
    if design == "graphic_eq":
        return {"graphic_eq_matrix": geq_design_matrix(fs)}
    return {}


def _gain_to_design(
    gain_db: Any,
    design: EQDesign,
    fs: float,
    *,
    crossover: float | None = None,
    graphic_eq_matrix: Any = None,
) -> Any:
    _validate_filter_design(design)
    if design == "graphic_eq":
        return gain_to_geq(gain_db, fs, design_matrix=graphic_eq_matrix)
    if design == "first_order_shelf":
        return gain_to_first_order_shelf(gain_db[0], gain_db[1], crossover, fs)
    return gain_to_one_pole(gain_db[0], gain_db[1])


def _validate_filter_design(design: str) -> None:
    if design not in EQ_DESIGNS:
        raise ValueError(
            "design must be 'graphic_eq', 'first_order_shelf', or 'one_pole', "
            f"got {design!r}"
        )


__all__ = [
    "EQ_DESIGNS",
    "EQDesign",
    "decay_to_first_order_shelf",
    "decay_to_geq",
    "decay_to_one_pole",
    "gain_to_first_order_shelf",
    "gain_to_one_pole",
]
