"""Ten-band graphic equalizer design.

This module owns the graphic-EQ band layout, least-squares control problem,
and biquad assembly. Public function names use the established ``geq``
abbreviation; ``graphic_eq`` is used when naming the design itself.
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear, minimize
from scipy.signal import sosfreqz

from ..auxiliary.utils import array_namespace, hertz_to_rad
from ._design_record import design_value, with_design
from .biquads import highshelf_biquad, lowshelf_biquad, peaking_biquad
from .probe_sos import probe_sos

COMMAND_FREQUENCIES = 16000.0 / 2.0 ** np.arange(9, -1, -1)
CENTER_FREQUENCIES = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], float)
SHELVING_CROSSOVER = np.array([46.0, 11360.0])
BANDWIDTH_R = 2.7
N_GRAPHIC_EQ_BANDS = 10
N_GRAPHIC_EQ_SECTIONS = 11

_NUM_CONTROL = 100
_NUM_CEILING = 600
_NUM_CHECK = 2**13
_CEILING_TOLERANCE_DB = 1e-3
_FFT_LEN = 2**16
_PROTOTYPE_GAIN_DB = 10.0


def _band_omega(fs: float) -> tuple[np.ndarray, np.ndarray]:
    return hertz_to_rad(CENTER_FREQUENCIES, fs), hertz_to_rad(SHELVING_CROSSOVER, fs)


def _geq_sections(
    center_omega: np.ndarray,
    shelving_omega: np.ndarray,
    bandwidth_r: float,
    gain_db: Any,
) -> Any:
    """Turn GEQ command gains into an unnormalized SOS bank."""
    xp = array_namespace(gain_db)
    if xp is np:
        gain_db = np.asarray(gain_db, dtype=float)
    expected = len(center_omega) + 3
    if gain_db.ndim == 0 or gain_db.shape[0] != expected:
        got = 0 if gain_db.ndim == 0 else gain_db.shape[0]
        raise ValueError(f"expected {expected} GEQ gains, got {got}")

    q = math.sqrt(bandwidth_r) / (bandwidth_r - 1.0)
    gains = 10.0 ** (gain_db / 20.0)
    sections = []
    for band in range(expected):
        gain = gains[band]
        if band == 0:
            zero, one = xp.zeros_like(gain), xp.ones_like(gain)
            b = xp.stack([gain, zero, zero], 0)
            a = xp.stack([one, zero, zero], 0)
        elif band == 1:
            b, a = lowshelf_biquad(shelving_omega[0], gain)
        elif band == expected - 1:
            b, a = highshelf_biquad(shelving_omega[1], gain)
        else:
            b, a = peaking_biquad(center_omega[band - 2], gain, q)
        sections.append(xp.concatenate([b, a], 0))
    return xp.stack(sections, 0)


@lru_cache(maxsize=8)
def _geq_control_problem(fs: float) -> tuple[np.ndarray, np.ndarray]:
    control_frequencies = np.round(np.logspace(0, np.log10(fs / 2.1), _NUM_CONTROL + 1))
    interpolation = np.empty((len(control_frequencies), N_GRAPHIC_EQ_BANDS))
    for band in range(N_GRAPHIC_EQ_BANDS):
        unit = np.zeros(N_GRAPHIC_EQ_BANDS)
        unit[band] = 1.0
        interpolation[:, band] = np.interp(
            control_frequencies, COMMAND_FREQUENCIES, unit
        )

    return _prototype_system(fs, control_frequencies), interpolation


@lru_cache(maxsize=8)
def _geq_ceiling_problem(fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Dense check grid (DC, then log-spaced to Nyquist) and its model."""
    frequencies = np.r_[0.0, np.logspace(0, np.log10(fs / 2), _NUM_CEILING)]
    return frequencies, _prototype_system(fs, frequencies)


def _prototype_system(fs: float, frequencies: np.ndarray) -> np.ndarray:
    """Per-section dB response per dB of command gain (the linear model)."""
    center_omega, shelving_omega = _band_omega(fs)
    prototype = _geq_sections(
        center_omega,
        shelving_omega,
        BANDWIDTH_R,
        _PROTOTYPE_GAIN_DB * np.ones(N_GRAPHIC_EQ_SECTIONS),
    )
    system, _, _ = probe_sos(prototype, frequencies, _FFT_LEN, fs)
    return system / _PROTOTYPE_GAIN_DB


def geq_design_matrix(fs: float) -> np.ndarray:
    """Return the constant map from ten band targets to eleven command gains."""
    system, interpolation = _geq_control_problem(float(fs))
    return np.linalg.pinv(system) @ interpolation


def gain_to_geq(
    gain_db: Any,
    fs: float,
    *,
    design_matrix: Any = None,
    return_design: bool = False,
) -> Any:
    """Design a ten-band graphic EQ from amplitudes in dB.

    ``gain_db`` has shape ``(10,)`` or ``(10, n_channels)`` and is ordered on
    :data:`COMMAND_FREQUENCIES` (31.25 Hz through 16 kHz).
    """
    xp = array_namespace(gain_db)
    if xp is np:
        gain_db = np.asarray(gain_db, dtype=float)
    if gain_db.ndim == 0 or gain_db.shape[0] != N_GRAPHIC_EQ_BANDS:
        got = 0 if gain_db.ndim == 0 else gain_db.shape[0]
        raise ValueError(f"graphic_eq takes {N_GRAPHIC_EQ_BANDS} gains, got {got}")

    matrix = geq_design_matrix(fs) if design_matrix is None else design_matrix
    if xp is not np and array_namespace(matrix) is np:
        matrix = xp.as_tensor(matrix, dtype=gain_db.dtype, device=gain_db.device)
    center_omega, shelving_omega = _band_omega(float(fs))
    sos = _geq_sections(center_omega, shelving_omega, BANDWIDTH_R, matrix @ gain_db)
    sos = sos / sos[:, 3:4, ...]
    return with_design(
        sos,
        {"type": "graphic_eq", "gain_db": design_value(gain_db)},
        return_design,
    )


def gain_to_bounded_geq(
    gain_db: Any,
    fs: float,
    *,
    max_command_gain_db: float = 20.0,
) -> np.ndarray:
    """Design a graphic EQ with bounded internal section gains.

    This is the constrained, NumPy-only counterpart of :func:`gain_to_geq`.
    The flat-gain section remains unbounded; each of the ten frequency-shaped
    sections is limited to ``max_command_gain_db`` in either direction.

    ``gain_db`` has shape ``(10,)`` or ``(10, n_channels)`` and is ordered on
    :data:`COMMAND_FREQUENCIES` (31.25 Hz through 16 kHz).
    """
    if type(gain_db).__module__.split(".", 1)[0] == "torch":
        raise TypeError("gain_to_bounded_geq is NumPy-only")
    gain_db = np.asarray(gain_db, dtype=float)
    if gain_db.ndim not in (1, 2) or gain_db.shape[0] != N_GRAPHIC_EQ_BANDS:
        got = 0 if gain_db.ndim == 0 else gain_db.shape[0]
        raise ValueError(f"graphic_eq takes {N_GRAPHIC_EQ_BANDS} gains, got {got}")

    limit = float(max_command_gain_db)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("max_command_gain_db must be finite and positive")

    system, interpolation = _geq_control_problem(float(fs))
    targets = gain_db[:, None] if gain_db.ndim == 1 else gain_db
    bound = np.concatenate([[np.inf], np.full(N_GRAPHIC_EQ_BANDS, limit)])
    command_gains = np.column_stack(
        [
            lsq_linear(
                system,
                interpolation @ targets[:, channel],
                bounds=(-bound, bound),
            ).x
            for channel in range(targets.shape[1])
        ]
    )
    if gain_db.ndim == 1:
        command_gains = command_gains[:, 0]

    center_omega, shelving_omega = _band_omega(float(fs))
    sos = _geq_sections(
        center_omega,
        shelving_omega,
        BANDWIDTH_R,
        command_gains,
    )
    return sos / sos[:, 3:4, ...]


def gain_to_passive_geq(
    gain_db: Any,
    fs: float,
    *,
    margin_db: float = 0.05,
    max_command_gain_db: float | None = 40.0,
    max_iterations: int = 8,
    rt_tolerance: float | None = 1.5,
) -> np.ndarray:
    """Design a graphic EQ that stays below unit gain, for use as an in-loop attenuation filter.

    The result stays at least ``margin_db`` below 0 dB across the spectrum (checked on a dense frequency grid), so an FDN with a lossless feedback matrix always decays. Sharp target profiles come out smoother than requested; a :class:`RuntimeWarning` reports when the result misses the target by more than ``rt_tolerance``.

    Args:
        gain_db: Target gains in dB, shape ``(10,)`` or ``(10, n_channels)``, ordered on :data:`COMMAND_FREQUENCIES` (31.25 Hz to 16 kHz). For reverberation times ``rt`` in seconds and delays in samples, pass ``-60 * delays / (fs * rt)``, or use :func:`~pyFDN.decay_to_geq`.
        fs: Sampling frequency in Hz.
        margin_db: Minimum attenuation in dB at every frequency. ``0`` allows unit gain; larger values shorten the longest decays.
        max_command_gain_db: Limit in dB on each band filter's gain; ``None`` removes it.
        max_iterations: Maximum number of refinement passes. Increase it if a warning reports a lowered flat gain.
        rt_tolerance: Warn when the achieved reverberation time at any band differs from the requested one by more than this factor. Must exceed 1; ``None`` disables the warning.

    Returns:
        SOS bank of shape ``(11, 6)`` or ``(11, 6, n_channels)``.
    """
    return _passive_geq(
        gain_db,
        fs,
        margin_db=margin_db,
        max_command_gain_db=max_command_gain_db,
        max_iterations=max_iterations,
        rt_tolerance=rt_tolerance,
    )


def _passive_geq(
    gain_db: Any,
    fs: float,
    *,
    margin_db: float = 0.05,
    max_command_gain_db: float | None = 40.0,
    max_iterations: int = 8,
    rt_tolerance: float | None = 1.5,
) -> np.ndarray:
    """Shared body of the passive design; call it directly from public functions so ``stacklevel=3`` reaches the user."""
    if type(gain_db).__module__.split(".", 1)[0] == "torch":
        raise TypeError("the passive graphic EQ design is NumPy-only")
    gain_db = np.asarray(gain_db, dtype=float)
    if gain_db.ndim not in (1, 2) or gain_db.shape[0] != N_GRAPHIC_EQ_BANDS:
        got = 0 if gain_db.ndim == 0 else gain_db.shape[0]
        raise ValueError(f"graphic_eq takes {N_GRAPHIC_EQ_BANDS} gains, got {got}")
    margin = float(margin_db)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("margin_db must be finite and non-negative")
    if max_command_gain_db is None:
        limit = None
    else:
        limit = float(max_command_gain_db)
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError("max_command_gain_db must be finite and positive")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be at least 1")
    tolerance = None if rt_tolerance is None else float(rt_tolerance)
    if tolerance is not None and not tolerance > 1.0:
        raise ValueError("rt_tolerance must exceed 1")

    fs = float(fs)
    system, interpolation = _geq_control_problem(fs)
    targets = gain_db[:, None] if gain_db.ndim == 1 else gain_db
    command_gains = np.empty((N_GRAPHIC_EQ_SECTIONS, targets.shape[1]))
    worst: tuple[float, int, int, float] | None = None  # factor, channel, band, dB
    for channel in range(targets.shape[1]):
        x, excess = _passive_command_gains(
            system,
            interpolation @ targets[:, channel],
            fs,
            -margin,
            limit,
            int(max_iterations),
        )
        if excess is not None:
            warnings.warn(
                f"passive GEQ fit did not converge and lowered its flat gain by {excess:.3g} dB, "
                "shortening decay at all frequencies; increase max_iterations of "
                "gain_to_passive_geq",
                RuntimeWarning,
                stacklevel=3,
            )
        command_gains[:, channel] = x
        if tolerance is not None:
            miss = _decay_rate_miss(x, targets[:, channel], fs)
            if miss is not None and (worst is None or miss[0] > worst[0]):
                worst = (miss[0], channel, *miss[1:])

    if tolerance is not None and worst is not None and worst[0] > tolerance:
        factor, channel, band, achieved = worst
        where = f" on channel {channel}" if targets.shape[1] > 1 else ""
        warnings.warn(
            f"passive GEQ fit misses the requested decay by {factor:.3g}x{where} "
            f"at {COMMAND_FREQUENCIES[band]:g} Hz ({targets[band, channel]:.3g} dB "
            f"requested, {achieved:.3g} dB achieved per pass); smooth the target "
            "profile for a closer fit",
            RuntimeWarning,
            stacklevel=3,
        )

    if gain_db.ndim == 1:
        command_gains = command_gains[:, 0]
    center_omega, shelving_omega = _band_omega(fs)
    sos = _geq_sections(center_omega, shelving_omega, BANDWIDTH_R, command_gains)
    return sos / sos[:, 3:4, ...]


def _decay_rate_miss(
    command_gains: np.ndarray, target: np.ndarray, fs: float
) -> tuple[float, int, float] | None:
    """Worst achieved/requested attenuation factor over decaying bands."""
    decaying = np.flatnonzero(target < 0.0)
    if decaying.size == 0:
        return None
    achieved = _cascade_db(command_gains, COMMAND_FREQUENCIES[decaying], fs)
    with np.errstate(divide="ignore"):
        ratio = target[decaying] / np.minimum(achieved, 0.0)
    factor = np.maximum(ratio, 1.0 / ratio)
    worst = int(np.argmax(factor))
    return float(factor[worst]), int(decaying[worst]), float(achieved[worst])


def _passive_command_gains(
    system: np.ndarray,
    target: np.ndarray,
    fs: float,
    ceiling_db: float,
    limit: float | None,
    max_iterations: int,
) -> tuple[np.ndarray, float | None]:
    """Command gains with the cascade capped at ``ceiling_db``, and the excess removed by the flat gain if the fit did not converge (else ``None``)."""
    frequencies, ceiling_system = _geq_ceiling_problem(fs)
    bounds = [(None, None)] + [(None if limit is None else -limit, limit)] * (
        N_GRAPHIC_EQ_SECTIONS - 1
    )
    lower = np.array([-np.inf if lo is None else lo for lo, _ in bounds])
    upper = -lower
    x = np.clip(np.linalg.lstsq(system, target, rcond=None)[0], lower, upper)
    correction = np.zeros_like(frequencies)
    converged = False
    for _ in range(max_iterations):
        x = minimize(
            lambda x: 0.5 * np.sum((system @ x - target) ** 2),
            x,
            jac=lambda x: system.T @ (system @ x - target),
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda x, cap: cap - ceiling_system @ x,
                    "jac": lambda x, cap: -ceiling_system,
                    "args": (ceiling_db - correction,),
                }
            ],
            options={"maxiter": 500, "ftol": 1e-10},
        ).x
        actual = _cascade_db(x, frequencies, fs)
        excess = actual.max() - ceiling_db
        if excess <= _CEILING_TOLERANCE_DB:
            converged = True
            break
        correction = actual - ceiling_system @ x
    # The flat gain shifts the cascade exactly, so it removes any excess left, including peaks between grid points.
    check = np.r_[0.0, np.logspace(0, np.log10(fs / 2), _NUM_CHECK)]
    excess = max(_cascade_db(x, check, fs).max() - ceiling_db, 0.0)
    x[0] -= excess
    return x, None if converged else excess


def _cascade_db(command_gains: np.ndarray, frequencies: np.ndarray, fs: float) -> Any:
    center_omega, shelving_omega = _band_omega(fs)
    sos = _geq_sections(center_omega, shelving_omega, BANDWIDTH_R, command_gains)
    _, response = sosfreqz(sos / sos[:, 3:4], worN=frequencies, fs=fs)
    return 20.0 * np.log10(np.maximum(np.abs(response), 1e-300))


__all__ = [
    "BANDWIDTH_R",
    "CENTER_FREQUENCIES",
    "COMMAND_FREQUENCIES",
    "N_GRAPHIC_EQ_BANDS",
    "N_GRAPHIC_EQ_SECTIONS",
    "SHELVING_CROSSOVER",
    "gain_to_bounded_geq",
    "gain_to_geq",
    "gain_to_passive_geq",
    "geq_design_matrix",
]
