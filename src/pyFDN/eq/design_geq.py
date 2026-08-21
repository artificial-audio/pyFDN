"""10-band graphic EQ design via constrained least squares.

Translation of designGEQ.m from fdnToolbox.

The design is linear in its target: the command gains that realize a target
magnitude are the least-squares solution of one matrix equation, and neither
that matrix nor the interpolation onto the control grid depends on the target.
:func:`design_geq` solves it with bounds, per call, as fdnToolbox does.
:func:`geq_design_matrix` folds the whole thing into one constant matrix
instead, which is what makes :func:`geq_sos` -- the same design as a closed-form,
differentiable map -- possible.

Reference:
    Välimäki and Reiss, "All About Audio Equalization: Solutions and Frontiers,"
    Applied Sciences, vol. 6, no. 5, p. 129, 2016.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from ..auxiliary.utils import hertz_to_rad
from ._backend import array_namespace
from .graphic_eq import graphic_eq
from .probe_sos import probe_sos

#: Octave band centres the graphic EQ places its peaking filters at, in Hz.
CENTER_FREQUENCIES = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], float)
#: Cut-off frequencies of the low and high shelving filters, in Hz.
SHELVING_CROSSOVER = np.array([46.0, 11360.0])
#: Bandwidth parameter of the peaking filters; ``Q = sqrt(R) / (R - 1)``.
BANDWIDTH_R = 2.7
#: Number of design bands: DC, the eight octave centres, Nyquist.
N_BANDS = len(CENTER_FREQUENCIES) + len(SHELVING_CROSSOVER)
#: Number of biquad sections the design produces: a flat gain plus one per band.
N_SECTIONS = N_BANDS + 1

_NUM_CONTROL = 100
_FFT_LEN = 2**16
_PROTOTYPE_GAIN = 10.0


def _band_omega(fs: float) -> tuple[np.ndarray, np.ndarray]:
    """The band layout in radians at ``fs``."""
    return hertz_to_rad(CENTER_FREQUENCIES, fs), hertz_to_rad(SHELVING_CROSSOVER, fs)


@lru_cache(maxsize=8)
def _control_problem(fs: float) -> tuple[np.ndarray, np.ndarray]:
    """The two constant matrices of the design, ``(G, W)``.

    ``G`` is the magnitude each unit command gain contributes at the control
    frequencies -- the design's system matrix -- and ``W`` interpolates the 10
    band targets onto the same control grid, so ``W @ target_db`` is the vector
    ``G`` is fitted against. Cached: both cost a 2**16-point ``freqz`` per band
    and depend on nothing but the sampling rate.
    """
    control_frequencies = np.round(np.logspace(0, np.log10(fs / 2.1), _NUM_CONTROL + 1))
    target_f = np.concatenate([[1.0], CENTER_FREQUENCIES, [float(fs)]])

    # W is read off column by column from np.interp itself -- np.interp is linear
    # in its values -- so it cannot drift from the interpolation it stands for.
    W = np.empty((len(control_frequencies), N_BANDS))
    for k in range(N_BANDS):
        unit = np.zeros(N_BANDS)
        unit[k] = 1.0
        W[:, k] = np.interp(control_frequencies, target_f, unit)

    center_omega, shelving_omega = _band_omega(fs)
    prototype_sos = graphic_eq(
        center_omega,
        shelving_omega,
        BANDWIDTH_R,
        _PROTOTYPE_GAIN * np.ones(N_SECTIONS),
    )
    G, _, _ = probe_sos(prototype_sos, control_frequencies, _FFT_LEN, fs)

    return G / _PROTOTYPE_GAIN, W


def geq_design_matrix(fs: float) -> np.ndarray:
    """Constant matrix taking 10 band targets in dB to 11 GEQ command gains.

    The unbounded closed form of :func:`design_geq`: the same least-squares
    problem, solved once for every target at once. Dropping the bounds is what
    buys the closed form, and they are inactive for the moderate band gains a
    decay or an output EQ asks for; a target steep enough to need them is
    realized a little less accurately here than by :func:`design_geq`.

    Returns
    -------
    np.ndarray
        Shape ``(11, 10)``: ``command_gains_db = M @ target_db``.
    """
    G, W = _control_problem(float(fs))
    return np.linalg.pinv(G) @ W


def geq_sos(target_db: Any, fs: float, *, design_matrix: Any = None) -> Any:
    """Band targets in dB to a normalized SOS bank, in one closed-form step.

    The design of :func:`design_geq` written as a map rather than a solve: the
    band targets become command gains through :func:`geq_design_matrix`, and the
    command gains become biquads through :func:`graphic_eq`. Both steps are
    plain arithmetic, so a torch target gives a torch SOS bank with gradients
    reaching back to it -- this is the ``map`` of the trainable absorption and
    output EQ in :mod:`pyFDN.train`.

    Parameters
    ----------
    target_db : array_like or torch.Tensor
        Target magnitude in dB at the 10 design bands (DC, 63 Hz … 8 kHz,
        Nyquist), shape ``(10,)`` or ``(10, n_channels)``.
    fs : float
        Sampling rate in Hz.
    design_matrix : array_like or torch.Tensor, optional
        The matrix of :func:`geq_design_matrix`, passed in when the caller keeps
        its own copy -- a torch module holds it as a buffer, on its own device
        and dtype. Computed here when omitted.

    Returns
    -------
    np.ndarray or torch.Tensor
        SOS bank of shape ``(11, 6) + target_db.shape[1:]``, normalized to
        ``a0 = 1``.
    """
    matrix = geq_design_matrix(fs) if design_matrix is None else design_matrix
    xp = array_namespace(target_db)
    if xp is not np and array_namespace(matrix) is np:
        # a torch target against the numpy matrix: meet on the target's side,
        # since that is the one carrying gradients, a device and a dtype.
        matrix = xp.as_tensor(matrix, dtype=target_db.dtype, device=target_db.device)
    center_omega, shelving_omega = _band_omega(float(fs))
    sos = graphic_eq(center_omega, shelving_omega, BANDWIDTH_R, matrix @ target_db)
    return sos / sos[:, 3:4, ...]


def design_geq(
    target_g: np.ndarray,
    fs: float = 48000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Design a 10-band graphic EQ matching a target magnitude response.

    The EQ has 8 peaking bandpass bands plus low and high shelving filters,
    plus a flat-gain section (11 sections total).

    Args:
        target_g: Target magnitude response in dB at 10 frequency bands
                  (DC=1 Hz, 63, 125, 250, 500, 1k, 2k, 4k, 8k Hz, Nyquist).
                  Shape ``(10,)``.
        fs: Sampling frequency in Hz (default 48000).

    Returns:
        ``(sos, target_f)`` where

        - ``sos`` — single SOS cascade of shape ``(n_sections, 6)`` (n_sections = 11).
        - ``target_f`` — 10-point frequency grid used for the target.
    """
    target_g = np.asarray(target_g, dtype=float).ravel()
    G, W = _control_problem(float(fs))
    target_interp = W @ target_g

    upper_bound = np.concatenate([[np.inf], 2 * _PROTOTYPE_GAIN * np.ones(N_BANDS)])
    lower_bound = -upper_bound

    result = lsq_linear(G, target_interp, bounds=(lower_bound, upper_bound))

    center_omega, shelving_omega = _band_omega(float(fs))
    sos = graphic_eq(center_omega, shelving_omega, BANDWIDTH_R, result.x)
    target_f = np.concatenate([[1.0], CENTER_FREQUENCIES, [float(fs)]])
    return sos, target_f
