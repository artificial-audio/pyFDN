"""Delay related functions."""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import group_delay

from pyFDN.auxiliary.math import outer_sum_approximation
from pyFDN.auxiliary.utils import ensure_3d


def ms_to_smp(ms: float | np.ndarray, fs: float) -> np.ndarray:
    """Convert milliseconds to samples."""
    return np.round(np.array(ms) * fs / 1000).astype(int)


def _assign_flamo_delay_samples(module: Any, samples: np.ndarray) -> None:
    """Assign sample-valued delays while preserving a FLAMO module's tensor setup."""
    import torch

    values = torch.as_tensor(
        samples, dtype=module.param.dtype, device=module.param.device
    )
    module.assign_value(module.sample2s(values))


def flamo_delay_feedback_matrix(
    model: Any,
    delays: ArrayLike,
    delays_in: ArrayLike,
    delays_out: ArrayLike,
    *,
    inplace: bool = False,
) -> Any:
    """Place a delay-matrix-delay chain in a FLAMO FDN feedback path.

    The model is expected to have the topology produced by
    :func:`pyFDN.dss_to_flamo`. Its feedforward delay is set to ``delays`` and
    its feedback matrix is wrapped by delays of ``delays_in`` and
    ``delays_out`` samples. By default, the operation returns a deep copy.
    """
    delay_vectors = tuple(
        np.asarray(values, dtype=np.int64).ravel()
        for values in (delays, delays_in, delays_out)
    )
    if not delay_vectors[0].size:
        raise ValueError("delay vectors must not be empty")
    if any(values.size != delay_vectors[0].size for values in delay_vectors[1:]):
        raise ValueError("delays, delays_in, and delays_out must have equal lengths")
    if any(np.any(values < 0) for values in delay_vectors):
        raise ValueError("delay values must be non-negative")

    from flamo.processor import system

    target = model if inplace else copy.deepcopy(model)
    feedback_loop = target.get_core().branchA.feedback_loop
    base_delay = feedback_loop.feedforward
    feedback_matrix = feedback_loop.feedback

    _assign_flamo_delay_samples(base_delay, delay_vectors[0])
    extra_delay_in = copy.deepcopy(base_delay)
    extra_delay_out = copy.deepcopy(base_delay)
    _assign_flamo_delay_samples(extra_delay_in, delay_vectors[1])
    _assign_flamo_delay_samples(extra_delay_out, delay_vectors[2])

    feedback_loop.feedback = system.Series(
        OrderedDict(
            [
                ("delay_in", extra_delay_in),
                ("matrix", feedback_matrix),
                ("delay_out", extra_delay_out),
            ]
        )
    )
    return target


def swap_flamo_recursion_paths(model: Any, *, inplace: bool = False) -> Any:
    """Swap the feedforward and feedback paths of a FLAMO FDN recursion."""
    target = model if inplace else copy.deepcopy(model)
    feedback_loop = target.get_core().branchA.feedback_loop
    feedback_loop.feedforward, feedback_loop.feedback = (
        feedback_loop.feedback,
        feedback_loop.feedforward,
    )
    return target


def mgrpdelay(matrix: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Group delay for each entry of an FIR matrix."""

    mat = ensure_3d(matrix)
    n, m, _ = mat.shape
    delays = []
    freq_ref = None
    for row in range(n):
        row_entries = []
        for col in range(m):
            coeffs = mat[row, col, :]
            if np.allclose(coeffs, 0):
                row_entries.append(np.full(512, np.nan, dtype=float))
                continue
            w, gd = group_delay((coeffs, [1.0]))
            if freq_ref is None:
                freq_ref = w
            if gd.size < w.size:
                padded = np.full(w.size, np.nan, dtype=float)
                padded[: gd.size] = gd
                gd = padded
            row_entries.append(gd)
        delays.append(row_entries)
    if freq_ref is None:
        freq_ref = np.linspace(0.0, np.pi, 512)
    return np.asarray(delays), freq_ref


def matrix_delay_approximation(matrix: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Rank-1 approximation of matrix group delay."""

    GD, _ = mgrpdelay(matrix)
    GD[np.isinf(GD)] = np.nan
    matrix_delay = np.nanmean(GD, axis=2)

    gdl, gdr = outer_sum_approximation(matrix_delay)
    approximation = gdl + gdr
    approximation_error = gdl[:, np.newaxis] + gdr[np.newaxis, :] - matrix_delay
    return approximation, approximation_error
