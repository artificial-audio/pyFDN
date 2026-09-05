"""Build time-domain processing graphs from a DSS system or a complete FDN build."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.build import FDNBuild
from pyFDN.td.connectors import Parallel, Recursion, Series
from pyFDN.td.operators import Delay, Gain, MatrixFIR, SOSBank, TimeOperator


def _feedback_operator(A: np.ndarray) -> TimeOperator:
    if A.ndim == 2:
        return Gain(A)
    if A.ndim == 3:
        return MatrixFIR(A)
    raise ValueError("A must be a 2-D (static) or 3-D (FIR) matrix")


def _append_sos(ops: list[TimeOperator], sos: np.ndarray | None) -> None:
    if sos is not None:
        ops.append(SOSBank(sos))


def dss_to_td(
    delays: ArrayLike,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    *,
    post_delay: np.ndarray | None = None,
    post_matrix: np.ndarray | None = None,
    post_output: np.ndarray | None = None,
    block_size: int | None = None,
) -> TimeOperator:
    """Assemble a delay state-space (DSS) system as a stateful ``td`` graph.

    The returned graph contains the input, feedback, output, and direct gains;
    the delay bank; and an :class:`~pyFDN.td.SOSBank` for each filter hook
    given. It can process a whole signal with ``process_signal``, process a
    stream with ``process_block``, and return to zero state with ``reset``.

    ``td.Recursion`` contributes ``block_size`` samples of delay to its forward
    path. This factory compensates for that implementation delay by shortening
    every explicit delay by the same amount, so the graph implements ``delays``
    exactly. To build the graph for a complete :class:`~pyFDN.FDNBuild`, use
    :func:`pyFDN.build_to_td`.

    Parameters
    ----------
    delays : array-like
        Positive delay lengths in samples, one per delay line.
    A : ndarray
        Feedback matrix, ``(N, N)`` static or ``(N, N, order)`` FIR.
    B, C, D : ndarray
        Input, output, and direct gain matrices.
    post_delay, post_matrix, post_output : ndarray, optional
        Optional SOS filter banks for the in-loop, feedback-path, and wet-signal
        hooks, in the same positions :func:`pyFDN.process_dss` uses.
    block_size : int, optional
        Internal recursion block size. Defaults to the shorter of 4096 samples
        and the shortest delay. It must be positive and cannot exceed the
        shortest delay.

    Returns
    -------
    TimeOperator
        A stateful graph from FDN inputs to outputs.
    """
    delays_arr = np.asarray(delays, dtype=int).reshape(-1)
    if delays_arr.size == 0 or np.any(delays_arr <= 0):
        raise ValueError("Delays must be a non-empty array of positive integers")

    shortest_delay = int(np.min(delays_arr))
    if block_size is None:
        block_size = min(2**12, shortest_delay)
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if block_size > shortest_delay:
        raise ValueError("block_size cannot exceed the shortest FDN delay")

    forward_ops: list[TimeOperator] = [Delay(delays_arr - block_size)]
    _append_sos(forward_ops, post_delay)

    feedback_ops: list[TimeOperator] = [_feedback_operator(np.asarray(A, dtype=float))]
    _append_sos(feedback_ops, post_matrix)

    # Recursion normally warns callers to compensate its inherent delay. The
    # explicit delays above are already compensated by this factory.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Recursion block processing inserts",
            category=UserWarning,
        )
        recursion = Recursion(
            Series(forward_ops),
            Series(feedback_ops),
            block_size=block_size,
            delay_position="forward",
        )

    wet_ops: list[TimeOperator] = [
        Gain(B),
        recursion,
        Gain(C),
    ]
    _append_sos(wet_ops, post_output)

    return Parallel([Series(wet_ops), Gain(D)], sum_output=True)


def build_to_td(build: FDNBuild, *, block_size: int | None = None) -> TimeOperator:
    """Assemble a complete :class:`FDNBuild` as a stateful ``td`` graph.

    Thin wrapper over :func:`dss_to_td` that unpacks the build's DSS system
    (``A``/``B``/``C``/``D``/``delays``) and its three filter hooks. ``fs`` is
    metadata here; delays and SOS coefficients are already expressed in
    samples.

    Parameters
    ----------
    build : FDNBuild
        Complete baked FDN configuration.
    block_size : int, optional
        See :func:`dss_to_td`.

    Returns
    -------
    TimeOperator
        A stateful graph from FDN inputs to outputs.
    """
    return dss_to_td(
        build.delays,
        build.A,
        build.B,
        build.C,
        build.D,
        post_delay=build.post_delay,
        post_matrix=build.post_matrix,
        post_output=build.post_output,
        block_size=block_size,
    )
