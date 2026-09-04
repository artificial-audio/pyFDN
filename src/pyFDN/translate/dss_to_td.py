"""Build time-domain processing graphs from complete FDN configurations."""

from __future__ import annotations

import warnings

import numpy as np

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


def build_to_td(
    build: FDNBuild, *, block_size: int | None = None
) -> TimeOperator:
    """Assemble a complete :class:`FDNBuild` as a stateful ``td`` graph.

    The returned graph contains the input, feedback, output, and direct gains;
    the delay bank; and an :class:`~pyFDN.td.SOSBank` for each populated build
    hook. It can process a whole signal with ``process_signal``, process a
    stream with ``process_block``, and return to zero state with ``reset``.

    ``td.Recursion`` contributes ``block_size`` samples of delay to its forward
    path. This factory compensates for that implementation delay by shortening
    every explicit delay by the same amount, so the graph implements the delay
    lengths stored in ``build`` exactly.

    Parameters
    ----------
    build
        Complete baked FDN configuration. Its sampling rate is metadata here;
        delays and SOS coefficients are already expressed in samples.
    block_size
        Internal recursion block size. Defaults to the shorter of 4096 samples
        and the shortest FDN delay. It must be positive and cannot exceed the
        shortest delay.

    Returns
    -------
    TimeOperator
        A stateful graph from FDN inputs to outputs.
    """
    delays = np.asarray(build.delays, dtype=int).reshape(-1)
    if delays.size == 0 or np.any(delays <= 0):
        raise ValueError("Delays must be a non-empty array of positive integers")

    shortest_delay = int(np.min(delays))
    if block_size is None:
        block_size = min(2**12, shortest_delay)
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if block_size > shortest_delay:
        raise ValueError("block_size cannot exceed the shortest FDN delay")

    forward_ops: list[TimeOperator] = [Delay(delays - block_size)]
    _append_sos(forward_ops, build.post_delay)

    feedback_ops: list[TimeOperator] = [
        _feedback_operator(np.asarray(build.A, dtype=float))
    ]
    _append_sos(feedback_ops, build.post_matrix)

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
        Gain(build.B),
        recursion,
        Gain(build.C),
    ]
    _append_sos(wet_ops, build.post_output)

    return Parallel([Series(wet_ops), Gain(build.D)], sum_output=True)
