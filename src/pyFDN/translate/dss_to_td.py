"""Build time-domain processing graphs from a DSS system or a complete FDN build."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.build import FDNBuild
from pyFDN.td.connectors import Parallel, Series
from pyFDN.td.operators import (
    Gain,
    MatrixFIR,
    RecursionState,
    SOSBank,
    TimeOperator,
)


def _feedback_operator(A: np.ndarray) -> TimeOperator:
    if A.ndim == 2:
        return Gain(A)
    if A.ndim == 3:
        return MatrixFIR(A)
    raise ValueError("A must be a 2-D (static) or 3-D (FIR) matrix")


class _BlockHook(TimeOperator):
    """Adapt any object with ``process_block`` to a ``channels``-wide operator."""

    def __init__(self, hook: Any, channels: int) -> None:
        self._hook = hook
        self.in_channels = self.out_channels = channels

    def process_block(self, block: ArrayLike) -> np.ndarray:
        return np.asarray(self._hook.process_block(block), dtype=float)

    def reset(self) -> None:
        reset = getattr(self._hook, "reset", None)
        if callable(reset):
            reset()


Hook = ArrayLike | TimeOperator | Any


def _append_hook(ops: list[TimeOperator], hook: Hook | None, channels: int) -> None:
    """Append a filter hook: an operator, a ``process_block`` object, or SOS."""
    if hook is None:
        return
    if isinstance(hook, TimeOperator):
        ops.append(hook)
    elif hasattr(hook, "process_block"):
        ops.append(_BlockHook(hook, channels))
    else:
        ops.append(SOSBank(hook))


class _DelayLoop(TimeOperator):
    """The FDN recursion: delays, ``post_delay``, and the feedback path.

    Maps the loop input ``B u`` to the delay-line outputs (after
    ``post_delay``), which drive both the output gains and the feedback path.
    The delay bank itself breaks the loop, so blocks of up to the shortest
    delay are processed exactly -- no extra delay to compensate -- and
    ``post_delay`` runs on the delay output in real time, which matters for
    time-varying or nonlinear hooks.
    """

    def __init__(
        self,
        delays: np.ndarray,
        feedback: TimeOperator,
        post_delay: TimeOperator | None,
        block_size: int,
    ) -> None:
        self.in_channels = self.out_channels = delays.size
        self._state = RecursionState(delays, block_size)
        self._block_size = block_size
        self._feedback = feedback
        self._post_delay = post_delay

    def process_block(self, block: ArrayLike) -> np.ndarray:
        x = np.asarray(block, dtype=float)
        out = np.empty_like(x)
        start = 0
        while start < x.shape[0]:
            n = min(self._block_size, x.shape[0] - start)
            delayed = self._state.get_values(n)
            if self._post_delay is not None:
                delayed = self._post_delay.process_block(delayed)
            feedback = self._feedback.process_block(delayed)
            self._state.set_values(x[start : start + n] + feedback)
            self._state.advance(n)
            out[start : start + n] = delayed
            start += n
        return out

    def reset(self) -> None:
        self._state.reset()
        self._feedback.reset()
        if self._post_delay is not None:
            self._post_delay.reset()


def dss_to_td(
    delays: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    *,
    post_delay: Hook | None = None,
    post_matrix: Hook | None = None,
    post_output: Hook | None = None,
    block_size: int | None = None,
) -> TimeOperator:
    """Assemble a delay state-space (DSS) system as a stateful ``td`` graph.

    The returned graph contains the input, feedback, output, and direct gains;
    the delay bank; and the filter hooks given. It can process a whole signal
    with ``process_signal``, process a stream with ``process_block``, and return
    to zero state with ``reset``.

    The delay bank breaks the feedback loop, so the graph processes blocks of
    up to ``block_size`` samples and implements ``delays`` exactly. To build the graph for a complete :class:`~pyFDN.FDNBuild`, use
    :func:`pyFDN.build_to_td`.

    Parameters
    ----------
    delays : array-like
        Positive delay lengths in samples, one per delay line.
    A : array-like
        Feedback matrix, ``(N, N)`` static or ``(N, N, order)`` FIR.
    B, C, D : array-like
        Input, output, and direct gain matrices.
    post_delay, post_matrix, post_output : optional
        In-loop (after the delays), feedback-path (after ``A``) and wet-signal
        (after ``C``) hooks. Each is an ``(sections, 6, channels)`` SOS bank
        (placed as a :class:`~pyFDN.td.SOSBank`), a
        :class:`~pyFDN.td.TimeOperator`, or any object with a
        ``process_block(block)`` method.
    block_size : int, optional
        Internal block size. Defaults to the shorter of 4096 samples
        and the shortest delay. It must be positive and cannot exceed the
        shortest delay.

    Returns
    -------
    TimeOperator
        A stateful graph from FDN inputs to outputs.
    """
    delays_arr = np.asarray(delays, dtype=int).reshape(-1)
    n_lines = delays_arr.size
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

    post_delay_ops: list[TimeOperator] = []
    _append_hook(post_delay_ops, post_delay, n_lines)
    feedback_ops: list[TimeOperator] = [_feedback_operator(np.asarray(A, dtype=float))]
    _append_hook(feedback_ops, post_matrix, n_lines)
    loop = _DelayLoop(
        delays_arr,
        Series(feedback_ops),
        post_delay_ops[0] if post_delay_ops else None,
        block_size,
    )

    wet_ops: list[TimeOperator] = [
        Gain(B),
        loop,
        Gain(C),
    ]
    _append_hook(wet_ops, post_output, np.shape(C)[0])

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
