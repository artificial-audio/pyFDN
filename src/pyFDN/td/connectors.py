from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.td.operators import Delay, TimeOperator

_RECURSION_BLOCK_SIZE = 1 << 6
# NOTE: the minimum path delay supported by Recursion if
#       twice as much in case of delay compensation


def _as_2d(block: ArrayLike) -> np.ndarray:
    """Coerce a signal block to ``(num_samples, channels)`` float array."""
    x = np.asarray(block, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if x.ndim != 2:
        raise ValueError("signal block must be 1-D or 2-D")
    return x


class Series(TimeOperator):
    """Chain operators left to right.
    Equivalent to FLAMO ``Series``.
    """

    def __init__(self, ops: list[TimeOperator]) -> None:
        self.ops = self._flatten_ops(ops)
        for i in range(1, len(self.ops)):
            if self.ops[i - 1].out_channels != self.ops[i].in_channels:
                raise ValueError(
                    f"Operator {i - 1} output-channel count does not match operator {i} input-channel count"
                )
        self.in_channels = self.ops[0].in_channels
        self.out_channels = self.ops[-1].out_channels

    def filter(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        for op in self.ops:
            x = op.filter(x)
        return x

    def reset(self) -> None:
        for op in self.ops:
            op.reset()

    def _flatten_ops(self, ops: list[TimeOperator]) -> list[TimeOperator]:
        """Flatten the operator list.
        Avoids nesting Series instances.

        Args:
            ops (list[TimeOperator] | TimeOperator): List of operators

        Raises:
            ValueError: At least one operator in argin ops is not child of TimeOperator
            ValueError: No operator in argin ops

        Returns:
            list[TimeOperator]: Flattened list of operators
        """
        flat: list[TimeOperator] = []
        for op in ops:
            if isinstance(op, Series):
                flat.extend(self._flatten_ops(op.ops))
            elif not isinstance(op, TimeOperator):
                raise ValueError("Operators need to be of class TimeOperator")
            else:
                flat.append(op)
        if not flat:
            raise ValueError("Series needs at least one operator")
        return flat


class Parallel(TimeOperator):
    """Feed the same input to every branch and combine the outputs (optional).
    Equivalent to FLAMO ``Parallel``.
    """

    def __init__(self, ops: list[TimeOperator], *, sum_output: bool = True) -> None:
        self.ops = self._flatten_ops(ops)
        self.sum_output = sum_output
        ins = {op.in_channels for op in ops}
        if len(ins) != 1:
            raise ValueError("Parallel operators must share input-channel count")
        self.in_channels = ops[0].in_channels
        if sum_output:
            outs = {op.out_channels for op in ops}
            if len(outs) != 1:
                raise ValueError(
                    "Summed Parallel operators must share out-channel count"
                )
            self.out_channels = ops[0].out_channels
        else:
            self.out_channels = sum(op.out_channels for op in ops)

    def filter(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        outs = [op.filter(x) for op in self.ops]
        if self.sum_output:
            acc = outs[0].copy()
            for out in outs[1:]:
                acc += out
            return acc
        return np.concatenate(outs, axis=1)

    def reset(self) -> None:
        for op in self.ops:
            op.reset()

    def _flatten_ops(self, ops: list[TimeOperator]) -> list[TimeOperator]:
        """Flatten the operator list.
        Avoids nesting Parallel instances.

        Args:
            ops (list[TimeOperator] | TimeOperator): List of operators

        Raises:
            ValueError: At least one operator in argin ops is not child of TimeOperator
            ValueError: No operator in argin ops

        Returns:
            list[TimeOperator]: Flattened list of operators
        """
        flat: list[TimeOperator] = []
        for op in ops:
            if isinstance(op, Parallel):
                flat.extend(self._flatten_ops(op.ops))
            elif not isinstance(op, TimeOperator):
                raise ValueError("Operators need to be of class TimeOperator")
            else:
                flat.append(op)
        if not flat:
            raise ValueError("Parallel needs at least one operator")
        return flat


class _RecursionState:
    def __init__(
        self, buffer_size: int, channels: int, position: str = "forward"
    ) -> None:
        if np.any(buffer_size <= 0):
            raise ValueError("Block size must be positive integers")
        self._position = position
        self.buffer_size = buffer_size
        self.channels = channels
        self.buffers = np.zeros((self.channels, self.buffer_size), dtype=float)
        self.pointers = np.zeros(self.channels, dtype=int)
        self._last_indices: np.ndarray | None = None

    def get_values(self, block_size: int) -> np.ndarray:
        if block_size > self.buffer_size:
            raise ValueError("Block size exceeds configured buffer size")
        offsets = (
            self.pointers[:, None] + np.arange(block_size)[None, :]
        ) % self.buffer_size
        self._last_indices = offsets
        gathered = self.buffers[np.arange(self.channels)[:, None], offsets]
        return gathered.T

    def set_values(self, block: ArrayLike) -> None:
        if self._last_indices is None:
            raise RuntimeError("get_values must be called before set_values")
        block_arr = np.asarray(block, dtype=float)
        if block_arr.shape != (self._last_indices.shape[1], self.channels):
            raise ValueError("Block shape mismatch when writing sample values")
        self.buffers[np.arange(self.channels)[:, None], self._last_indices] = (
            block_arr.T
        )

    def advance(self, block_size: int) -> None:
        self.pointers = (self.pointers + block_size) % self.buffer_size
        self._last_indices = None

    def reset(self) -> None:
        self.buffers[:] = 0.0
        self.pointers[:] = 0
        self._last_indices = None


class Recursion(TimeOperator):
    """Closed feedback loop with delay in the feedback path
    ``y[n] = fF(x[n] + fB(y[n-d]))``.
    Recursion processes audio in blocks, inherently adding one block-size
    of delay inside the loop. Compensation is left to the user.
    """

    def __init__(
        self,
        forward: TimeOperator,
        feedback: TimeOperator,
        delay_position: str = "forward",
    ) -> None:
        if forward.out_channels != feedback.in_channels:
            raise ValueError(
                "Forward output-channel count does not match feedback input-channel count"
            )
        if forward.in_channels != feedback.out_channels:
            raise ValueError(
                "Feedback output-channel count does not match forward input-channel count"
            )

        self._block_size = _RECURSION_BLOCK_SIZE

        self._forward = forward
        self._feedback = feedback
        self.in_channels = forward.in_channels
        self.out_channels = forward.out_channels

        match delay_position:
            case "forward":
                min_path_delay = self._find_min_delay(forward)
                if min_path_delay < self._block_size:
                    warnings.warn(
                        "Minimum delay in forward path cannot be shorter than Recursion internal block size, and cannot be compensated",
                        UserWarning,
                        stacklevel=2,
                    )
                self._filter_steps = self._filter_forward_delay
                self._state = _RecursionState(
                    self._block_size, self.out_channels, position=delay_position
                )
            case "feedback":
                min_path_delay = self._find_min_delay(feedback)
                if min_path_delay < self._block_size:
                    warnings.warn(
                        "Minimum delay in feedback path cannot be shorter than Recursion internal block size, and cannot be compensated",
                        UserWarning,
                        stacklevel=2,
                    )
                self._filter_steps = self._filter_feedback_delay
                self._state = _RecursionState(
                    self._block_size, self.in_channels, position=delay_position
                )
            case _:
                raise ValueError("delay_position value not valid")

    def filter(self, block: ArrayLike) -> np.ndarray:
        """Filter block of audio.

        Args:
            block (ArrayLike): Input block of audio.

        Raises:
            ValueError: Mismatch between audio-block channels and Recursion input channels

        Returns:
            np.ndarray: Output block of audio.
        """
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Recursion expects {self.in_channels} input channels")
        num_samples = x.shape[0]
        out = np.empty((num_samples, self.out_channels), dtype=float)
        n = 0
        while n < num_samples:
            block_size = min(self._block_size, num_samples - n)
            block_in = x[n : n + block_size, :]

            out[n : n + block_size, :] = self._filter_steps(block_in, block_size)

            n += block_size

        return out

    def _filter_forward_delay(self, block_in: ArrayLike, block_size: int) -> np.ndarray:
        state = self._state.get_values(block_size)

        fb_out = self._feedback.filter(state)
        fw_out = self._forward.filter(block_in + fb_out)

        self._state.set_values(fw_out)
        self._state.advance(block_size)

        return state

    def _filter_feedback_delay(
        self, block_in: ArrayLike, block_size: int
    ) -> np.ndarray:
        state = self._state.get_values(block_size)

        fw_out = self._forward.filter(block_in + state)
        fb_out = self._feedback.filter(fw_out)

        self._state.set_values(fb_out)
        self._state.advance(block_size)

        return fw_out

    def reset(self) -> None:
        """Resets the state of all internal TimeOperators."""
        self._forward.reset()
        self._feedback.reset()
        self._state.reset()

    def _find_min_delay(self, op: TimeOperator) -> int:
        """Returns the minimum input-to-output delay of a TimeOperator"""
        if isinstance(op, Delay):
            return int(np.min(op.delays))
        if isinstance(op, Series):
            return sum(self._find_min_delay(child) for child in op.ops)
        if isinstance(op, Parallel):
            return min(self._find_min_delay(child) for child in op.ops)
        return 0
