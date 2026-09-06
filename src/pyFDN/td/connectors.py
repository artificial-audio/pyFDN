from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.td.operators import RecursionState, TimeOperator


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

    def process_block(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        for op in self.ops:
            x = op.process_block(x)
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

    def process_block(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        outs = [op.process_block(x) for op in self.ops]
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


class Recursion(TimeOperator):
    """Closed feedback loop, ``y[n] = fF(x[n] + fB(y[n - block_size]))``.

    A feedback loop cannot be evaluated sample by sample without an algebraic
    loop, so ``Recursion`` computes whole blocks of ``block_size`` samples at a
    time: it reads the loop state written by the previous block, runs both paths
    over the block, and writes the result back. That read-before-write is what
    breaks the loop -- and it inserts **exactly ``block_size`` samples of delay
    into the loop**, on top of whatever delay the operators themselves have.

    The inserted delay is not compensated automatically; a warning is issued at
    construction as a reminder. To compensate, shorten the delay lines on the
    path named by ``delay_position`` by ``block_size`` samples, e.g.
    ``Delay(delays - block_size)``. The uncompensated delays must be at least
    ``block_size`` long; a shortened :class:`~pyFDN.td.Delay` may be zero.

    Parameters
    ----------
    forward
        Forward path ``fF``, from the loop input to the loop output.
    feedback
        Feedback path ``fB``, from the loop output back to the loop input.
    block_size
        Processing block size in samples, and therefore the amount of delay
        inserted into the loop. Required: it is a property of the loop the
        caller has to choose and compensate for, not an implementation detail.
        When compensating explicit delay lines, it must not exceed their
        shortest uncompensated delay.
    delay_position
        Which side of the loop the inserted ``block_size`` delay lands on, i.e.
        where the state buffer sits:

        ``"forward"`` (default)
            After the forward path: the loop output is read out of the state
            buffer, so it is the forward output that arrives ``block_size``
            samples late (``y[n] = fF(...)[n - block_size]``). Compensate on the
            forward path. Use this when the delay lines are in the forward path
            (the usual FDN layout: delays forward, mixing matrix feedback).
        ``"feedback"``
            After the feedback path: the forward output is returned immediately
            and it is the signal fed back that is ``block_size`` samples late
            (``y[n] = fF(x[n] + fB(y)[n - block_size])``). Compensate on the
            feedback path. Use this when the delay lines are in the feedback
            path, e.g. an outer acoustic-feedback loop around a whole system.

        Both give the same total loop delay; they differ in where in the loop
        that delay sits, hence in which path has to absorb the compensation and
        whether the operator's own output is delayed.
    """

    def __init__(
        self,
        forward: TimeOperator,
        feedback: TimeOperator,
        *,
        block_size: int,
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

        block_size = int(block_size)
        if block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        self._block_size = block_size

        self._forward = forward
        self._feedback = feedback
        self.in_channels = forward.in_channels
        self.out_channels = forward.out_channels

        match delay_position:
            case "forward":
                self._process_steps = self._process_forward_delay
                state_channels = self.out_channels
            case "feedback":
                self._process_steps = self._process_feedback_delay
                state_channels = self.in_channels
            case _:
                raise ValueError("delay_position value not valid")

        self.delay_position = delay_position
        self._state = RecursionState(
            np.full(state_channels, block_size, dtype=int), block_size
        )

        warnings.warn(
            f"Recursion block processing inserts {block_size} samples of delay "
            f"into the loop, on the {delay_position} path. It is not compensated "
            f"automatically: shorten the delay lines on that path by "
            f"{block_size} samples if the loop delay has to be exact.",
            UserWarning,
            stacklevel=2,
        )

    def process_block(self, block: ArrayLike) -> np.ndarray:
        """Process one block of audio.

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

            out[n : n + block_size, :] = self._process_steps(block_in, block_size)

            n += block_size

        return out

    def _process_forward_delay(
        self, block_in: ArrayLike, block_size: int
    ) -> np.ndarray:
        """``delay_position="forward"``: the state buffer holds the forward-path
        output, so the block returned is the forward output of the *previous*
        block -- the inserted delay sits after ``fF``."""
        state = self._state.get_values(block_size)

        fb_out = self._feedback.process_block(state)
        fw_out = self._forward.process_block(block_in + fb_out)

        self._state.set_values(fw_out)
        self._state.advance(block_size)

        return state

    def _process_feedback_delay(
        self, block_in: ArrayLike, block_size: int
    ) -> np.ndarray:
        """``delay_position="feedback"``: the state buffer holds the feedback-path
        output, so the current forward output is returned undelayed and the
        inserted delay sits after ``fB``."""
        state = self._state.get_values(block_size)

        fw_out = self._forward.process_block(block_in + state)
        fb_out = self._feedback.process_block(fw_out)

        self._state.set_values(fb_out)
        self._state.advance(block_size)

        return fw_out

    def reset(self) -> None:
        """Resets the state of all internal TimeOperators."""
        self._forward.reset()
        self._feedback.reset()
        self._state.reset()
