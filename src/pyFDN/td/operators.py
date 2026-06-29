"""Stateful time-domain operators mirroring FLAMO's structural node types.

Each operator maps a block ``(num_samples, in_channels)`` to
``(num_samples, out_channels)`` and keeps whatever state it needs across
calls, so a long signal can be streamed through in consecutive blocks. The
composites (:class:`Series`, :class:`Parallel`, :class:`Recursion`) mirror the
FLAMO ``Series`` / ``Parallel`` / ``Recursion`` modules; the leaves wrap the
existing pyFDN DSP components (:class:`pyFDN.dsp.FeedbackDelay`,
:class:`pyFDN.dsp.SOSFilterBank`, :class:`pyFDN.dsp.FIRMatrixFilter`).

The only subtle one is :class:`Recursion`: a feedback loop cannot be evaluated
sample-by-sample without an algebraic loop, so it is processed in blocks no
larger than the shortest loop delay, exactly as :func:`pyFDN.process_fdn` does.
See that class for details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.dsp.dfilt_matrix import FIRMatrixFilter
from pyFDN.dsp.feedback_delay import FeedbackDelay
from pyFDN.dsp.sos_filter_bank import SOSFilterBank
from pyFDN.dsp.time_varying_matrix import TimeVaryingMatrix as _DspTimeVaryingMatrix

# Cap on the recursion block size (also the FFT-free analogue of process_fdn's
# 2**12 cap); the true block size is min(this, shortest loop delay).
_MAX_BLOCK = 1 << 12


def _as_2d(block: ArrayLike) -> np.ndarray:
    """Coerce a signal block to ``(num_samples, channels)`` float array."""
    x = np.asarray(block, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if x.ndim != 2:
        raise ValueError("signal block must be 1-D or 2-D")
    return x


class TimeOperator(ABC):
    """A stateful ``(T, in_channels) -> (T, out_channels)`` time-domain block.

    Subclasses set ``in_channels`` / ``out_channels`` and implement
    :meth:`process`. :meth:`reset` returns the operator to its initial
    (zero) state.
    """

    in_channels: int
    out_channels: int

    @abstractmethod
    def process(self, block: ArrayLike) -> np.ndarray:
        """Process one block and advance internal state."""

    def reset(self) -> None:  # noqa: B027 -- intentional no-op default for stateless ops
        """Clear internal state (no-op for stateless operators)."""


class Identity(TimeOperator):
    """Pass-through (used for FFT/iFFT layers and empty forward residuals)."""

    def __init__(self, channels: int) -> None:
        self.in_channels = self.out_channels = int(channels)

    def process(self, block: ArrayLike) -> np.ndarray:
        return _as_2d(block)


class Gain(TimeOperator):
    """Static gain matrix ``y = x @ M.T`` with ``M`` of shape ``(out, in)``.

    Covers input/output/direct gains (``B``, ``C``, ``D``) and a constant
    feedback matrix ``A``; a trainable FLAMO ``Matrix`` realizes to a constant
    matrix and is compiled to this too.
    """

    def __init__(self, matrix: ArrayLike) -> None:
        m = np.asarray(matrix, dtype=float)
        if m.ndim == 1:
            m = m[:, np.newaxis]
        if m.ndim != 2:
            raise ValueError("gain matrix must be 1-D or 2-D")
        self.matrix = m
        self.out_channels, self.in_channels = m.shape

    def process(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Gain expects {self.in_channels} input channels")
        return x @ self.matrix.T


class Delay(TimeOperator):
    """Per-channel integer feed-forward delay line, ``y[n, c] = x[n - m_c, c]``.

    Stateful across blocks. Inside a :class:`Recursion` the loop delay is
    instead realized by a :class:`pyFDN.dsp.FeedbackDelay`; this class is for
    delays that sit on a plain (non-recursive) path. The loop delays are read
    off :attr:`delays` by :class:`Recursion`.
    """

    def __init__(self, delays: ArrayLike) -> None:
        d = np.asarray(delays, dtype=int).reshape(-1)
        if np.any(d < 0):
            raise ValueError("delays must be non-negative integers")
        self.delays = d
        self.in_channels = self.out_channels = d.size
        self.max_delay = int(d.max()) if d.size else 0
        self._tail = np.zeros((self.max_delay, d.size), dtype=float)

    def process(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Delay expects {self.in_channels} input channels")
        if self.max_delay == 0:
            return x.copy()
        buf = np.concatenate([self._tail, x], axis=0)
        out = np.empty_like(x)
        for c, m in enumerate(self.delays):
            lo = self.max_delay - int(m)
            out[:, c] = buf[lo : lo + x.shape[0], c]
        self._tail = buf[-self.max_delay :].copy()
        return out

    def reset(self) -> None:
        self._tail[:] = 0.0


class SOSBank(TimeOperator):
    """Per-channel SOS filter cascade (e.g. in-loop absorption).

    Thin wrapper over :class:`pyFDN.dsp.SOSFilterBank`; ``sos`` has the
    canonical ``(n_sections, 6, N)`` layout.
    """

    def __init__(self, sos: ArrayLike) -> None:
        sos_arr = np.asarray(sos, dtype=float)
        if sos_arr.ndim != 3 or sos_arr.shape[1] != 6:
            raise ValueError("sos must have shape (n_sections, 6, N)")
        self._sos = sos_arr
        self.in_channels = self.out_channels = sos_arr.shape[2]
        self._bank = SOSFilterBank(sos_arr, self.in_channels)

    def process(self, block: ArrayLike) -> np.ndarray:
        return self._bank.filter(_as_2d(block))

    def reset(self) -> None:
        self._bank = SOSFilterBank(self._sos, self.in_channels)


class MatrixFIR(TimeOperator):
    """Matrix of FIR filters (e.g. a paraunitary scattering feedback matrix).

    Thin wrapper over :class:`pyFDN.dsp.FIRMatrixFilter`; ``coeffs`` has shape
    ``(n_out, n_in, n_taps)`` in the ``z^{-1}`` convention.
    """

    def __init__(self, coeffs: ArrayLike) -> None:
        c = np.asarray(coeffs, dtype=float)
        if c.ndim != 3:
            raise ValueError("coeffs must have shape (n_out, n_in, n_taps)")
        self._coeffs = c
        self.out_channels, self.in_channels, _ = c.shape
        self._filt = FIRMatrixFilter(c)

    def process(self, block: ArrayLike) -> np.ndarray:
        return self._filt.filter(_as_2d(block))

    def reset(self) -> None:
        self._filt = FIRMatrixFilter(self._coeffs)


class TimeVaryingMatrix(TimeOperator):
    """Sinusoidally modulated orthogonal mixing matrix (time-varying feedback).

    Adapts :class:`pyFDN.dsp.time_varying_matrix.TimeVaryingMatrix` -- which
    already rotates adjacent channel pairs by a per-sample modulated angle -- to
    the operator protocol, so it can sit on a :class:`Recursion` feedback path,
    e.g. ``Series([Gain(A), TimeVaryingMatrix(tvm)])``. This is the operator
    analogue of the ``extra_matrix`` argument of :func:`pyFDN.process_fdn`.

    Because the matrix changes every sample, the loop is genuinely time-varying
    and has no static transfer function -- a render only the time-domain engine
    can produce (FLAMO's frequency-domain render cannot).

    Parameters
    ----------
    matrix
        A built :class:`pyFDN.dsp.time_varying_matrix.TimeVaryingMatrix` (or any
        object exposing ``N`` and a stateful ``filter((T, N)) -> (T, N)``).
    """

    def __init__(self, matrix: _DspTimeVaryingMatrix) -> None:
        self._tvm = matrix
        self.in_channels = self.out_channels = int(matrix.N)

    def process(self, block: ArrayLike) -> np.ndarray:
        return self._tvm.filter(_as_2d(block))

    def reset(self) -> None:
        # Rewind the modulation clock without re-drawing the random modulation
        # parameters, so a reset render is reproducible.
        self._tvm.sample_index = 0


class Series(TimeOperator):
    """Chain operators left to right (FLAMO ``Series``)."""

    ops: list[TimeOperator]

    def __init__(self, ops: list[TimeOperator]) -> None:
        flat: list[TimeOperator] = []
        for op in ops:
            if isinstance(op, Series):
                flat.extend(op.ops)
            else:
                flat.append(op)
        if not flat:
            raise ValueError("Series needs at least one operator")
        self.ops = flat
        self.in_channels = flat[0].in_channels
        self.out_channels = flat[-1].out_channels

    def process(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        for op in self.ops:
            x = op.process(x)
        return x

    def reset(self) -> None:
        for op in self.ops:
            op.reset()


class Parallel(TimeOperator):
    """Feed the same input to every branch and combine the outputs.

    Mirrors FLAMO ``Parallel``: ``sum_output=True`` sums the branch outputs
    (requires equal ``out_channels``), otherwise channels are concatenated.
    """

    def __init__(
        self, branches: list[TimeOperator], *, sum_output: bool = True
    ) -> None:
        if not branches:
            raise ValueError("Parallel needs at least one branch")
        self.branches = branches
        self.sum_output = sum_output
        self.in_channels = branches[0].in_channels
        if sum_output:
            outs = {b.out_channels for b in branches}
            if len(outs) != 1:
                raise ValueError("summed Parallel branches must share out_channels")
            self.out_channels = branches[0].out_channels
        else:
            self.out_channels = sum(b.out_channels for b in branches)

    def process(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        outs = [b.process(x) for b in self.branches]
        if self.sum_output:
            acc = outs[0].copy()
            for o in outs[1:]:
                acc += o
            return acc
        return np.concatenate(outs, axis=1)

    def reset(self) -> None:
        for b in self.branches:
            b.reset()


def _split_leading_delay(forward: TimeOperator) -> tuple[np.ndarray, TimeOperator]:
    """Peel the loop delay off the front of a recursion's forward path.

    Returns ``(delay_lengths, rest)`` where ``rest`` is everything in the
    forward path after the delay (an :class:`Identity` if the delay is the
    whole path). The forward path must begin with a :class:`Delay`; this is the
    element that breaks the algebraic loop and sets the block size.
    """
    ops = forward.ops if isinstance(forward, Series) else [forward]
    delay = ops[0] if ops else None
    if not isinstance(delay, Delay):
        raise ValueError(
            "Recursion forward path must begin with a Delay (the loop delay)"
        )
    rest_ops = ops[1:]
    if not rest_ops:
        rest: TimeOperator = Identity(delay.out_channels)
    elif len(rest_ops) == 1:
        rest = rest_ops[0]
    else:
        rest = Series(rest_ops)
    return delay.delays, rest


class Recursion(TimeOperator):
    r"""Closed feedback loop ``y = fF(x + fB(y))`` (FLAMO ``Recursion``).

    A feedback loop has an algebraic dependency unless the loop contains a
    delay, so it is processed in blocks no larger than the shortest loop delay.
    The loop delay (the leading :class:`Delay` of the forward path ``fF``) is
    realized by a :class:`pyFDN.dsp.FeedbackDelay`, whose read-before-write
    circular buffer makes each delay-line read return the value written one full
    delay earlier. Concretely, per block (``rest`` = ``fF`` after the delay,
    ``fB`` = feedback):

    1. ``d   = delay_bank.get_values(block)``   -- the delayed loop signal
    2. ``y   = rest(d)``                          -- finish the forward path
    3. ``fb  = fB(y)``                            -- feedback path
    4. ``delay_bank.set_values(x + fb)``          -- write next loop input

    Because the full per-line delay lives in the circular buffer and the block
    size never exceeds the shortest delay, the result lines up sample-for-sample
    with the FLAMO frequency-domain render -- the same scheme, and the same
    alignment guarantee, as :func:`pyFDN.process_fdn`.
    """

    def __init__(self, forward: TimeOperator, feedback: TimeOperator) -> None:
        delays, rest = _split_leading_delay(forward)
        self._delays = np.asarray(delays, dtype=int).reshape(-1)
        if np.any(self._delays <= 0):
            raise ValueError("loop delays must be positive integers")
        self._rest = rest
        self._feedback = feedback
        n = self._delays.size
        if rest.in_channels != n or rest.out_channels != n:
            raise ValueError("forward residual must be N-by-N in channels")
        if feedback.in_channels != n or feedback.out_channels != n:
            raise ValueError("feedback path must be N-by-N in channels")
        self.in_channels = self.out_channels = n
        self._block = min(_MAX_BLOCK, int(self._delays.min()))
        self._bank = FeedbackDelay(self._delays, self._block)

    def process(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Recursion expects {self.in_channels} input channels")
        num_samples = x.shape[0]
        out = np.empty((num_samples, self.out_channels), dtype=float)
        start = 0
        while start < num_samples:
            bs = min(self._block, num_samples - start)
            xb = x[start : start + bs]
            d = self._bank.get_values(bs)
            y = self._rest.process(d)
            out[start : start + bs] = y
            fb = self._feedback.process(y)
            self._bank.set_values(xb + fb)
            self._bank.advance(bs)
            start += bs
        return out

    def reset(self) -> None:
        self._bank = FeedbackDelay(self._delays, self._block)
        self._rest.reset()
        self._feedback.reset()
