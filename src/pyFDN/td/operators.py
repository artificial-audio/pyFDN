"""Stateful time-domain operators.

Each operator maps a block ``(num_samples, in_channels)`` to
``(num_samples, out_channels)`` and keeps whatever state it needs across
calls, so a long signal can be streamed through in consecutive blocks. The
composites (:class:`~pyFDN.td.connectors.Series`,
:class:`~pyFDN.td.connectors.Parallel`,
:class:`~pyFDN.td.connectors.Recursion`) live in
:mod:`pyFDN.td.connectors` and wire these leaves into a graph.

:class:`RecursionState` is not an operator but the delay-line buffer bank the
graph is built on: it is what :class:`~pyFDN.td.connectors.Recursion` uses to
break its feedback loop, and what :func:`pyFDN.process_dss` uses for the DSS
delay lines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import lfilter, sosfilt


def _as_2d(block: ArrayLike) -> np.ndarray:
    """Coerce a signal block to ``(num_samples, channels)`` float array."""
    x = np.asarray(block, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if x.ndim != 2:
        raise ValueError("signal block must be 1-D or 2-D")
    return x


class TimeOperator(ABC):
    """Abstract class. Parent of all classes belonging to the td-graph group.
    A stateful ``(T, in_channels) -> (T, out_channels)`` time-domain block.
    Subclasses set ``in_channels`` / ``out_channels`` and implement
    :meth:`process_block`. :meth:`reset` returns the operator to its initial
    (zero) state.
    """

    in_channels: int
    out_channels: int

    @abstractmethod
    def process_block(self, block: ArrayLike) -> np.ndarray:
        """Process one block and advance internal state."""

    def reset(self) -> None:  # noqa: B027 -- intentional no-op default for stateless ops
        """Clear internal state (no-op for stateless operators)."""

    def process_signal(
        self, signal: ArrayLike, *, squeeze: bool = False
    ) -> np.ndarray:
        """Process a whole signal in one call, from the current state.

        Convenience wrapper around :meth:`process_block` for the common case of
        rendering an operator tree offline. Operators that process in blocks
        internally (:class:`~pyFDN.td.connectors.Recursion`) do so regardless of
        how the signal is handed to them, so this gives the same result as
        streaming ``signal`` through :meth:`process_block` block by block.

        Parameters
        ----------
        signal
            Input of shape ``(num_samples,)`` or ``(num_samples, in_channels)``.
        squeeze
            Squeeze singleton output channels (default ``False``).

        Returns
        -------
        np.ndarray
            Output of shape ``(num_samples, out_channels)``.
        """
        out = self.process_block(_as_2d(signal))
        return out.squeeze() if squeeze else out


class Identity(TimeOperator):
    """Stateless pass-through.
    Used as a placeholder branch, e.g. an empty forward residual."""

    def __init__(self, channels: int) -> None:
        self.in_channels = self.out_channels = int(channels)

    def process_block(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Identity expects {self.in_channels} input channels")
        return x


class Gain(TimeOperator):
    """Stateless static gain matrix ``y = x @ M.T`` with ``M`` of shape ``(out, in)``."""

    def __init__(self, matrix: ArrayLike) -> None:
        m = np.asarray(matrix, dtype=float)
        if m.ndim == 1:
            m = m[:, np.newaxis]
        if m.ndim != 2:
            raise ValueError("gain matrix must be 1-D or 2-D")
        self.out_channels, self.in_channels = m.shape
        self.matrix = m

    def process_block(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Gain expects {self.in_channels} input channels")
        return x @ self.matrix.T


class AbsoluteValue(TimeOperator):
    """Stateless memoryless nonlinearity ``y[n, c] = |x[n, c]|``.

    The one non-linear operator in the set: it has no transfer function and no
    frequency-domain counterpart, so a graph containing it can only be rendered
    in the time domain. Useful as a rectifier inside a loop (distortion,
    envelope-style feedback) and as the simplest way to test that the block
    engine stays sample-exact when superposition no longer holds -- note that a
    non-linear operator makes the render depend on the input *level*.
    """

    def __init__(self, channels: int) -> None:
        self.in_channels = self.out_channels = int(channels)

    def process_block(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"AbsoluteValue expects {self.in_channels} input channels")
        return np.abs(x)


class Delay(TimeOperator):
    """Stateful per-channel integer feed-forward delay line, ``y[n, c] = x[n - m_c, c]``.
    It is not equivalent to :class:`RecursionState`, as it does not accept direct
    feedback synchronization."""

    def __init__(self, delays: ArrayLike) -> None:
        d = np.asarray(delays, dtype=int).reshape(-1)
        if np.any(d < 0):
            raise ValueError("delays must be non-negative integers")
        self.in_channels = self.out_channels = d.size
        self.delays = d
        self.max_delay = int(d.max()) if self.in_channels else 0
        self._tail = np.zeros((self.max_delay, self.in_channels), dtype=float)

    def process_block(self, block: ArrayLike) -> np.ndarray:
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
    """Stateful per-channel SOS filter cascade (e.g. in-loop absorption).

    One cascade of second-order sections per channel, filtered with
    :func:`scipy.signal.sosfilt`. State persists across
    :meth:`process_block` calls, so
    a long signal can be processed in consecutive blocks.

    Parameters
    ----------
    sos
        Per-channel SOS bank of shape ``(n_sections, 6, N)`` where
        ``N = num_channels``. Section rows are ``[b0, b1, b2, a0, a1, a2]``.
        This is the canonical SOS bank layout in pyFDN: it matches the FLAMO
        ``parallelSOSFilter`` input and the output of
        :func:`pyFDN.first_order_absorption`, :func:`pyFDN.one_pole_absorption`,
        and :func:`pyFDN.absorption_geq`.

    Attributes
    ----------
    sos
        The same bank in the ``(N, n_sections, 6)`` layout :func:`scipy.signal.sosfilt`
        expects, one cascade per row.

    Notes
    -----
    Pass an instance as ``post_delay`` to :func:`pyFDN.process_dss` to apply
    frequency-dependent absorption inside the feedback loop.
    :func:`pyFDN.build_to_td` constructs this class for every populated SOS hook
    in an :class:`pyFDN.FDNBuild`.
    """

    def __init__(self, sos: ArrayLike) -> None:
        sos_arr = np.asarray(sos, dtype=float)
        if sos_arr.ndim != 3 or sos_arr.shape[1] != 6:
            raise ValueError(
                f"sos must have shape (n_sections, 6, N); got {sos_arr.shape}"
            )
        self.in_channels = self.out_channels = sos_arr.shape[2]
        # Canonical (n_sections, 6, N) -> (N, n_sections, 6) for scipy sosfilt.
        self.sos = np.ascontiguousarray(sos_arr.transpose(2, 0, 1))
        self.num_sections = sos_arr.shape[0]
        self._state = np.zeros((self.in_channels, self.num_sections, 2), dtype=float)

    def process_block(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"SOSBank expects {self.in_channels} input channels")
        out = np.empty_like(x)
        for i in range(self.in_channels):
            out[:, i], self._state[i] = sosfilt(
                self.sos[i], np.ascontiguousarray(x[:, i]), zi=self._state[i]
            )
        return out

    def reset(self) -> None:
        self._state[:] = 0.0


class MatrixFIR(TimeOperator):
    """Stateful matrix of FIR filters (e.g. a paraunitary scattering feedback matrix).

    Every matrix entry is an FIR filter run with :func:`scipy.signal.lfilter`;
    state persists across :meth:`process_block` calls. For long impulse
    responses use :class:`MatrixConvolver` instead, which computes the same
    convolution by FFT.

    Parameters
    ----------
    coeffs
        FIR coefficients of shape ``(n_out, n_in, n_taps)`` in the ``z^{-1}``
        convention (``coeffs[i, j, k]`` is the tap of ``z^{-k}`` from input ``j``
        to output ``i``).

    Notes
    -----
    :func:`pyFDN.process_dss` constructs this filter automatically when its
    feedback matrix ``A`` has shape ``(n_out, n_in, n_taps)``.
    """

    def __init__(self, coeffs: ArrayLike) -> None:
        c = np.asarray(coeffs, dtype=float)
        if c.ndim != 3:
            raise ValueError("coeffs must have shape (n_out, n_in, n_taps)")
        self.out_channels, self.in_channels, self.num_taps = c.shape
        self.coeffs = c
        self._state = np.zeros(
            (self.out_channels, self.in_channels, max(self.num_taps - 1, 1))
        )

    def process_block(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"MatrixFIR expects {self.in_channels} input channels")
        if self.num_taps == 1:
            return x @ self.coeffs[:, :, 0].T
        out = np.zeros((x.shape[0], self.out_channels))
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                y, self._state[i, j] = lfilter(
                    self.coeffs[i, j], [1.0], x[:, j], zi=self._state[i, j]
                )
                out[:, i] += y
        return out

    def reset(self) -> None:
        self._state[:] = 0.0


class MatrixConvolver(TimeOperator):
    """Stateful matrix of FIR filters via streaming overlap-save FFT convolution.
    The FFT counterpart of :class:`MatrixFIR`: same ``(n_out, n_in, n_taps)``
    coefficient layout, but built for **long** impulse responses (e.g. room
    RIRs) where time-domain ``lfilter`` would be prohibitively slow. State
    persists across :meth:`process_block` calls, so it works both whole-signal
    and block-by-block -- including as the feedback path of a
    :class:`~pyFDN.td.connectors.Recursion` (the loudspeaker -> microphone room
    coupling of a reverberation enhancement system). Output equals the linear
    convolution to numerical precision."""

    def __init__(self, coeffs: ArrayLike) -> None:
        c = np.asarray(coeffs, dtype=float)
        if c.ndim != 3:
            raise ValueError("coeffs must have shape (n_out, n_in, n_taps)")
        self.out_channels, self.in_channels, self._taps = c.shape
        self._coeffs = c
        self._overlap = max(self._taps - 1, 0)
        self._hist = np.zeros((self._overlap, self.in_channels), dtype=float)
        self._filter_fft: dict[int, np.ndarray] = {}

    def _filters(self, nfft: int) -> np.ndarray:
        """Convenience cache: filters' frequency-domain representation
        is computed only once per ``nfft`` value, on first use."""
        cached = self._filter_fft.get(nfft)
        if cached is None:
            c_transp = np.transpose(self._coeffs, (2, 0, 1))
            cached = rfft(c_transp, n=nfft, axis=0)  # (nfft, out, in)
            self._filter_fft[nfft] = cached
        return cached

    def process_block(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"MatrixConvolver expects {self.in_channels} channels")
        num_samples = x.shape[0]
        overlap = self._overlap
        ext = np.concatenate([self._hist, x], axis=0)  # (overlap + T, in)
        if overlap:
            self._hist = ext[-overlap:].copy()
        nfft = next_fast_len(ext.shape[0] + self._taps - 1)
        spectrum = rfft(ext, n=nfft, axis=0)  # (nfft, in)
        mixed = np.einsum("foi,fi->fo", self._filters(nfft), spectrum)  # (nfft, out)
        full = irfft(mixed, n=nfft, axis=0)  # (nfft, out)
        return full[overlap : overlap + num_samples]  # (T, out)

    def reset(self) -> None:
        self._hist[:] = 0.0


class TimeVaryingMatrix(TimeOperator):
    """Stateful sinusoidally modulated orthogonal mixing matrix (time-varying feedback).

    Each adjacent channel pair is rotated by a sinusoidally modulated angle, so
    the operator is orthogonal at every sample but never constant. Sitting on a
    :class:`~pyFDN.td.connectors.Recursion` feedback path, e.g.
    ``Series([Gain(A), TimeVaryingMatrix(N, 1.5, 0.35, fs, 0.1)])``, it makes the
    loop genuinely time-varying -- there is no static transfer function. This is
    the operator form of the ``post_matrix`` argument of
    :func:`pyFDN.process_dss`.

    Translation of the MATLAB implementation ``timeVaryingMatrix.m`` from
    fdnToolbox. Original MATLAB code: (c) Sebastian Jiro Schlecht, 2019.
    Python translation: Alma Hova, 2026.

    Parameters
    ----------
    N : int
        Number of channels (the matrix is N x N). Must be a positive even integer.
    cycles_per_second : float
        Frequency of the time variation in Hz (controls oscillation speed).
    amplitude : float
        Maximum angle deflection in radians (strength of modulation).
    fs : float
        Sampling rate in Hz.
    spread : float
        Randomness factor (controls how differently each eigenmode behaves).

    Attributes
    ----------
    num_pairs : int
        Number of eigenmode pairs (``N // 2``), i.e. independent 2-D rotation planes.
    phase, frequency, angle_amplitude : ndarray
        Per-pair modulation parameters, drawn from the global NumPy RNG at
        construction. Seed ``np.random.seed`` beforehand for a reproducible
        modulation.
    sample_index : int
        Current sample index; the modulation clock.
    """

    def __init__(
        self,
        N: int,
        cycles_per_second: float,
        amplitude: float,
        fs: float,
        spread: float,
    ) -> None:
        # Enforce N to be a positive, even integer.
        N = int(N)
        if N <= 0:
            raise ValueError("N must be a positive integer")
        if N % 2 != 0:
            raise ValueError("N must be even")

        self.N = N
        self.in_channels = self.out_channels = N
        self.cycles_per_second = cycles_per_second
        self.amplitude = amplitude
        self.fs = fs
        self.spread = spread

        # Number of independent 2-D rotation planes (conjugate eigenvalue pairs)
        self.num_pairs = N // 2

        # Random initial phase between 0 and 2*pi for each 2-D plane
        self.phase = 2 * np.pi * np.random.rand(self.num_pairs)

        # Unique modulation frequency for each pair, using the spread factor
        self.frequency = self.cycles_per_second * (
            1 + self.spread * (2 * np.random.rand(self.num_pairs) - 1)
        )

        # Modulation amplitude
        self.angle_amplitude = self.amplitude * (
            1 + self.spread * (2 * np.random.rand(self.num_pairs) - 1)
        )

        # Global time tracker index, initialized to 0
        self.sample_index = 0

    def process_block(self, block: ArrayLike) -> np.ndarray:
        """Apply the time-varying orthogonal transformation to one block.

        The operation is equivalent to constructing the block-diagonal rotation
        matrix from ``rotation_matrix_from_angles`` at every sample, but applies
        the 2-D rotations directly to the whole input block.
        """
        x = _as_2d(block)
        if x.shape[1] != self.N:
            raise ValueError(f"TimeVaryingMatrix expects {self.N} input channels")

        length = x.shape[0]
        sample_indices = self.sample_index + np.arange(length)
        time = sample_indices[:, np.newaxis] / self.fs
        angles = self.angle_amplitude * np.sin(
            2 * np.pi * self.frequency * time + self.phase
        )
        cos = np.cos(angles)
        sin = np.sin(angles)

        x_pairs = x.reshape(length, self.num_pairs, 2)
        out = np.empty(x.shape, dtype=np.result_type(x, self.angle_amplitude, float))
        out_pairs = out.reshape(length, self.num_pairs, 2)
        out_pairs[..., 0] = cos * x_pairs[..., 0] - sin * x_pairs[..., 1]
        out_pairs[..., 1] = sin * x_pairs[..., 0] + cos * x_pairs[..., 1]

        self.sample_index += length

        return out

    def reset(self) -> None:
        # Rewind the modulation clock without re-drawing the random modulation
        # parameters, so a reset render is reproducible.
        self.sample_index = 0


class RecursionState:
    """Vectorised bank of block-addressable delay lines.

    The state store a feedback loop is built on: :meth:`get_values` reads the
    next ``block_size`` output samples of every line, :meth:`set_values` writes
    the samples going back in, and :meth:`advance` moves the read/write
    pointers. Reading before writing is what breaks the algebraic loop, so a
    whole block can be computed at once.

    Used by :class:`~pyFDN.td.connectors.Recursion` (all lines the length of one
    processing block) and by :func:`pyFDN.process_dss` (one line per FDN
    delay).

    Parameters
    ----------
    delays
        Length of each delay line in samples, shape ``(num_delays,)``. Must be
        positive.
    max_block_size
        Largest block :meth:`get_values` will be asked for. Must not exceed the
        shortest delay, otherwise a block would wrap around its own line.
    """

    def __init__(self, delays: ArrayLike, max_block_size: int) -> None:
        delays_arr = np.asarray(delays, dtype=int).reshape(-1)
        if delays_arr.ndim != 1:
            raise ValueError("Delays must be a 1-D array")
        if np.any(delays_arr <= 0):
            raise ValueError("Delays must be positive integers")

        self.delays = delays_arr
        self.num_delays = delays_arr.size
        self.max_block_size = int(max_block_size)
        self.max_delay = int(np.max(delays_arr))
        self.buffer = np.zeros((self.num_delays, self.max_delay), dtype=float)
        self.pointers = np.zeros(self.num_delays, dtype=int)
        self._last_indices: np.ndarray | None = None

    def get_values(self, block_size: int) -> np.ndarray:
        """Read the next ``block_size`` samples out of every delay line."""
        if block_size > self.max_block_size:
            raise ValueError("Block size exceeds configured maximum")
        offsets = (
            self.pointers[:, None] + np.arange(block_size)[None, :]
        ) % self.delays[:, None]
        self._last_indices = offsets
        gathered = self.buffer[np.arange(self.num_delays)[:, None], offsets]
        return gathered.T

    def set_values(self, block: ArrayLike) -> None:
        """Write a ``(block_size, num_delays)`` block into the slots just read."""
        if self._last_indices is None:
            raise RuntimeError("get_values must be called before set_values")
        block_arr = np.asarray(block, dtype=float)
        if block_arr.shape != (self._last_indices.shape[1], self.num_delays):
            raise ValueError("Block shape mismatch when writing delay values")
        self.buffer[np.arange(self.num_delays)[:, None], self._last_indices] = (
            block_arr.T
        )

    def advance(self, block_size: int) -> None:
        """Move the read/write pointers on by one block."""
        self.pointers = (self.pointers + block_size) % self.delays
        self._last_indices = None

    def reset(self) -> None:
        """Zero the buffers and rewind the pointers."""
        self.buffer[:] = 0.0
        self.pointers[:] = 0
        self._last_indices = None
