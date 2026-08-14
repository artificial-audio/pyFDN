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
from scipy.fft import irfft, next_fast_len, rfft

from pyFDN.dsp.dfilt_matrix import FIRMatrixFilter
from pyFDN.dsp.sos_filter_bank import SOSFilterBank
from pyFDN.dsp.time_varying_matrix import TimeVaryingMatrix as _DspTimeVaryingMatrix


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
    :meth:`filter`. :meth:`reset` returns the operator to its initial
    (zero) state.
    """

    in_channels: int
    out_channels: int

    @abstractmethod
    def filter(self, block: ArrayLike) -> np.ndarray:
        """Filter one block and advance internal state."""

    def reset(self) -> None:  # noqa: B027 -- intentional no-op default for stateless ops
        """Clear internal state (no-op for stateless operators)."""


class Identity(TimeOperator):
    """Stateless pass-through.
    Used for FFT/iFFT layers when converting to FLAMO and empty forward residuals."""

    def __init__(self, channels: int) -> None:
        self.in_channels = self.out_channels = int(channels)

    def filter(self, block: ArrayLike) -> np.ndarray:
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

    def filter(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Gain expects {self.in_channels} input channels")
        return x @ self.matrix.T


class Delay(TimeOperator):
    """Stateful per-channel integer feed-forward delay line, ``y[n, c] = x[n - m_c, c]``.
    It is not equivalent to FeedbackDelay class, as it does not accept direct feedback synchronization."""

    def __init__(self, delays: ArrayLike) -> None:
        d = np.asarray(delays, dtype=int).reshape(-1)
        if np.any(d < 0):
            raise ValueError("delays must be non-negative integers")
        self.in_channels = self.out_channels = d.size
        self.delays = d
        self.max_delay = int(d.max()) if self.in_channels else 0
        self._tail = np.zeros((self.max_delay, self.in_channels), dtype=float)

    def filter(self, block: ArrayLike) -> np.ndarray:
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
    Wrapper over :class:`pyFDN.dsp.SOSFilterBank`. ``sos`` has the
    canonical ``(n_sections, 6, N)`` layout."""

    def __init__(self, sos: ArrayLike) -> None:
        sos_arr = np.asarray(sos, dtype=float)
        if sos_arr.ndim != 3 or sos_arr.shape[1] != 6:
            raise ValueError("sos must have shape (n_sections, 6, N)")
        self.in_channels = self.out_channels = sos_arr.shape[2]
        self._sos = sos_arr
        self._bank = SOSFilterBank(sos_arr, self.in_channels)

    def filter(self, block: ArrayLike) -> np.ndarray:
        return self._bank.filter(_as_2d(block))

    def reset(self) -> None:
        self._bank = SOSFilterBank(self._sos, self.in_channels)


class MatrixFIR(TimeOperator):
    """Stateful matrix of FIR filters (e.g. a paraunitary scattering feedback matrix).
    Wrapper over :class:`pyFDN.dsp.FIRMatrixFilter`. ``coeffs`` has shape
    ``(n_out, n_in, n_taps)`` in the ``z^{-1}`` convention."""

    def __init__(self, coeffs: ArrayLike) -> None:
        c = np.asarray(coeffs, dtype=float)
        if c.ndim != 3:
            raise ValueError("coeffs must have shape (n_out, n_in, n_taps)")
        self.out_channels, self.in_channels, _ = c.shape
        self._coeffs = c
        self._bank = FIRMatrixFilter(c)

    def filter(self, block: ArrayLike) -> np.ndarray:
        return self._bank.filter(_as_2d(block))

    def reset(self) -> None:
        self._bank = FIRMatrixFilter(self._coeffs)


class MatrixConvolver(TimeOperator):
    """Stateful matrix of FIR filters via streaming overlap-save FFT convolution.
    The FFT counterpart of :class:`MatrixFIR`: same ``(n_out, n_in, n_taps)``
    coefficient layout, but built for **long** impulse responses (e.g. room
    RIRs) where time-domain ``lfilter`` would be prohibitively slow. State
    persists across :meth:`filter` calls, so it works both whole-signal and
    block-by-block -- including as the feedback path of a :class:`Recursion`
    (the loudspeaker -> microphone room coupling of a reverberation enhancement
    system). Output equals the linear convolution to numerical precision."""

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
        are computed only ones per :meth:``nfft`` value, at the first :meth:``filter`` call."""
        cached = self._filter_fft.get(nfft)
        if cached is None:
            c_transp = np.transpose(self._coeffs, (2, 0, 1))
            cached = rfft(c_transp, n=nfft, axis=0)  # (nfft, out, in)
            self._filter_fft[nfft] = cached
        return cached

    def filter(self, block: ArrayLike) -> np.ndarray:
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
    Adapts :class:`pyFDN.dsp.time_varying_matrix.TimeVaryingMatrix` -- which
    already rotates adjacent channel pairs by a per-sample modulated angle -- to
    the operator protocol, so it can sit on a :class:`Recursion` feedback path,
    e.g. ``Series([Gain(A), TimeVaryingMatrix(tvm)])``. This is the operator
    analogue of the ``extra_matrix`` argument of :func:`pyFDN.process_fdn`.
    Because the matrix changes every sample, the loop is genuinely time-varying
    and has no static transfer function -- there is no equivalent in FLAMO.

    Parameters
    ----------
    matrix
        A built :class:`pyFDN.dsp.time_varying_matrix.TimeVaryingMatrix` (or any
        object exposing ``N`` and a stateful ``filter((T, N)) -> (T, N)``).
    """

    def __init__(self, matrix: _DspTimeVaryingMatrix) -> None:
        self._tvm = matrix
        self.in_channels = self.out_channels = int(matrix.N)

    def filter(self, block: ArrayLike) -> np.ndarray:
        return self._tvm.filter(_as_2d(block))

    def reset(self) -> None:
        # Rewind the modulation clock without re-drawing the random modulation
        # parameters, so a reset render is reproducible.
        self._tvm.sample_index = 0
