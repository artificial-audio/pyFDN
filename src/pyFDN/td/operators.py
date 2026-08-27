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
break its feedback loop, and what :func:`pyFDN.process_fdn` uses for the FDN
delay lines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

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

    def process(self, signal: ArrayLike, *, squeeze: bool = False) -> np.ndarray:
        """Filter a whole signal in one call, from the current state.

        Convenience wrapper around :meth:`filter` for the common case of
        rendering an operator tree offline. Operators that process in blocks
        internally (:class:`~pyFDN.td.connectors.Recursion`) do so regardless of
        how the signal is handed to them, so this gives the same result as
        streaming ``signal`` through :meth:`filter` block by block.

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
        out = self.filter(_as_2d(signal))
        return out.squeeze() if squeeze else out


class Identity(TimeOperator):
    """Stateless pass-through.
    Used as a placeholder branch, e.g. an empty forward residual."""

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

    def filter(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"AbsoluteValue expects {self.in_channels} input channels")
        return np.abs(x)


class DCBlocker(TimeOperator):
    """Stateful per-channel first-order DC blocker with optional slow energy
    compensation, ``y[n, c] = x[n, c] - x[n-1, c] + R * y[n-1, c]``.

    The differencing term removes the DC offset a nonlinearity such as
    :class:`ControllableFullWaveRect` would otherwise inject into a feedback
    loop, at the cost of also attenuating content near DC. When
    ``correct_loss`` is enabled, a slowly-varying gain tracks the ratio of
    input to output power through two exponential envelope followers -- one
    over the signal power, one smoothing the resulting gain -- and rescales
    the output to compensate for that loss.

    Parameters
    ----------
    channels : int
        Number of channels processed independently.
    R : float
        Pole location of the blocker, ``0 < R < 1``. Closer to 1 pushes the
        cutoff frequency down and preserves more low-frequency content.
    correct_loss : bool
        If ``True``, apply the energy-compensation gain described above.
    fs : float
        Sampling rate in Hz, used to convert the time constants below into
        per-sample smoothing coefficients.
    env_tau_s : float
        Time constant of the power envelope followers, in seconds.
    gain_tau_s : float
        Time constant of the gain smoothing, in seconds.
    max_gain : float
        Ceiling on the compensation gain. A signal the blocker removes almost
        entirely -- anything close to pure DC -- has an output power near zero,
        so the uncapped ratio grows without bound: it would undo the blocking
        it is compensating for and run away inside a feedback loop.
    """

    def __init__(
        self,
        channels: int,
        R: float = 0.995,
        correct_loss: bool = False,
        fs: float = 48000.0,
        env_tau_s: float = 0.05,  # RMS tracking time constant (50 ms)
        gain_tau_s: float = 0.02,  # gain smoothing time constant (20 ms)
        max_gain: float = 4.0,  # +12 dB
    ) -> None:
        self.in_channels = self.out_channels = int(channels)
        self.R = float(R)
        self.correct_loss = bool(correct_loss)
        self.max_gain = float(max_gain)

        self.prev_x = np.zeros(self.in_channels)
        self.prev_y = np.zeros(self.in_channels)

        # Envelope follower states (power domain)
        self.eps = 1e-12
        self.in_pow = np.full(self.in_channels, 1e-12)
        self.out_pow = np.full(self.in_channels, 1e-12)
        self.gain = np.ones(self.in_channels)

        self.alpha_env = float(np.exp(-1.0 / (fs * env_tau_s)))
        self.alpha_gain = float(np.exp(-1.0 / (fs * gain_tau_s)))

    def _one_pole(
        self, signal: np.ndarray, alpha: float, state: np.ndarray
    ) -> np.ndarray:
        """Run ``s[n] = alpha * s[n-1] + (1 - alpha) * signal[n]`` over a block.

        ``state`` holds ``s[-1]`` on entry and is updated in place to ``s[T-1]``.
        """
        out, _ = lfilter(
            [1.0 - alpha], [1.0, -alpha], signal, axis=0, zi=alpha * state[np.newaxis]
        )
        state[:] = out[-1]
        return np.asarray(out)

    def filter(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"DCBlocker expects {self.in_channels} input channels")
        if x.shape[0] == 0:
            return x

        # y[n] = x[n] - x[n-1] + R * y[n-1]. The transposed direct form II state
        # of that recursion is -x[-1] + R * y[-1], which is what carries the
        # blocker across block boundaries.
        y, _ = lfilter(
            [1.0, -1.0],
            [1.0, -self.R],
            x,
            axis=0,
            zi=(-self.prev_x + self.R * self.prev_y)[np.newaxis],
        )
        self.prev_x[:] = x[-1]
        self.prev_y[:] = y[-1]  # pre-compensation, i.e. the actual filter state

        if not self.correct_loss:
            return np.asarray(y)

        in_pow = self._one_pole(x * x, self.alpha_env, self.in_pow)
        out_pow = self._one_pole(y * y, self.alpha_env, self.out_pow)
        target_gain = np.sqrt((in_pow + self.eps) / (out_pow + self.eps))
        # Cap before smoothing so the smoothed gain inherits the same ceiling.
        np.minimum(target_gain, self.max_gain, out=target_gain)
        return np.asarray(y * self._one_pole(target_gain, self.alpha_gain, self.gain))

    def reset(self) -> None:
        self.prev_x[:] = 0.0
        self.prev_y[:] = 0.0
        self.in_pow[:] = 1e-12
        self.out_pow[:] = 1e-12
        self.gain[:] = 1.0


class ControllableFullWaveRect(TimeOperator):
    """Stateful, controllable, memoryless nonlinearity,
    ``y[n, c] = g_cfwr * ((1 - alpha) * x[n, c] + alpha * abs(x[n, c]))``,
    applied to ``active_channels`` only; the rest pass through unchanged.

    At ``alpha = 0`` the nonlinearity drops out and only the DC blocker below
    is left, at ``alpha = 1`` it is a full-wave rectifier;
    ``g_cfwr = sqrt(2 - 2 * abs(alpha - 0.5))`` keeps the output
    power roughly constant across ``alpha``. ``abs(x)`` here is not the plain
    absolute value but its first-order antiderivative-antialiasing
    approximation (Parker et al. 2016), which reduces the aliasing that
    rectification  would otherwise fold back from above Nyquist.
    The result is passed through an internal :class:`DCBlocker` with energy
    compensation, since rectification also injects a DC offset that would
    otherwise accumulate in a feedback loop.
    """

    def __init__(self, channels: int, alpha: float, active_channels: ArrayLike) -> None:
        self.in_channels = self.out_channels = int(channels)
        self.state = np.zeros((1, self.in_channels), dtype=float)
        self.dc_blocker = DCBlocker(self.in_channels, R=0.995, correct_loss=True)
        self.alpha = float(alpha)
        self.g_cfwr = np.sqrt(2 - 2 * abs(self.alpha - 0.5))
        self._mask = np.zeros(self.in_channels, dtype=bool)
        self._mask[np.asarray(active_channels)] = True

    def anti_dev(self, x: np.ndarray) -> np.ndarray:
        y = 0.5 * x * np.abs(x)
        return y

    def abs(self, x: np.ndarray) -> np.ndarray:
        # x_prev[n] = x[n - 1], carrying the last sample of the previous block
        x_prev = np.concatenate([self.state, x[:-1]], axis=0)
        den = x - x_prev
        num = self.anti_dev(x) - self.anti_dev(x_prev)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(np.abs(den) <= 1e-8, np.abs(x + x_prev) / 2, num / den)
        self.state = x[-1:, :].copy()
        return y

    def filter(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"ControllableFullWaveRect expects {self.in_channels} input channels"
            )
        y = self.g_cfwr * ((1 - self.alpha) * x + self.alpha * self.abs(x))
        y = self.dc_blocker.filter(y)
        out = x.copy()
        out[:, self._mask] = y[:, self._mask]
        return out

    def reset(self) -> None:
        self.state[:] = 0.0
        self.dc_blocker.reset()


class SDFD(TimeOperator):
    """Stateful, controllable Signal-Dependent Fractional Delay,
    ``y[n, c] = (1 - d) n[n-1, c] + d n[n, c] + d p[n-2, c] + (1 - d) p[n-1, c]``,
    applied to ``active_channels`` only; the rest pass through unchanged.

    ``p = max(x, 0)`` and ``n = min(x, 0)`` are the positive and negative
    half-wave rectified branches of the input, each delayed by a different,
    ``d``-dependent amount: the positive branch by roughly ``1 + d`` samples,
    the negative one by roughly ``1 - d``. Recombining the two smears the
    signal's zero crossings without reshaping the rest of the waveform, which
    reads as a soft, amplitude-dependent distortion rather than a hard clip.
    """

    def __init__(self, channels: int, d: float, active_channels: ArrayLike) -> None:
        self.in_channels = self.out_channels = int(channels)
        self.d = float(d)
        self._mask = np.zeros(self.in_channels, dtype=bool)
        self._mask[np.asarray(active_channels)] = True
        # p_state[0] = p[n-2], p_state[1] = p[n-1]
        self.p_state = np.zeros((2, self.in_channels), dtype=float)
        self.n_state = np.zeros(self.in_channels, dtype=float)

    def filter(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"SDFD expects {self.in_channels} input channels")

        p = np.maximum(x, 0.0)
        n = np.minimum(x, 0.0)

        p_ext = np.concatenate([self.p_state, p], axis=0)  # (T + 2, C)
        n_ext = np.concatenate([self.n_state[np.newaxis], n], axis=0)  # (T + 1, C)

        p1 = p_ext[1:-1]
        p2 = p_ext[:-2]
        n1 = n_ext[:-1]

        d = self.d
        y = (1 - d) * n1 + d * n + d * p2 + (1 - d) * p1

        self.p_state = p_ext[-2:].copy()
        self.n_state = n_ext[-1].copy()

        out = x.copy()
        out[:, self._mask] = y[:, self._mask]
        return out

    def reset(self) -> None:
        self.p_state[:] = 0.0
        self.n_state[:] = 0.0


class RingModulator(TimeOperator):
    """Stateful, controllable ring modulator,
    ``y[n, c] = mod_amp * x[n, c] * sin(2 * pi * mod_freq * n / fs)``, applied
    to ``active_channels`` only; the rest pass through unchanged.

    Multiplying by a sine shifts the spectrum of the active channels by
    ``+-mod_freq`` rather than adding harmonics on top of it, so the effect
    reads as tremolo at low ``mod_freq`` and as an inharmonic, bell-like
    retuning once ``mod_freq`` enters the audible range. A unit-amplitude
    sine carries only half the power of the signal it multiplies, so
    ``mod_amp = sqrt(2)`` keeps the operator energy-preserving. The
    modulation phase runs continuously across :meth:`filter` calls, tracked
    by ``sample_index``.
    """

    def __init__(
        self,
        channels: int,
        mod_freq: float,
        mod_amp: float,
        fs: float,
        active_channels: ArrayLike,
    ) -> None:
        self.in_channels = self.out_channels = int(channels)
        self.mod_freq = float(mod_freq)
        self.mod_amp = float(mod_amp)
        self.fs = float(fs)
        self._mask = np.zeros(self.in_channels, dtype=bool)
        self._mask[np.asarray(active_channels)] = True
        self.sample_index = 0

    def filter(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"RingModulator expects {self.in_channels} input channels")

        length = x.shape[0]
        n = self.sample_index + np.arange(length)
        mod = self.mod_amp * np.sin(2 * np.pi * self.mod_freq * n / self.fs)
        self.sample_index += length

        out = x.copy()
        out[:, self._mask] = x[:, self._mask] * mod[:, np.newaxis]
        return out

    def reset(self) -> None:
        self.sample_index = 0


class PitchShift(TimeOperator):
    """Stateful, controllable dual-read-head pitch shifter.

    Writes into one circular buffer per channel and reads it back with two
    read heads spaced half a window apart, each sliding at a rate set by
    ``transpose_cents`` and crossfaded with a complementary sine window so
    the head that is about to wrap is always faded out. Only
    ``active_channels`` are shifted; the rest pass through unchanged.
    """

    def __init__(
        self,
        channels: int,
        max_delay_samps: int,
        window_size: int,
        transpose_cents: float,
        fs: float,
        active_channels: ArrayLike,
        # 3 is the smallest delay that keeps the cubic interpolator, which
        # reaches two samples forward, off the sample just written.
        min_delay_samps: int = 3,
    ) -> None:
        if max_delay_samps <= window_size + min_delay_samps:
            raise ValueError("max_delay_samps must be > window_size + min_delay_samps")

        self.in_channels = self.out_channels = int(channels)
        self.max_delay = int(max_delay_samps)
        self.window_size = int(window_size)
        self.min_delay = int(min_delay_samps)
        self.fs = float(fs)
        self._mask = np.zeros(self.in_channels, dtype=bool)
        self._mask[np.asarray(active_channels)] = True
        self.dc_blocker = DCBlocker(self.in_channels, R=0.995, correct_loss=True)

        self.buffer = np.zeros((self.max_delay, self.in_channels), dtype=float)
        self.write_ptr = 0
        self.phase_1 = 0.0
        self.phase_2 = 0.5  # 180 degrees offset

        self.set_transpose_cents(transpose_cents)

    def set_transpose_cents(self, cents: float) -> None:
        self.transpose_cents = float(cents)
        self.pitch_ratio = 2.0 ** (self.transpose_cents / 1200.0)
        # Delay slope: dD/dn = 1 - ratio, with D = min_delay + phase * window
        self.phase_inc = (1.0 - self.pitch_ratio) / float(self.window_size)

    def _read_interpolated(self, ptr: float) -> np.ndarray:
        """Cubic interpolated read from the buffer, one value per channel."""
        ptr = ptr % self.max_delay
        i = int(ptr)
        f = ptr - i

        y0 = self.buffer[(i - 1) % self.max_delay]
        y1 = self.buffer[i]
        y2 = self.buffer[(i + 1) % self.max_delay]
        y3 = self.buffer[(i + 2) % self.max_delay]

        a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
        b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
        c = -0.5 * y0 + 0.5 * y2
        d = y1
        return a * f**3 + b * f**2 + c * f + d

    def filter(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"PitchShift expects {self.in_channels} input channels")

        # The read heads only ever see the buffer, so the DC blocker is applied
        # to the whole block at the end rather than once per sample.
        y = np.empty_like(x)
        for i in range(x.shape[0]):
            w = self.write_ptr
            self.buffer[w] = x[i]
            self.write_ptr = (w + 1) % self.max_delay

            d1 = self.min_delay + self.phase_1 * self.window_size
            d2 = self.min_delay + self.phase_2 * self.window_size

            s1 = self._read_interpolated(w - d1)
            s2 = self._read_interpolated(w - d2)

            f1 = np.sin(np.pi * self.phase_1)
            f2 = np.sin(np.pi * self.phase_2)

            y[i] = s1 * f1 + s2 * f2

            self.phase_1 = (self.phase_1 + self.phase_inc) % 1.0
            self.phase_2 = (self.phase_2 + self.phase_inc) % 1.0

        y = self.dc_blocker.filter(y)
        out = x.copy()
        out[:, self._mask] = y[:, self._mask]
        return out

    def reset(self) -> None:
        self.buffer[:] = 0.0
        self.write_ptr = 0
        self.phase_1 = 0.0
        self.phase_2 = 0.5
        self.dc_blocker.reset()


class _Grain(TypedDict):
    read_ptr: float
    pos: int


class GranularPitchShift(TimeOperator):
    """Stateful, controllable granular pitch shifter.

    Two grains, triggered half a grain apart, read a shared per-channel
    circular buffer at a rate set by ``transpose_cents``. Each grain is
    windowed by a raised-cosine envelope and, on reaching ``grain_dur_samps``,
    restarts at a new random position inside the buffer -- trading the
    continuous read-head wraparound of :class:`PitchShift` for grain-boundary
    clicks disguised by the envelope. Only ``active_channels`` are shifted;
    the rest pass through unchanged.
    """

    def __init__(
        self,
        channels: int,
        max_delay_samps: int,
        grain_dur_samps: int,
        transpose_cents: float,
        active_channels: ArrayLike,
        fade_ratio: float = 0.25,
        seed: int | None = None,
    ) -> None:
        if max_delay_samps <= 2 * grain_dur_samps:
            raise ValueError("max_delay_samps must be greater than 2 * grain_dur_samps")
        if not (0 < fade_ratio <= 0.5):
            raise ValueError("fade_ratio must be in (0, 0.5]")

        self.in_channels = self.out_channels = int(channels)
        self.max_delay = int(max_delay_samps)
        self.grain_dur = int(grain_dur_samps)
        self.fade_ratio = float(fade_ratio)
        self.seed = seed  # kept so reset() can rewind the grain positions too
        self.rng = np.random.default_rng(seed)
        self._mask = np.zeros(self.in_channels, dtype=bool)
        self._mask[np.asarray(active_channels)] = True
        self.dc_blocker = DCBlocker(self.in_channels, R=0.995, correct_loss=True)

        self.set_transpose_cents(transpose_cents)

        self.buffer = np.zeros((self.max_delay, self.in_channels), dtype=float)
        self.write_ptr = 0
        self.samples_written = 0

        # Two grains interleaved with 180 degrees phase offset
        self.grains = [
            self._new_grain(phase_offset=0),
            self._new_grain(phase_offset=self.grain_dur // 2),
        ]

    def set_transpose_cents(self, cents: float) -> None:
        self.transpose_cents = float(cents)
        self.pitch_ratio = 2.0 ** (self.transpose_cents / 1200.0)

    def _random_read_start(self) -> float:
        """Pick a random starting read position inside the filled buffer."""
        filled = min(self.samples_written, self.max_delay)
        max_age = max(filled - self.grain_dur, 1)
        age = self.rng.integers(self.grain_dur, max_age + self.grain_dur)
        return float((self.write_ptr - int(age)) % self.max_delay)

    def _new_grain(self, phase_offset: int = 0) -> _Grain:
        return {
            "read_ptr": self._random_read_start(),
            "pos": phase_offset % self.grain_dur,
        }

    def _grain_envelope(self, pos: int) -> float:
        fade_len = int(self.fade_ratio * self.grain_dur)
        if fade_len == 0:
            return 1.0
        if pos < fade_len:
            return np.sin(0.5 * np.pi * pos / fade_len)
        elif pos >= self.grain_dur - fade_len:
            pos_in_fade = pos - (self.grain_dur - fade_len)
            return np.cos(0.5 * np.pi * pos_in_fade / fade_len)
        else:
            return 1.0

    def _read_interpolated(self, ptr: float) -> np.ndarray:
        """Cubic interpolated read from the buffer, one value per channel."""
        ptr = ptr % self.max_delay
        i = int(ptr)
        f = ptr - i

        y0 = self.buffer[(i - 1) % self.max_delay]
        y1 = self.buffer[i]
        y2 = self.buffer[(i + 1) % self.max_delay]
        y3 = self.buffer[(i + 2) % self.max_delay]

        a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
        b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
        c = -0.5 * y0 + 0.5 * y2
        d = y1
        return a * f**3 + b * f**2 + c * f + d

    def filter(self, block: ArrayLike) -> np.ndarray:
        x = _as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"GranularPitchShift expects {self.in_channels} input channels"
            )

        # The grains only ever read the buffer, so the DC blocker is applied to
        # the whole block at the end rather than once per sample.
        y = np.zeros_like(x)
        for i in range(x.shape[0]):
            self.buffer[self.write_ptr] = x[i]
            self.write_ptr = (self.write_ptr + 1) % self.max_delay
            self.samples_written += 1

            for g in self.grains:
                val = self._read_interpolated(g["read_ptr"])
                env = self._grain_envelope(g["pos"])
                y[i] += val * env
                g["read_ptr"] = (g["read_ptr"] + self.pitch_ratio) % self.max_delay
                g["pos"] += 1

                if g["pos"] >= self.grain_dur:
                    g["read_ptr"] = self._random_read_start()
                    g["pos"] = 0

        y = self.dc_blocker.filter(y)
        out = x.copy()
        out[:, self._mask] = y[:, self._mask]
        return out

    def reset(self) -> None:
        self.buffer[:] = 0.0
        self.write_ptr = 0
        self.samples_written = 0
        self.dc_blocker.reset()
        # Rewind the draw as well, so a seeded operator repeats exactly.
        self.rng = np.random.default_rng(self.seed)
        self.grains = [
            self._new_grain(phase_offset=0),
            self._new_grain(phase_offset=self.grain_dur // 2),
        ]


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

    One cascade of second-order sections per channel, filtered with
    :func:`scipy.signal.sosfilt`. State persists across :meth:`filter` calls, so
    a long signal can be processed in consecutive blocks.

    Parameters
    ----------
    sos
        Per-channel SOS bank of shape ``(n_sections, 6, N)`` where
        ``N = num_channels``. Section rows are ``[b0, b1, b2, a0, a1, a2]``.
        This is the canonical SOS bank layout in pyFDN: it matches the FLAMO
        ``parallelSOSFilter`` input and the output of
        :func:`pyFDN.decay_to_first_order_shelf`,
        :func:`pyFDN.decay_to_one_pole`, and :func:`pyFDN.decay_to_geq`.

    Attributes
    ----------
    sos
        The same bank in the ``(N, n_sections, 6)`` layout :func:`scipy.signal.sosfilt`
        expects, one cascade per row.

    Notes
    -----
    Pass an instance as ``post_delay`` to :func:`pyFDN.process_fdn` to apply
    frequency-dependent absorption inside the feedback loop.
    :func:`pyFDN.build_to_impz` also constructs this class internally when an
    :class:`pyFDN.FDNBuild` contains per-delay-line ``filters``.
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

    def filter(self, block: ArrayLike) -> np.ndarray:
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
    state persists across :meth:`filter` calls. For long impulse responses use
    :class:`MatrixConvolver` instead, which computes the same convolution by FFT.

    Parameters
    ----------
    coeffs
        FIR coefficients of shape ``(n_out, n_in, n_taps)`` in the ``z^{-1}``
        convention (``coeffs[i, j, k]`` is the tap of ``z^{-k}`` from input ``j``
        to output ``i``).

    Notes
    -----
    :func:`pyFDN.process_fdn` constructs this filter automatically when its
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

    def filter(self, block: ArrayLike) -> np.ndarray:
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
    persists across :meth:`filter` calls, so it works both whole-signal and
    block-by-block -- including as the feedback path of a
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

    Each adjacent channel pair is rotated by a sinusoidally modulated angle, so
    the operator is orthogonal at every sample but never constant. Sitting on a
    :class:`~pyFDN.td.connectors.Recursion` feedback path, e.g.
    ``Series([Gain(A), TimeVaryingMatrix(N, 1.5, 0.35, fs, 0.1)])``, it makes the
    loop genuinely time-varying -- there is no static transfer function. This is
    the operator form of the ``post_matrix`` argument of
    :func:`pyFDN.process_fdn`.

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

    def filter(self, block: ArrayLike) -> np.ndarray:
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
    processing block) and by :func:`pyFDN.process_fdn` (one line per FDN delay).

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
