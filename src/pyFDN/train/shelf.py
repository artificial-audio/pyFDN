"""The first-order shelving designs of :mod:`pyFDN.auxiliary.acoustics`, for training.

:mod:`pyFDN.train.decay` and :mod:`pyFDN.train.geq` parametrize the in-loop
absorption and the output EQ by ten band values driving a graphic EQ. This
module is the same two filters at the other end of the complexity scale: **one
biquad each**, the first-order shelf of Jot (AES 2015) that
:func:`pyFDN.first_order_absorption` and :func:`pyFDN.first_order_shelving_eq`
design in numpy, rewritten as a differentiable map so the two shelf endpoints
are the trained parameter.

A first-order shelf has exactly two degrees of freedom once its crossover is
fixed -- its value at DC and its value at Nyquist -- so that is the parameter:
two reverberation times for the decay, two gains in dB for the output EQ. What
is between the endpoints is not free, and that is the point: the fit cannot
place a bump in one octave, so the decay it finds is the monotone tilt an
absorptive room usually has, described by two numbers instead of ten.

The design is unconditionally stable. Its pole is
``(1 - t/sqrt(k)) / (1 + t/sqrt(k))`` with ``t = tan(2*pi*f_c/fs) > 0`` for any
crossover below ``fs/4`` and ``sqrt(k) > 0`` for any pair of endpoint gains, so
no value of the parameter can put it outside the unit circle. What a decay
still needs a floor for is the *sign* of the attenuation: an RT driven below
zero turns ``-60 d / (rt fs)`` from an attenuation into a gain, and a loop
filter above unity is what makes an FDN diverge. ``DecayShelf`` floors it the
same way :mod:`pyFDN.train.decay`'s ``DecayGEQ`` does.

Both modules map onto an ordinary ``(1, 6, n_channels)`` SOS bank, so
:func:`pyFDN.extract_build` reads the trained filter back out unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .decay import MAX_ATTENUATION_DB

#: Number of shelf endpoints: the value at DC and the value at Nyquist.
N_ENDPOINTS = 2


def shelf_crossover_omega(fs: float, crossover_frequency: float | None) -> float:
    """Crossover in radians, defaulted and clamped as in the numpy design.

    Mirrors :func:`pyFDN.auxiliary.acoustics._first_order_shelf`: the default is
    ``fs/8`` (the midpoint of the bilinear-warped frequency axis) and anything
    above ``fs/5`` is clamped, since the pole leaves the unit circle at ``fs/4``.
    """
    f_c = fs / 8.0 if crossover_frequency is None else float(crossover_frequency)
    return min(f_c, fs / 5.0) / fs * 2.0 * np.pi


def shelf_sos_torch(h_dc: Any, h_ny: Any, omega: float) -> Any:
    """Endpoint gains (linear) to a one-section SOS bank, differentiably.

    Torch mirror of :func:`pyFDN.auxiliary.acoustics._first_order_shelf`.

    Parameters
    ----------
    h_dc, h_ny : torch.Tensor
        ``(n_channels,)`` linear-magnitude gains at DC and at Nyquist.
    omega : float
        Crossover in radians, from :func:`shelf_crossover_omega`.

    Returns
    -------
    torch.Tensor
        ``(1, 6, n_channels)`` SOS bank ``[b0, b1, b2, a0, a1, a2]``,
        normalized to ``a0 = 1``.
    """
    import torch

    t = float(np.tan(omega))
    sqrt_k = torch.sqrt(h_dc / h_ny)

    a0 = t / sqrt_k + 1.0
    b0 = (t * sqrt_k + 1.0) * h_ny / a0
    b1 = (t * sqrt_k - 1.0) * h_ny / a0
    a1 = (t / sqrt_k - 1.0) / a0

    zero = torch.zeros_like(b0)
    one = torch.ones_like(b0)
    return torch.stack([b0, b1, zero, one, a1, zero], dim=0)[None, :, :]


def make_decay_shelf(
    rt: np.ndarray,
    delays: np.ndarray,
    fs: float,
    nfft: int,
    *,
    crossover_frequency: float | None = None,
    alias_decay_db: float = 0.0,
    device: Any = None,
    dtype: Any = None,
    requires_grad: bool = True,
) -> Any:
    """Build the in-loop absorption whose parameter is ``(rt_dc, rt_nyquist)``.

    The one-biquad counterpart of :func:`pyFDN.train.decay.make_decay_geq`: the
    same homogeneous decay -- one reverberation time shared by every delay line,
    scaled per line by its round-trip length -- described by a first-order shelf
    instead of a ten-band graphic EQ.

    Parameters
    ----------
    rt : array_like
        Reverberation time in seconds at DC and at Nyquist, shape ``(2,)`` --
        the module's trainable parameter.
    delays : array_like
        Delay lengths in samples, one per line.
    fs : float
        Sampling rate in Hz.
    nfft : int
        FFT size, matching the rest of the model.
    crossover_frequency : float, optional
        Shelf crossover in Hz, fixed (not trained). Default ``fs/8``.
    alias_decay_db : float
        Anti-time-aliasing decay; must match every other module in the system.
    requires_grad : bool
        Whether the two reverberation times are trained.

    Returns
    -------
    flamo.processor.dsp.parallelSOSFilter
        A subclass whose ``param`` is the ``(2,)`` RT in seconds and whose
        ``map`` is the differentiable shelf design, so ``map(param)`` is a
        ``(1, 6, N)`` SOS bank -- what :func:`pyFDN.extract_build` reads.
    """
    return _decay_shelf_class()(
        rt,
        delays,
        fs,
        nfft=nfft,
        crossover_frequency=crossover_frequency,
        alias_decay_db=alias_decay_db,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )


def make_output_shelf(
    gain_db: Any,
    n_channels: int,
    fs: float,
    nfft: int,
    *,
    crossover_frequency: float | None = None,
    alias_decay_db: float = 0.0,
    device: Any = None,
    dtype: Any = None,
    requires_grad: bool = True,
) -> Any:
    """Build the output EQ whose parameter is ``(db_dc, db_nyquist)``.

    The one-biquad counterpart of :func:`pyFDN.train.geq.make_output_geq`. The
    post filter sits *outside* the recursion, so any pair of gains is a valid
    filter and no floor is needed.

    Parameters
    ----------
    gain_db : array_like or float
        Initial gain in dB at DC and at Nyquist, per output channel: a scalar or
        a ``(2,)`` vector is broadcast across channels, the full shape being
        ``(2, n_channels)``.
    n_channels : int
        Number of output channels the filter runs on.
    fs, nfft : float, int
        Sampling rate in Hz and the FFT size of the rest of the model.
    crossover_frequency : float, optional
        Shelf crossover in Hz, fixed (not trained). Default ``fs/8``.
    alias_decay_db : float
        Anti-time-aliasing decay; must match every other module in the system.
    requires_grad : bool
        Whether the two gains are trained.

    Returns
    -------
    flamo.processor.dsp.parallelSOSFilter
        A subclass whose ``param`` is the ``(2, n_channels)`` gain in dB and
        whose ``map`` is the differentiable shelf design, so ``map(param)`` is a
        ``(1, 6, n_channels)`` SOS bank -- what :func:`pyFDN.extract_build`
        reads as ``post_eq``.
    """
    return _output_shelf_class()(
        gain_db,
        n_channels,
        fs,
        nfft=nfft,
        crossover_frequency=crossover_frequency,
        alias_decay_db=alias_decay_db,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )


_DECAY_SHELF: Any = None
_OUTPUT_SHELF: Any = None


def _decay_shelf_class() -> Any:
    """The ``parallelSOSFilter`` subclass, built on first use.

    flamo is imported lazily throughout pyFDN, and a class statement cannot be;
    so the class is defined here, once, and cached.
    """
    global _DECAY_SHELF
    if _DECAY_SHELF is not None:
        return _DECAY_SHELF

    import torch
    from flamo.processor import dsp

    class DecayShelf(dsp.parallelSOSFilter):  # type: ignore[misc]
        """In-loop absorption parametrized by RT at DC and at Nyquist.

        See the module docstring. ``param`` is the ``(2,)`` RT in seconds;
        ``map`` designs the one-biquad shelf from it, differentiably.
        """

        def __init__(
            self,
            rt: np.ndarray,
            delays: np.ndarray,
            fs: float,
            *,
            nfft: int = 2**14,
            crossover_frequency: float | None = None,
            alias_decay_db: float = 0.0,
            device: Any = None,
            dtype: Any = None,
            requires_grad: bool = True,
        ) -> None:
            delays_arr = np.asarray(delays, dtype=np.float64).ravel()
            rt_arr = np.asarray(rt, dtype=np.float64).ravel()
            if rt_arr.size != N_ENDPOINTS:
                raise ValueError(
                    f"rt must have 2 endpoints (DC, Nyquist), got {rt_arr.size}"
                )
            from pyFDN.auxiliary.flamo import _get_device

            super().__init__(
                size=(int(delays_arr.size),),
                n_sections=1,
                nfft=nfft,
                fs=int(fs),
                alias_decay_db=alias_decay_db,
                # resolved here rather than left as None: flamo's Series asserts
                # every module in it carries the same dtype.
                device=_get_device(device),
                dtype=torch.float32 if dtype is None else dtype,
                # the design below already returns a0 = 1; flamo's own
                # normalization writes in place, which autograd rejects.
                normalize_a0=False,
            )
            torch_dtype, dev = self.param.dtype, self.param.device  # type: ignore[has-type]
            self.fs_hz = float(fs)
            self.omega = shelf_crossover_omega(float(fs), crossover_frequency)
            # One round trip of the longest delay line -- the RT at which a band
            # already asks for MAX_ATTENUATION_DB, i.e. an instantaneous decay.
            self.rt_floor = (
                60.0 / MAX_ATTENUATION_DB * float(delays_arr.max()) / float(fs)
            )
            self.register_buffer(
                "delays_samples",
                torch.tensor(delays_arr, dtype=torch_dtype, device=dev),
            )
            # Replaces the (1, 6, N) coefficient parameter of the base class:
            # the decay is one RT pair, shared by every delay line.
            self.param = torch.nn.Parameter(
                torch.tensor(rt_arr, dtype=torch_dtype, device=dev),
                requires_grad=requires_grad,
            )
            self.map = self.rt_to_sos

        def rt_to_sos(self, rt: Any) -> Any:
            """``(rt_dc, rt_nyquist)`` in seconds to a ``(1, 6, N)`` SOS bank."""
            # dB of attenuation per sample, then per delay-line round trip.
            slope = -60.0 / (self._floored(rt) * self.fs_hz)
            target_db = torch.outer(slope, self.delays_samples)  # (2, N)
            h = 10.0 ** (target_db / 20.0)
            return shelf_sos_torch(h[0], h[1], self.omega)

        def _floored(self, rt: Any) -> Any:
            """``rt`` held above :attr:`rt_floor`, smoothly.

            The shelf itself is stable for any endpoint gains, but a gradient
            step that puts an RT at or below zero flips ``-60 d / (rt fs)`` from
            an attenuation into a *gain*, and a loop filter above unity is what
            makes the recursion diverge. The floor is one round trip of the
            longest delay line and the softplus knee is one floor wide: the
            identity to float precision above ~20 floors, a few percent long
            just above it, and never below it -- so a band that dips across zero
            still has a gradient to come back on.
            """
            return self.rt_floor + torch.nn.functional.softplus(
                rt - self.rt_floor, beta=1.0 / self.rt_floor
            )

    _DECAY_SHELF = DecayShelf
    return _DECAY_SHELF


def _output_shelf_class() -> Any:
    """The ``parallelSOSFilter`` subclass, built on first use. See above."""
    global _OUTPUT_SHELF
    if _OUTPUT_SHELF is not None:
        return _OUTPUT_SHELF

    import torch
    from flamo.processor import dsp

    class OutputShelf(dsp.parallelSOSFilter):  # type: ignore[misc]
        """Output EQ parametrized by gain in dB at DC and at Nyquist."""

        def __init__(
            self,
            gain_db: Any,
            n_channels: int,
            fs: float,
            *,
            nfft: int = 2**14,
            crossover_frequency: float | None = None,
            alias_decay_db: float = 0.0,
            device: Any = None,
            dtype: Any = None,
            requires_grad: bool = True,
        ) -> None:
            gains = np.asarray(gain_db, dtype=np.float64)
            if gains.ndim == 0:
                gains = np.full(N_ENDPOINTS, float(gains))
            if gains.shape[0] != N_ENDPOINTS:
                raise ValueError(
                    f"gain_db must have 2 endpoints (DC, Nyquist), got {gains.shape[0]}"
                )
            if gains.ndim == 1:
                gains = gains[:, None]
            gains = np.ascontiguousarray(
                np.broadcast_to(gains, (N_ENDPOINTS, int(n_channels)))
            )
            from pyFDN.auxiliary.flamo import _get_device

            super().__init__(
                size=(int(n_channels),),
                n_sections=1,
                nfft=nfft,
                fs=int(fs),
                alias_decay_db=alias_decay_db,
                device=_get_device(device),
                dtype=torch.float32 if dtype is None else dtype,
                normalize_a0=False,
            )
            torch_dtype, dev = self.param.dtype, self.param.device  # type: ignore[has-type]
            self.fs_hz = float(fs)
            self.omega = shelf_crossover_omega(float(fs), crossover_frequency)
            self.param = torch.nn.Parameter(
                torch.tensor(gains, dtype=torch_dtype, device=dev),
                requires_grad=requires_grad,
            )
            self.map = self.gain_to_sos

        def gain_to_sos(self, gain_db: Any) -> Any:
            """Endpoint gains in dB to a ``(1, 6, n_channels)`` SOS bank."""
            h = 10.0 ** (gain_db / 20.0)
            return shelf_sos_torch(h[0], h[1], self.omega)

    _OUTPUT_SHELF = OutputShelf
    return _OUTPUT_SHELF
