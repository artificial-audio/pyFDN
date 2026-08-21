"""Trainable absorption and output EQ: two roles, any :class:`EQDesign`.

A trainable filter in an FDN varies along two axes, and only two:

* **the role.** In-loop absorption sets the decay: its parameter is a
  reverberation time, one target per delay line scaled by that line's round-trip
  length, and it must stay contractive or the recursion diverges. The output EQ
  sits outside the recursion: its parameter is a gain in dB per output channel,
  it constrains nothing, and it needs no floor.
* **the design.** How many numbers describe the filter and how many biquads
  they become -- a ten-band graphic EQ, a first-order shelf, a one-pole. That is
  exactly :class:`pyFDN.eq.EQDesign`, and nothing here needs to know which one
  it was handed.

So this module holds two classes, not one per combination: :class:`DecayFilter`
and :class:`OutputFilter`, each taking a design. Adding a design adds no class.

Both parametrize by the *target* -- an RT, a band level -- rather than by the
design's own coefficients, which is what makes the trained numbers the ones you
would plot and, for the decay, what keeps the loop contractive for every value
the parameter can take. Both map onto an ordinary ``(n_sections, 6, n_channels)``
SOS bank, so :func:`pyFDN.extract_build` reads the trained filter back out
unchanged.

Shared and per-line decay
-------------------------

:class:`DecayFilter` reads its parameter shape to decide how much freedom the
decay has:

* ``(n_params,)`` -- **shared**. One reverberation time per band for the whole
  network; what differs between delay lines is only their round-trip length.
  This is the homogeneous decay an FDN is designed for, and the classical
  choice: every mode decays at the rate its frequency prescribes.
* ``(n_params, n_delays)`` -- **per line**. Each delay line carries its own
  reverberation time per band. The decay is no longer homogeneous -- lines fall
  silent at different rates, which a physical room does not do -- but a fit has
  ``n_delays`` times the freedom to place energy in time, which is worth having
  when the target is a measured response rather than an ideal room.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..eq.designs import EQDesign

#: Most attenuation, in dB per delay-line round trip, a single band may ask for.
#: Not a stability bound -- any negative dB is contractive -- but a numerical
#: one: a band demanding several hundred dB more than its neighbours overflows
#: the graphic-EQ design in float32. 60 dB per round trip is already an
#: instantaneous decay, and stays four orders of magnitude clear of that.
MAX_ATTENUATION_DB = 60.0


def make_decay_filter(
    design: EQDesign,
    rt: np.ndarray,
    delays: np.ndarray,
    fs: float,
    nfft: int,
    *,
    alias_decay_db: float = 0.0,
    device: Any = None,
    dtype: Any = None,
    requires_grad: bool = True,
) -> Any:
    """Build the in-loop absorption filter whose parameter is the RT.

    Parameters
    ----------
    design : EQDesign
        The filter design driven by the reverberation time -- a
        :class:`~pyFDN.eq.GraphicEQ`, :class:`~pyFDN.eq.FirstOrderShelf` or
        :class:`~pyFDN.eq.OnePole`.
    rt : array_like
        Reverberation time in seconds at the design's bands. ``(n_params,)`` for
        one decay shared by every delay line, ``(n_params, n_delays)`` for one
        per line -- see the module docstring.
    delays : array_like
        Delay lengths in samples, one per line.
    fs : float
        Sampling rate in Hz.
    nfft : int
        FFT size, matching the rest of the model.
    alias_decay_db : float
        Anti-time-aliasing decay; must match every other module in the system.
    requires_grad : bool
        Whether the reverberation time is trained.

    Returns
    -------
    flamo.processor.dsp.parallelSOSFilter
        A subclass whose ``param`` is the RT in seconds and whose ``map`` is the
        differentiable design, so ``map(param)`` is an
        ``(n_sections, 6, n_delays)`` SOS bank -- what
        :func:`pyFDN.extract_build` reads.
    """
    return _filter_classes()[0](
        design,
        rt,
        delays,
        fs,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )


def make_output_filter(
    design: EQDesign,
    gain_db: Any,
    n_channels: int,
    fs: float,
    nfft: int,
    *,
    alias_decay_db: float = 0.0,
    device: Any = None,
    dtype: Any = None,
    requires_grad: bool = True,
) -> Any:
    """Build the output EQ whose parameter is the gain in dB.

    The post filter sits *outside* the recursion, so unlike the decay it puts no
    constraint on stability and needs no bound: any gain is a valid filter, and
    the trained number is the gain you would plot. It starts flat unless you say
    otherwise, and being a parametrized design it stays a smooth curve rather
    than a bank of free biquad coefficients.

    Parameters
    ----------
    design : EQDesign
        The filter design driven by the band gains.
    gain_db : array_like or float
        Initial gain in dB at the design's bands, per output channel. A scalar
        or an ``(n_params,)`` vector is broadcast across channels; the full
        shape is ``(n_params, n_channels)``.
    n_channels : int
        Number of output channels the filter runs on.
    fs, nfft : float, int
        Sampling rate in Hz and the FFT size of the rest of the model.
    alias_decay_db : float
        Anti-time-aliasing decay; must match every other module in the system.
    requires_grad : bool
        Whether the band gains are trained.

    Returns
    -------
    flamo.processor.dsp.parallelSOSFilter
        A subclass whose ``param`` is the ``(n_params, n_channels)`` gain in dB
        and whose ``map`` is the differentiable design, so ``map(param)`` is an
        ``(n_sections, 6, n_channels)`` SOS bank -- what
        :func:`pyFDN.extract_build` reads as ``post_eq``.
    """
    return _filter_classes()[1](
        design,
        gain_db,
        n_channels,
        fs,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )


_CLASSES: Any = None


def _filter_classes() -> Any:
    """``(DecayFilter, OutputFilter)``, built on first use.

    flamo is imported lazily throughout pyFDN, and a class statement cannot be;
    so the classes are defined here, once, and cached.
    """
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES

    import torch
    from flamo.processor import dsp

    class DesignedSOS(dsp.parallelSOSFilter):  # type: ignore[misc]
        """A ``parallelSOSFilter`` whose ``param`` is an :class:`EQDesign` target.

        Holds what the two roles have in common: the flamo plumbing, the
        design's per-``fs`` constants as buffers, and the parameter that
        replaces the base class's raw coefficients.
        """

        def __init__(
            self,
            design: EQDesign,
            param_value: np.ndarray,
            n_channels: int,
            fs: float,
            *,
            nfft: int,
            alias_decay_db: float,
            device: Any,
            dtype: Any,
            requires_grad: bool,
        ) -> None:
            from pyFDN.auxiliary.flamo import _get_device

            super().__init__(
                size=(int(n_channels),),
                n_sections=design.n_sections,
                nfft=nfft,
                fs=int(fs),
                alias_decay_db=alias_decay_db,
                # resolved here rather than left as None: flamo's Series asserts
                # every module in it carries the same dtype.
                device=_get_device(device),
                dtype=torch.float32 if dtype is None else dtype,
                # the designs already return a0 = 1; flamo's own normalization
                # writes in place, which autograd rejects.
                normalize_a0=False,
            )
            torch_dtype, dev = self.param.dtype, self.param.device  # type: ignore[has-type]
            self.design = design
            self.fs_hz = float(fs)
            self._design_buffers = tuple(design.buffers(self.fs_hz))
            for name, value in design.buffers(self.fs_hz).items():
                self.register_buffer(
                    name, torch.tensor(value, dtype=torch_dtype, device=dev)
                )
            # Replaces the (K, 6, N) coefficient parameter of the base class with
            # the design's own targets. torch.tensor, not as_tensor: a float64
            # argument would otherwise share memory with the caller's array.
            self.param = torch.nn.Parameter(
                torch.tensor(param_value, dtype=torch_dtype, device=dev),
                requires_grad=requires_grad,
            )

        def design_sos(self, target_db: Any) -> Any:
            """Targets in dB through the design, with its constants."""
            buffers = {name: getattr(self, name) for name in self._design_buffers}
            return self.design.sos(target_db, self.fs_hz, **buffers)

    class DecayFilter(DesignedSOS):
        """In-loop absorption parametrized by reverberation time.

        See the module docstring. ``param`` is the RT in seconds, shared across
        delay lines or one per line; ``map`` designs the SOS bank from it,
        differentiably.
        """

        def __init__(
            self,
            design: EQDesign,
            rt: np.ndarray,
            delays: np.ndarray,
            fs: float,
            *,
            nfft: int = 2**14,
            alias_decay_db: float = 0.0,
            device: Any = None,
            dtype: Any = None,
            requires_grad: bool = True,
        ) -> None:
            delays_arr = np.asarray(delays, dtype=np.float64).ravel()
            rt_arr = _validated_target(
                rt, design, n_channels=delays_arr.size, broadcast=False
            )
            super().__init__(
                design,
                rt_arr,
                delays_arr.size,
                fs,
                nfft=nfft,
                alias_decay_db=alias_decay_db,
                device=device,
                dtype=dtype,
                requires_grad=requires_grad,
            )
            torch_dtype, dev = self.param.dtype, self.param.device
            self.register_buffer(
                "delays_samples",
                torch.tensor(delays_arr, dtype=torch_dtype, device=dev),
            )
            # The RT at which a band already asks for MAX_ATTENUATION_DB, i.e.
            # one round trip. A shared RT serves every line at once, so its floor
            # is set by the longest; a per-line RT gets its own line's floor,
            # which for the shorter lines is a good deal less conservative.
            floor = 60.0 / MAX_ATTENUATION_DB * delays_arr / float(fs)
            self.register_buffer(
                "rt_floor",
                torch.tensor(
                    floor if rt_arr.ndim == 2 else floor.max(),
                    dtype=torch_dtype,
                    device=dev,
                ),
            )
            self.map = self.rt_to_sos

        def rt_to_sos(self, rt: Any) -> Any:
            """Reverberation time in seconds to an ``(n_sections, 6, N)`` SOS bank."""
            # A shared RT gains a trailing axis so that both cases broadcast
            # against the delays the same way.
            per_line = rt if rt.ndim == 2 else rt[:, None]
            # dB of attenuation per sample, then per delay-line round trip.
            target_db = (
                -60.0 * self.delays_samples / (self._floored(per_line) * self.fs_hz)
            )
            return self.design_sos(target_db)

        def _floored(self, rt: Any) -> Any:
            """``rt`` held above :attr:`rt_floor`, smoothly.

            A gradient step can put a band's RT at or below zero, where
            ``-60 d / (rt fs)`` is not an attenuation but a *gain*, and a loop
            filter above unity is what makes an FDN diverge. Flooring it *hard*
            is worse than the crossing for a multi-band design: the floored band
            then asks for hundreds of dB of attenuation while its neighbours ask
            for none, and the graphic-EQ design overflows float32 into NaN. So
            the floor is one round trip (:data:`MAX_ATTENUATION_DB` of
            attenuation, four orders of magnitude below where the design
            overflows) and the softplus knee is one floor wide: the identity to
            float precision for every RT above ~20 floors (0.9 s for a 43 ms
            round trip), a few percent long just above the floor, and never
            below it.

            The knee leaves a usable gradient for a band that dips across zero,
            which is enough to pull it back. A band driven *far* under -- many
            knees -- has a gradient that underflows and stays at the floor,
            which is the right mapped value for it anyway: an RT below one round
            trip is an instantaneous decay however far below it goes.

            Written as ``floor * (1 + softplus(u))`` on the floor-relative
            ``u``, rather than as a ``beta=1/floor`` softplus of ``rt - floor``:
            the two are identical term for term, but ``beta`` must be a number,
            and a per-line floor is a vector.
            """
            floor = self.rt_floor
            return floor * (1.0 + torch.nn.functional.softplus((rt - floor) / floor))

    class OutputFilter(DesignedSOS):
        """Output EQ parametrized by gain in dB. See the module docstring."""

        def __init__(
            self,
            design: EQDesign,
            gain_db: Any,
            n_channels: int,
            fs: float,
            *,
            nfft: int = 2**14,
            alias_decay_db: float = 0.0,
            device: Any = None,
            dtype: Any = None,
            requires_grad: bool = True,
        ) -> None:
            gains = _validated_target(
                gain_db, design, n_channels=int(n_channels), broadcast=True
            )
            super().__init__(
                design,
                gains,
                int(n_channels),
                fs,
                nfft=nfft,
                alias_decay_db=alias_decay_db,
                device=device,
                dtype=dtype,
                requires_grad=requires_grad,
            )
            self.map = self.gain_to_sos

        def gain_to_sos(self, gain_db: Any) -> Any:
            """Gains in dB to an ``(n_sections, 6, n_channels)`` SOS bank."""
            return self.design_sos(gain_db)

    _CLASSES = (DecayFilter, OutputFilter)
    return _CLASSES


def _validated_target(
    value: Any, design: EQDesign, *, n_channels: int, broadcast: bool
) -> np.ndarray:
    """A design target as ``(n_params,)`` or ``(n_params, n_channels)``.

    ``broadcast`` fills the channel axis in (the output EQ always carries one,
    since its gains are per channel); without it a ``(n_params,)`` target stays
    one-dimensional and means "shared across channels".
    """
    target = np.asarray(value, dtype=np.float64)
    if target.ndim == 0:
        target = np.full(design.n_params, float(target))
    if target.shape[0] != design.n_params:
        raise ValueError(
            f"target must have {design.n_params} values -- "
            f"{design.param_description} -- got {target.shape[0]}"
        )
    if target.ndim > 2:
        raise ValueError(
            f"target must be 1- or 2-dimensional, got shape {target.shape}"
        )
    if target.ndim == 2 and target.shape[1] != n_channels:
        raise ValueError(
            f"a per-channel target must have {n_channels} columns, "
            f"got {target.shape[1]}"
        )
    if broadcast and target.ndim == 1:
        target = np.broadcast_to(target[:, None], (design.n_params, n_channels))
    return np.ascontiguousarray(target)
