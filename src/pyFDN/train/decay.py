"""A trainable decay: in-loop absorption parametrized by reverberation time.

The in-loop absorption filter is what sets an FDN's decay, and training it
directly -- as raw SOS coefficients -- does not work: nothing holds the poles
inside the unit circle, and the first thing a fit wants from a too-quiet FDN is
more loop gain, so the network diverges within a few dozen steps.

:class:`DecayGEQ` parametrizes the same filter by the quantity that *is* the
decay: the reverberation time per octave band, in seconds. Positive RT maps to
a negative dB attenuation per delay-line round trip, so the loop is contractive
for every value the parameter can take, and the trained number is the one the
notebook plots. The map is the graphic-EQ design of :func:`pyFDN.absorption_geq`
(Schlecht and Habets, DAFx 2017), made differentiable in :mod:`pyFDN.train.geq`:

1. ``rt`` (per band) -> attenuation in dB per delay line, ``-60 d_i / (rt fs)``,
   with ``rt`` held above a floor of one round trip so a step across zero does
   not turn the attenuation into a gain (see ``DecayGEQ._floored``).
2. that target becomes the GEQ command gains through the constant matrix of
   :func:`~pyFDN.train.geq.geq_design_matrix` -- no iterative design inside the
   training loop.
3. the command gains become biquad sections in
   :func:`~pyFDN.train.geq.geq_sos_torch`.

One reverberation time per band is shared by every delay line; what differs
between lines is only the round-trip length ``d_i``, which is exactly the
homogeneous decay an FDN is designed for.

The mapped value is an ordinary ``(n_sections, 6, n_delays)`` SOS bank, so
:func:`pyFDN.extract_build` reads the trained decay back out unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .geq import N_BANDS, N_SECTIONS, geq_design_matrix, geq_sos_torch

#: Most attenuation, in dB per delay-line round trip, a single band may ask for.
#: Not a stability bound -- any negative dB is contractive -- but a numerical
#: one: a band demanding several hundred dB more than its neighbours overflows
#: the graphic-EQ design in float32. 60 dB per round trip is already an
#: instantaneous decay, and stays four orders of magnitude clear of that.
MAX_ATTENUATION_DB = 60.0


def make_decay_geq(
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
    rt : array_like
        Reverberation time in seconds at the 10 GEQ design bands (DC, 63 Hz …
        8 kHz, Nyquist) -- the module's trainable parameter.
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
        differentiable GEQ design, so ``map(param)`` is an ``(11, 6, N)`` SOS
        bank -- what :func:`pyFDN.extract_build` reads.
    """
    return _decay_geq_class()(
        rt,
        delays,
        fs,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )


_DECAY_GEQ: Any = None


def _decay_geq_class() -> Any:
    """The ``parallelSOSFilter`` subclass, built on first use.

    flamo is imported lazily throughout pyFDN, and a class statement cannot be;
    so the class is defined here, once, and cached.
    """
    global _DECAY_GEQ
    if _DECAY_GEQ is not None:
        return _DECAY_GEQ

    import torch
    from flamo.processor import dsp

    class DecayGEQ(dsp.parallelSOSFilter):  # type: ignore[misc]
        """In-loop absorption parametrized by per-band reverberation time.

        See the module docstring. ``param`` is the RT in seconds at the 10 GEQ
        design bands; ``map`` designs the SOS bank from it, differentiably.
        """

        def __init__(
            self,
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
            rt_arr = np.asarray(rt, dtype=np.float64).ravel()
            if rt_arr.size != N_BANDS:
                raise ValueError(
                    f"rt must have {N_BANDS} bands (DC, 63 Hz … 8 kHz, Nyquist), "
                    f"got {rt_arr.size}"
                )
            from pyFDN.auxiliary.flamo import _get_device

            super().__init__(
                size=(int(delays_arr.size),),
                n_sections=N_SECTIONS,
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
            # One round trip of the longest delay line, the RT below which the
            # GEQ is asked for more than MAX_ATTENUATION_DB in a single band.
            self.rt_floor = (
                60.0 / MAX_ATTENUATION_DB * float(delays_arr.max()) / float(fs)
            )
            self.register_buffer(
                "delays_samples",
                torch.tensor(delays_arr, dtype=torch_dtype, device=dev),
            )
            self.register_buffer(
                "geq_matrix",
                torch.tensor(
                    geq_design_matrix(float(fs)), dtype=torch_dtype, device=dev
                ),
            )
            # Replaces the (K, 6, N) coefficient parameter of the base class:
            # the decay is one RT per band, shared by every delay line.
            # torch.tensor, not as_tensor: a float64 `rt` would otherwise share
            # memory with the caller's array, and training would rewrite it.
            self.param = torch.nn.Parameter(
                torch.tensor(rt_arr, dtype=torch_dtype, device=dev),
                requires_grad=requires_grad,
            )
            self.map = self.rt_to_sos

        def rt_to_sos(self, rt: Any) -> Any:
            """Reverberation time in seconds to an ``(11, 6, N)`` SOS bank."""
            # dB of attenuation per sample, then per delay-line round trip.
            slope = -60.0 / (self._floored(rt) * self.fs_hz)
            target_db = torch.outer(slope, self.delays_samples)  # (10, N)
            return geq_sos_torch(self.geq_matrix @ target_db, self.fs_hz)

        def _floored(self, rt: Any) -> Any:
            """``rt`` held above :attr:`rt_floor`, smoothly.

            A gradient step can put a band's RT at or below zero, where
            ``-60 d / (rt fs)`` is meaningless. Flooring it *hard* is worse than
            the crossing: the floored band then asks the GEQ for hundreds of dB
            of attenuation while its neighbours ask for none, and the design
            overflows float32 into NaN -- which is what actually breaks such a
            run. So the floor is one round trip of the longest delay line
            (:data:`MAX_ATTENUATION_DB` per round trip, four orders of
            magnitude below where the design overflows), and the softplus knee
            is one floor wide: the identity to float precision for every RT
            above ~20 floors (0.9 s for a 43 ms round trip), a few percent long
            just above the floor, and never below it.

            The knee leaves a usable gradient for a band that dips across zero,
            which is enough to pull it back. A band driven *far* under -- many
            knees -- has a gradient that underflows and stays at the floor,
            which is the right mapped value for it anyway: an RT below one round
            trip is an instantaneous decay however far below it goes.
            """
            import torch

            return self.rt_floor + torch.nn.functional.softplus(
                rt - self.rt_floor, beta=1.0 / self.rt_floor
            )

    _DECAY_GEQ = DecayGEQ
    return _DECAY_GEQ
