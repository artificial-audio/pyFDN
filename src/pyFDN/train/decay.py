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
(Schlecht and Habets, DAFx 2017), rewritten in closed form so it is
differentiable:

1. ``rt`` (per band) -> attenuation in dB per delay line, ``-60 d_i / (rt fs)``.
2. that target, interpolated onto the control grid and least-squares-fitted to
   the prototype band responses, gives the GEQ command gains. Both steps are
   linear, so they collapse into one constant matrix (:func:`geq_design_matrix`)
   applied to the target -- no iterative design inside the training loop.
3. the command gains become biquad sections by the same shelving/peak
   prototypes as :func:`pyFDN.graphic_eq`, evaluated in torch.

The mapped value is an ordinary ``(n_sections, 6, n_delays)`` SOS bank, so
:func:`pyFDN.extract_build` reads the trained decay back out unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Band layout of pyFDN's 10-band graphic EQ, shared with design_geq: the eight
# octave centres plus the two shelving crossovers, targeted at DC and Nyquist.
_CENTER_FREQUENCIES = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], float)
_SHELVING_CROSSOVER = np.array([46.0, 11360.0])
_R = 2.7
_NUM_CONTROL = 100


def geq_design_matrix(fs: float) -> np.ndarray:
    """Constant matrix taking 10 band targets in dB to 11 GEQ command gains.

    :func:`pyFDN.design_geq` solves a bounded least-squares problem for the
    command gains at every call. The problem is linear in the target and its
    matrix does not depend on it, so for training it is solved once, here, and
    the bounds are dropped -- they are inactive for the small per-round-trip
    attenuations an absorption filter asks for.

    Returns
    -------
    np.ndarray
        Shape ``(11, 10)``: ``command_gains_db = M @ target_db``.
    """
    from pyFDN.auxiliary.utils import hertz_to_rad
    from pyFDN.graphicEQ.graphic_eq import graphic_eq
    from pyFDN.graphicEQ.probe_sos import probe_sos

    control_frequencies = np.round(np.logspace(0, np.log10(fs / 2.1), _NUM_CONTROL + 1))
    target_f = np.concatenate([[1.0], _CENTER_FREQUENCIES, [float(fs)]])

    # W: the linear interpolation from the 10 band targets onto the control grid,
    # read off column by column from np.interp itself so it cannot drift from
    # design_geq's own interpolation.
    n_bands = len(target_f)
    W = np.empty((len(control_frequencies), n_bands))
    for k in range(n_bands):
        unit = np.zeros(n_bands)
        unit[k] = 1.0
        W[:, k] = np.interp(control_frequencies, target_f, unit)

    prototype_gain = 10.0
    prototype_sos = graphic_eq(
        hertz_to_rad(_CENTER_FREQUENCIES, fs),
        hertz_to_rad(_SHELVING_CROSSOVER, fs),
        _R,
        prototype_gain * np.ones(n_bands + 1),
    )
    G, _, _ = probe_sos(prototype_sos, control_frequencies, 2**16, fs)
    G = G / prototype_gain

    return np.linalg.pinv(G) @ W


def _peak_biquad(omega: Any, gain: Any, Q: float) -> tuple[Any, Any]:
    """Peaking biquad, batched over ``gain``. Torch mirror of ``bandpass_filter``."""
    import torch

    t = float(np.tan(omega / (2 * Q)))
    cos2 = float(-2 * np.cos(omega))
    sg = torch.sqrt(gain)
    ones = torch.ones_like(gain)
    b = torch.stack([sg + gain * t, cos2 * sg, sg - gain * t])
    a = torch.stack([sg + t * ones, cos2 * sg, sg - t * ones])
    return b, a


def _shelving_biquad(omega: float, gain: Any, kind: str) -> tuple[Any, Any]:
    """Shelving biquad, batched over ``gain``. Torch mirror of ``shelving_filter``."""
    import torch

    t = float(np.tan(omega / 2))
    t2 = t * t
    sqrt2 = float(np.sqrt(2.0))
    g2 = gain**0.5
    g4 = gain**0.25
    ones = torch.ones_like(gain)

    b = g2 * torch.stack(
        [
            g2 * t2 + sqrt2 * t * g4 + ones,
            2 * g2 * t2 - 2 * ones,
            g2 * t2 - sqrt2 * t * g4 + ones,
        ]
    )
    a = torch.stack(
        [
            g2 + sqrt2 * t * g4 + t2 * ones,
            2 * t2 * ones - 2 * g2,
            g2 - sqrt2 * t * g4 + t2 * ones,
        ]
    )
    if kind == "high":
        return a * gain, b
    return b, a


def geq_sos_torch(gain_db: Any, fs: float) -> Any:
    """Command gains in dB to an SOS bank, differentiably.

    Parameters
    ----------
    gain_db : torch.Tensor
        ``(11, n_channels)`` command gains: flat section, low shelf, eight
        peaking bands, high shelf -- the section order of
        :func:`pyFDN.graphic_eq`.

    Returns
    -------
    torch.Tensor
        ``(11, 6, n_channels)`` SOS bank, normalized to ``a0 = 1``.
    """
    import torch

    center_omega = 2 * np.pi * _CENTER_FREQUENCIES / fs
    shelving_omega = 2 * np.pi * _SHELVING_CROSSOVER / fs
    Q = float(np.sqrt(_R) / (_R - 1))

    gains = 10.0 ** (gain_db / 20.0)
    n_bands = gains.shape[0]
    sections = []
    for band in range(n_bands):
        g = gains[band]
        if band == 0:
            zero = torch.zeros_like(g)
            one = torch.ones_like(g)
            b, a = torch.stack([g, zero, zero]), torch.stack([one, zero, zero])
        elif band == 1:
            b, a = _shelving_biquad(shelving_omega[0], g, "low")
        elif band == n_bands - 1:
            b, a = _shelving_biquad(shelving_omega[1], g, "high")
        else:
            b, a = _peak_biquad(center_omega[band - 2], g, Q)
        sections.append(torch.cat([b, a], dim=0))

    sos = torch.stack(sections, dim=0)  # (n_bands, 6, n_channels)
    return sos / sos[:, 3:4, :]


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
            n_bands = len(_CENTER_FREQUENCIES) + 2
            if rt_arr.size != n_bands:
                raise ValueError(
                    f"rt must have {n_bands} bands (DC, 63 Hz … 8 kHz, Nyquist), "
                    f"got {rt_arr.size}"
                )
            from pyFDN.auxiliary.flamo import _get_device

            super().__init__(
                size=(int(delays_arr.size),),
                n_sections=n_bands + 1,
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
            slope = -60.0 / (rt.clamp_min(1e-3) * self.fs_hz)
            target_db = torch.outer(slope, self.delays_samples)  # (10, N)
            return geq_sos_torch(self.geq_matrix @ target_db, self.fs_hz)

    _DECAY_GEQ = DecayGEQ
    return _DECAY_GEQ
