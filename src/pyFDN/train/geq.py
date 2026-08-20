"""The graphic-EQ design of :func:`pyFDN.graphic_eq`, rewritten for training.

Two trainable modules in this package are the *same* filter design driven by
different quantities: the in-loop absorption of :mod:`pyFDN.train.decay`, whose
band targets come from a reverberation time, and the output EQ below, whose
band targets are the trained parameter itself. What they share is this module:
the ten-band graphic EQ of Schlecht and Habets (DAFx 2017) as a closed-form,
differentiable map from band targets in dB to an SOS bank.

Two things had to change to get there:

* :func:`pyFDN.design_geq` solves a bounded least-squares problem for the
  command gains at every call. It is linear in its target, so it collapses into
  one constant matrix (:func:`geq_design_matrix`) computed once, outside the
  training loop.
* the shelving and peaking prototypes of :func:`pyFDN.graphic_eq` are evaluated
  in torch (:func:`geq_sos_torch`) rather than numpy, so gradients reach the
  band gains.

The result is an ordinary ``(11, 6, n_channels)`` SOS bank, which is what
:func:`pyFDN.extract_build` reads back out.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Band layout of pyFDN's 10-band graphic EQ, shared with design_geq: the eight
# octave centres plus the two shelving crossovers, targeted at DC and Nyquist.
CENTER_FREQUENCIES = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], float)
SHELVING_CROSSOVER = np.array([46.0, 11360.0])
#: Number of design bands: DC, the eight octave centres, Nyquist.
N_BANDS = len(CENTER_FREQUENCIES) + 2
#: Number of biquad sections the design produces: a flat gain plus one per band.
N_SECTIONS = N_BANDS + 1

_R = 2.7
_NUM_CONTROL = 100


def geq_design_matrix(fs: float) -> np.ndarray:
    """Constant matrix taking 10 band targets in dB to 11 GEQ command gains.

    :func:`pyFDN.design_geq` solves a bounded least-squares problem for the
    command gains at every call. The problem is linear in the target and its
    matrix does not depend on it, so for training it is solved once, here, and
    the bounds are dropped -- they are inactive for the band gains a decay or an
    output EQ asks for.

    Returns
    -------
    np.ndarray
        Shape ``(11, 10)``: ``command_gains_db = M @ target_db``.
    """
    from pyFDN.auxiliary.utils import hertz_to_rad
    from pyFDN.graphicEQ.graphic_eq import graphic_eq
    from pyFDN.graphicEQ.probe_sos import probe_sos

    control_frequencies = np.round(np.logspace(0, np.log10(fs / 2.1), _NUM_CONTROL + 1))
    target_f = np.concatenate([[1.0], CENTER_FREQUENCIES, [float(fs)]])

    # W: the linear interpolation from the 10 band targets onto the control grid,
    # read off column by column from np.interp itself so it cannot drift from
    # design_geq's own interpolation.
    W = np.empty((len(control_frequencies), N_BANDS))
    for k in range(N_BANDS):
        unit = np.zeros(N_BANDS)
        unit[k] = 1.0
        W[:, k] = np.interp(control_frequencies, target_f, unit)

    prototype_gain = 10.0
    prototype_sos = graphic_eq(
        hertz_to_rad(CENTER_FREQUENCIES, fs),
        hertz_to_rad(SHELVING_CROSSOVER, fs),
        _R,
        prototype_gain * np.ones(N_BANDS + 1),
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

    center_omega = 2 * np.pi * CENTER_FREQUENCIES / fs
    shelving_omega = 2 * np.pi * SHELVING_CROSSOVER / fs
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


def make_output_geq(
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
    """Build the output EQ whose parameter is the band gain in dB.

    The post filter sits *outside* the recursion, so unlike the decay it puts no
    constraint on stability and needs no bound: any band gain is a valid filter,
    and the trained number is the gain you would plot. It starts flat unless you
    say otherwise, and being a graphic EQ it stays a smooth ten-band curve
    rather than 66 free biquad coefficients.

    Parameters
    ----------
    gain_db : array_like or float
        Initial gain in dB at the 10 design bands (DC, 63 Hz … 8 kHz, Nyquist),
        per output channel. A scalar or a ``(10,)`` vector is broadcast across
        channels; the full shape is ``(10, n_channels)``.
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
        A subclass whose ``param`` is the ``(10, n_channels)`` gain in dB and
        whose ``map`` is the differentiable GEQ design, so ``map(param)`` is an
        ``(11, 6, n_channels)`` SOS bank -- what :func:`pyFDN.extract_build`
        reads as ``post_eq``.
    """
    return _output_geq_class()(
        gain_db,
        n_channels,
        fs,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )


_OUTPUT_GEQ: Any = None


def _output_geq_class() -> Any:
    """The ``parallelSOSFilter`` subclass, built on first use.

    flamo is imported lazily throughout pyFDN, and a class statement cannot be;
    so the class is defined here, once, and cached.
    """
    global _OUTPUT_GEQ
    if _OUTPUT_GEQ is not None:
        return _OUTPUT_GEQ

    import torch
    from flamo.processor import dsp

    class OutputGEQ(dsp.parallelSOSFilter):  # type: ignore[misc]
        """Output EQ parametrized by per-band gain in dB. See the module docstring."""

        def __init__(
            self,
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
            gains = np.asarray(gain_db, dtype=np.float64)
            if gains.ndim == 0:
                gains = np.full(N_BANDS, float(gains))
            if gains.shape[0] != N_BANDS:
                raise ValueError(
                    f"gain_db must have {N_BANDS} bands (DC, 63 Hz … 8 kHz, "
                    f"Nyquist), got {gains.shape[0]}"
                )
            if gains.ndim == 1:
                gains = gains[:, None]
            gains = np.ascontiguousarray(
                np.broadcast_to(gains, (N_BANDS, int(n_channels)))
            )
            from pyFDN.auxiliary.flamo import _get_device

            super().__init__(
                size=(int(n_channels),),
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
            self.register_buffer(
                "geq_matrix",
                torch.tensor(
                    geq_design_matrix(float(fs)), dtype=torch_dtype, device=dev
                ),
            )
            # Replaces the (K, 6, N) coefficient parameter of the base class with
            # the ten band gains. torch.tensor, not as_tensor: a float64 argument
            # would otherwise share memory with the caller's array.
            self.param = torch.nn.Parameter(
                torch.tensor(gains, dtype=torch_dtype, device=dev),
                requires_grad=requires_grad,
            )
            self.map = self.gain_to_sos

        def gain_to_sos(self, gain_db: Any) -> Any:
            """Band gains in dB to an ``(11, 6, n_channels)`` SOS bank."""
            return geq_sos_torch(self.geq_matrix @ gain_db, self.fs_hz)

    _OUTPUT_GEQ = OutputGEQ
    return _OUTPUT_GEQ
