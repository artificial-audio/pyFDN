"""Losses on the impulse response in the time domain."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._targets import _CachedTarget
from .base import ResponseLoss

if TYPE_CHECKING:
    import torch

    from pyFDN.train.response import Response


class MatchImpulseResponse(ResponseLoss):
    """Mean squared error against a reference impulse response, sample by sample.

    The strictest of the matching losses -- it fits phase as well as magnitude,
    which for a reverberator is usually more than you want. Reach for
    :class:`~pyFDN.MatchSpectrogram` unless you are fitting an early part or a
    short filter.

    Parameters
    ----------
    target : array_like
        Reference IR, shape ``(n_samples,)``, ``(n_samples, n_out)`` or
        ``(n_samples, n_out, n_in)``. Zero-padded or truncated to the model's
        ``nfft``.
    """

    def __init__(self, target: Any) -> None:
        self._target = _CachedTarget(target)

    def __call__(self, response: Response) -> torch.Tensor:
        import torch

        return torch.nn.functional.mse_loss(response.h, self._target(response))


class Energy(ResponseLoss):
    """Squared deviation of the response's total energy from ``target``.

    A blunt level anchor: useful next to a magnitude-only loss to stop an
    objective that is invariant to overall gain from drifting.
    """

    def __init__(self, target: float = 1.0) -> None:
        self.target = float(target)

    def __call__(self, response: Response) -> torch.Tensor:
        energy = (response.h**2).sum()
        return (energy - self.target) ** 2


# Octave band edges around the 63 Hz … 8 kHz centres, the range a measured RIR
# actually carries. Below the first edge and above the last, a room impulse
# response is noise, and its "decay" is the noise floor's.
_OCTAVE_EDGES = (44.0, 88.0, 177.0, 354.0, 707.0, 1414.0, 2828.0, 5657.0, 11314.0)


class MatchEnergyDecay(ResponseLoss):
    """RMS dB error of the octave-band energy decay curves against a reference.

    The loss that sees the *decay* -- and the one to add when the decay is a
    trained parameter (:class:`pyFDN.Trainable` ``absorption``). A magnitude
    spectrogram distance is not a substitute: it compares two signals frame by
    frame, and two rooms with identical decay still have uncorrelated fine
    structure, so predicting *silence* scores better there than predicting the
    right amount of the wrong detail. On a 16-line FDN fitted to a 2.4 s hall,
    the mel spectrogram distance is minimized by an FDN whose RT is 40% short
    of the measurement; this loss is minimized within a few percent of it.

    Each band's Schroeder curve is normalized to its own value at :math:`t=0`,
    so the loss reads the decay and nothing else -- level is left to whatever
    else is in the objective.

    The value is in **dB**, which puts it many orders of magnitude above a
    spectrogram distance: weight accordingly, and read ``TrainLog.loss_log``
    (which stores every term unweighted) to see what each term is worth.

    Parameters
    ----------
    target : array_like
        Reference IR, shape ``(n_samples,)``, ``(n_samples, n_out)`` or
        ``(n_samples, n_out, n_in)``. Zero-padded or truncated to the model's
        ``nfft``.
    window, hop : int
        STFT window and hop in samples for the band energies. The default 4096
        (85 ms at 48 kHz) resolves the 63 Hz octave; shorter windows leave the
        low bands with too few bins to be worth reading.
    bands : sequence of float, optional
        Band edges in Hz; defaults to the octave bands from 44 Hz to 11.3 kHz.
    floor_db : float
        Only the part of each band's curve where the **target** is still above
        this level is compared. Past it a measurement is reading its own noise
        floor, and fitting that would fit the microphone.
    """

    def __init__(
        self,
        target: Any,
        *,
        window: int = 4096,
        hop: int | None = None,
        bands: Any = None,
        floor_db: float = -45.0,
    ) -> None:
        self.window = int(window)
        self.hop = int(hop) if hop is not None else int(window) // 4
        self.bands = tuple(
            float(f) for f in (bands if bands is not None else _OCTAVE_EDGES)
        )
        self.floor_db = float(floor_db)
        self._target = _CachedTarget(target)
        self._reference: torch.Tensor | None = None
        self._mask: torch.Tensor | None = None

    def check(self, model: Any) -> None:
        nfft = int(model.nfft)
        if self.window > nfft:
            raise ValueError(
                f"{type(self).__name__} window ({self.window}) is longer than the "
                f"model's nfft ({nfft}); there is no decay to read."
            )

    def _band_edc_db(self, h: torch.Tensor, fs: float) -> torch.Tensor:
        """``(n_channels, n_bands, n_frames)`` normalized Schroeder curves in dB."""
        import torch

        # (n_samples, n_out, n_in) -> (n_out * n_in, n_samples), the batch layout
        # torch.stft wants.
        x = h.permute(1, 2, 0).reshape(-1, h.shape[0])
        window = torch.hann_window(self.window, dtype=x.dtype, device=x.device)
        spectrum = torch.stft(
            x,
            n_fft=self.window,
            hop_length=self.hop,
            window=window,
            center=False,
            return_complex=True,
        )
        power = spectrum.real**2 + spectrum.imag**2  # (batch, freq, frames)
        freqs = torch.fft.rfftfreq(self.window, 1.0 / fs).to(x.device)

        band_power = torch.stack(
            [
                power[:, (freqs >= lo) & (freqs < hi), :].sum(dim=1)
                for lo, hi in zip(self.bands[:-1], self.bands[1:], strict=True)
            ],
            dim=1,
        )  # (batch, n_bands, frames)
        # Schroeder backward integration, per band.
        edc = torch.flip(torch.cumsum(torch.flip(band_power, [-1]), dim=-1), [-1])
        eps = torch.finfo(edc.dtype).tiny
        return 10.0 * torch.log10(edc / (edc[..., :1] + eps) + eps)

    def __call__(self, response: Response) -> torch.Tensor:
        import torch

        if self._reference is None:
            self._reference = self._band_edc_db(self._target(response), response.fs)
            self._mask = self._reference > self.floor_db
            if not bool(self._mask.any()):
                raise ValueError(
                    f"the reference never rises above floor_db={self.floor_db}; "
                    "it carries no decay to fit"
                )
        difference = (self._band_edc_db(response.h, response.fs) - self._reference)[
            self._mask
        ]
        return torch.sqrt((difference**2).mean())


def _reverse_cumsum(x: torch.Tensor, axis: int) -> torch.Tensor:
    """Cumulative sum running from the far end of ``axis`` back towards the near one."""
    import torch

    return torch.flip(torch.cumsum(torch.flip(x, [axis]), axis), [axis])


class MatchCumulativeEnergy(ResponseLoss):
    r"""Doubly-cumulated energy against a reference -- decay *and* colour, no bands.

    Takes the short-time power spectrum of both signals and integrates it twice,
    **backwards in time** and **downwards in frequency**:

    .. math::

        E[f, t] = \sum_{t' \ge t} \; \sum_{f' \ge f} \big| S[f', t'] \big|^2

    so :math:`E[f, t]` is the energy still to come after time :math:`t` in the
    band above :math:`f`, and :math:`E[0, 0]` is the total energy. The loss is
    the RMS difference of the two surfaces after a compressive power (below).

    Why cumulate twice
    ------------------
    The time direction is Schroeder backward integration -- the reason
    :class:`MatchEnergyDecay` exists, and the thing a spectrogram distance
    cannot see (two rooms with the same decay have uncorrelated fine structure,
    so against detail you cannot predict, silence scores better than the right
    amount of the wrong detail).

    The frequency direction does the same job **in place of splitting into
    octave bands**. Band edges are an arbitrary quantization: energy either side
    of one is treated as unrelated, and a fit can satisfy a band average while
    getting its shape wrong. A cumulative sum is the limit of ever-finer bands
    -- every bin is compared against every wider band containing it, at once --
    and it is monotone and smooth in both axes, which is worth a great deal to
    a gradient. The two cumulative sums are along different axes, so they
    commute; the surface does not depend on which is applied first.

    Between them the two directions carry both things a fit needs: read down
    the :math:`t = 0` edge and you have the integrated spectrum (the colour),
    read across the :math:`f = 0` edge and you have the full-band energy decay
    curve, and the interior ties the two together band by band.

    Compression instead of decibels
    -------------------------------
    A cumulative energy surface spans the whole dynamic range of the decay --
    six orders of magnitude and more -- and a plain MSE on it would see nothing
    but the first few frames. ``power`` compresses that range by raising the
    normalized surface to a fractional power: 0.5 (the default) compares
    amplitudes rather than energies, and lower values compress harder, moving
    weight onto the quiet end -- the late tail and the top of the spectrum.

    A logarithm is the obvious alternative and is worse here: it turns the
    silence *below* the response into an unbounded penalty, so the gradient is
    dominated by whichever bin is closest to zero. A power keeps the compressed
    surface bounded, its gradient :math:`\propto x^{p-1}` finite everywhere the
    floor allows, and 0 a perfectly ordinary value to predict.

    Parameters
    ----------
    target : array_like
        Reference IR, shape ``(n_samples,)``, ``(n_samples, n_out)`` or
        ``(n_samples, n_out, n_in)``. Zero-padded or truncated to the model's
        ``nfft``.
    window, hop : int
        STFT window and hop in samples. Unlike :class:`MatchEnergyDecay` this
        loss needs no window long enough to resolve an octave -- it never splits
        into octaves -- so the default is short. Lengthening it to 4096 changed
        the fit of the notes below by less than it cost in wall clock.
    power : float
        The compression exponent :math:`p \in (0, 1]` applied to the normalized
        surface. 1 is no compression (raw energies), 0.5 the default, 0.25
        stronger.
    floor_db : float
        Hard floor on the normalized surface, in energy dB below the reference's
        total energy. It bounds the gradient of the compression near zero and
        keeps the fit off the numerical floor of the render; ``clamp`` means no
        gradient flows from anything below it.
    frequency : {"descending", "ascending", "both"}
        Which way the frequency cumulation runs -- and with it, **the loss's
        balance between the ends of the spectrum**. Cumulating ``"descending"``
        (the default, high to low) puts every bin's energy into the rows below
        it, so an error in a *low* band moves only the largest values on the
        surface -- the ones the compression weights least -- while a high band
        has rows of its own. ``"ascending"`` mirrors that. ``"both"`` scores the
        two directions separately and averages -- the even-handed choice, and
        the better one where it has been measured (see the notes).

    Notes
    -----
    The surface is normalized by the **reference's** total energy, one constant
    -- not each surface by its own -- so a level error is a genuine error of the
    loss rather than something it is blind to. Fitting shape only, with the
    level left to another term, is the one thing this loss deliberately does
    not do.

    ``frequency`` is not a detail, and it outweighs ``power``. Fitting the
    16-line FDN of ``example_train_fdn_to_rir`` to a 2.4 s concert hall for 300
    Adam steps, from a flat 1 s decay that knows nothing about the room, and
    scoring the render against the measurement it never saw:

    ==================  =======  ==============  ============  ==============
    ``frequency``       ``power``  mean RT error  level shape   RT at 63 Hz
    ==================  =======  ==============  ============  ==============
    ``"descending"``    0.5      16.8 %          2.56 dB       0.26 s
    ``"descending"``    0.25     19.0 %          1.46 dB       0.44 s
    ``"ascending"``     0.5      12.8 %          1.12 dB       2.35 s
    ``"both"``          0.5      10.3 %          0.88 dB       2.17 s
    ==================  =======  ==============  ============  ==============

    where the room is 2.8 s at 63 Hz. Cumulating downwards alone leaves the
    bottom octave with almost no gradient and the fit abandons it; the other
    two directions recover it, and averaging them is best overall. A longer
    analysis window does not help (15.4 % at ``window=4096``, 63 Hz still at
    0.20 s), which is what identifies the cause as weighting rather than
    frequency resolution.

    The default is ``"descending"`` -- the plain reading of "cumulate the
    energy above this frequency" -- but on this evidence ``"both"`` is what to
    reach for when the fit has to find a decay it was not given.
    """

    def __init__(
        self,
        target: Any,
        *,
        window: int = 1024,
        hop: int | None = None,
        power: float = 0.5,
        floor_db: float = -100.0,
        frequency: str = "descending",
    ) -> None:
        self.window = int(window)
        self.hop = int(hop) if hop is not None else int(window) // 4
        self.power = float(power)
        if not 0.0 < self.power <= 1.0:
            raise ValueError(
                f"power must be in (0, 1]; got {self.power} (1 is no compression, "
                "smaller compresses harder)"
            )
        self.floor_db = float(floor_db)
        if frequency not in ("descending", "ascending", "both"):
            raise ValueError(
                "frequency must be 'descending', 'ascending' or 'both'; got "
                f"{frequency!r}"
            )
        self.frequency = frequency
        self._target = _CachedTarget(target)
        self._reference: list[torch.Tensor] | None = None
        self._scale: list[torch.Tensor] | None = None

    def check(self, model: Any) -> None:
        nfft = int(model.nfft)
        if self.window > nfft:
            raise ValueError(
                f"{type(self).__name__} window ({self.window}) is longer than the "
                f"model's nfft ({nfft}); there is no decay to read."
            )

    def _directions(self) -> tuple[str, ...]:
        """The cumulation directions this loss scores, one surface each."""
        if self.frequency == "both":
            return ("descending", "ascending")
        return (self.frequency,)

    def _energy(self, h: torch.Tensor) -> torch.Tensor:
        """``(n_channels, n_freq, n_frames)`` short-time power, cumulated in time."""
        import torch

        # (n_samples, n_out, n_in) -> (n_out * n_in, n_samples), the batch layout
        # torch.stft wants.
        x = h.permute(1, 2, 0).reshape(-1, h.shape[0])
        window = torch.hann_window(self.window, dtype=x.dtype, device=x.device)
        spectrum = torch.stft(
            x,
            n_fft=self.window,
            hop_length=self.hop,
            window=window,
            center=False,
            return_complex=True,
        )
        energy = spectrum.real**2 + spectrum.imag**2  # (batch, freq, frames)
        return _reverse_cumsum(energy, -1)  # backwards in time

    def _surfaces(self, h: torch.Tensor) -> list[torch.Tensor]:
        """The doubly-cumulated energy, one surface per cumulation direction."""
        import torch

        energy = self._energy(h)
        return [
            _reverse_cumsum(energy, -2)
            if direction == "descending"
            else torch.cumsum(energy, -2)
            for direction in self._directions()
        ]

    def _compressed(self, surface: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        floor = 10.0 ** (self.floor_db / 10.0)
        return (surface / scale).clamp_min(floor) ** self.power

    def __call__(self, response: Response) -> torch.Tensor:
        import torch

        # Both are set together, and mypy needs the guard to say so.
        if self._reference is None or self._scale is None:
            references = self._surfaces(self._target(response))
            # The largest value on a surface is the total energy: everything
            # after frame 0, on whichever side of the spectrum the cumulation
            # started from. One number for the whole reference, so relative
            # levels between input/output paths survive the normalization.
            self._scale = [
                r.amax(dim=(-2, -1)).mean().clamp_min(torch.finfo(r.dtype).tiny)
                for r in references
            ]
            if not all(bool(scale > 0) for scale in self._scale):
                raise ValueError("the reference carries no energy to fit")
            self._reference = [
                self._compressed(r, scale)
                for r, scale in zip(references, self._scale, strict=True)
            ]

        # Each direction is normalized and compressed on its own before it is
        # scored: averaging the raw surfaces instead would let the one that
        # starts from the loud end of the spectrum swamp the other, which is
        # the whole thing "both" exists to avoid.
        terms = [
            torch.sqrt(((self._compressed(surface, scale) - reference) ** 2).mean())
            for surface, scale, reference in zip(
                self._surfaces(response.h), self._scale, self._reference, strict=True
            )
        ]
        return sum(terms[1:], terms[0]) / len(terms)
