"""Losses on the impulse response in the time domain."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import ResponseLoss

if TYPE_CHECKING:
    import torch

    from pyFDN.train.response import Response

from .distances import CompressedEnergyDistance, MaskedSquaredError, SquaredError
from .features import CumulativeEnergySurface, EnergyDecayCurve, Waveform
from .match import Match
from .reductions import MaskedRms, Mean, MeanRmsOverGroups


class MatchImpulseResponse(Match):
    """Mean squared error against a reference impulse response, sample by sample."""

    def __init__(self, target: Any) -> None:
        super().__init__(
            target=target,
            feature=Waveform(),
            distance=SquaredError(),
            reduction=Mean(),
        )


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


class MatchEnergyDecay(Match):
    """RMS dB error of the octave-band energy decay curves against a reference.

    The loss that sees the *decay* -- and the one to add when the decay is a
    trained parameter (a :class:`~pyFDN.AttenuationFilter` in the ``post_delay``
    hook). A magnitude
    spectrogram distance is not a substitute for fitting a decay; see :doc:`the
    design note </training_losses>`.

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
        distance = MaskedSquaredError(floor_db=self.floor_db)
        super().__init__(
            target=target,
            feature=EnergyDecayCurve(
                window=self.window, hop=self.hop, bands=self.bands
            ),
            distance=distance,
            reduction=MaskedRms(distance=distance),
        )

    def check(self, model: Any) -> None:
        nfft = int(model.nfft)
        if self.window > nfft:
            raise ValueError(
                f"{type(self).__name__} window ({self.window}) is longer than the "
                f"model's nfft ({nfft}); there is no decay to read."
            )


class MatchCumulativeEnergy(Match):
    r"""Doubly-cumulated energy against a reference -- decay *and* colour, no bands.

    Takes the short-time power spectrum of both signals and integrates it twice,
    **backwards in time** and **downwards in frequency**:

    .. math::

        E[f, t] = \sum_{t' \ge t} \; \sum_{f' \ge f} \big| S[f', t'] \big|^2

    so :math:`E[f, t]` is the energy still to come after time :math:`t` in the
    band above :math:`f`, and :math:`E[0, 0]` is the total energy. The loss is
    the RMS difference of the two surfaces after a compressive power.

    Cumulating twice carries both things a fit needs: read down the
    :math:`t = 0` edge and you have the integrated spectrum (the colour), read
    across the :math:`f = 0` edge and you have the full-band decay, and the
    interior ties them together band by band -- without the arbitrary
    quantization of octave bands. The compressive ``power`` (rather than a
    logarithm) keeps the surface's six-orders-of-magnitude dynamic range
    visible to the loss while staying bounded. See :doc:`the design note
    </training_losses>` for the reasoning behind the defaults.

    Parameters
    ----------
    target : array_like
        Reference IR, shape ``(n_samples,)``, ``(n_samples, n_out)`` or
        ``(n_samples, n_out, n_in)``. Zero-padded or truncated to the model's
        ``nfft``.
    window, hop : int
        STFT window and hop in samples. Unlike :class:`MatchEnergyDecay` this
        loss needs no window long enough to resolve an octave -- it never splits
        into octaves -- so the default is short.
    power : float
        The compression exponent :math:`p \in (0, 1]` applied to the normalized
        surface. 1 is no compression (raw energies), 0.5 the default, 0.25
        stronger.
    floor_db : float
        Hard floor on the normalized surface, in energy dB below the reference's
        total energy. It bounds the gradient of the compression near zero and
        keeps the fit off the numerical floor of the render; ``clamp`` means no
        gradient flows from anything below it.n
    frequency : {"descending", "ascending", "both"}
        Which way the frequency cumulation runs, and with it the loss's balance
        between the ends of the spectrum. The default ``"descending"`` (the
        plain reading of "energy above this frequency") gives a low band little
        gradient; ``"both"`` scores both directions and averages, and is what to
        reach for when the fit has to find a decay it was not given. See :doc:`the
        design note </training_losses>` for the comparison.

    Notes
    -----
    The surface is normalized by the **reference's** total energy, one constant
    -- not each surface by its own -- so a level error is a genuine error of the
    loss rather than something it is blind to. Fitting shape only, with the
    level left to another term, is the one thing this loss deliberately does
    not do.
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
        directions = (
            ("descending", "ascending") if frequency == "both" else (frequency,)
        )
        super().__init__(
            target=target,
            feature=CumulativeEnergySurface(
                window=self.window, hop=self.hop, directions=directions
            ),
            distance=CompressedEnergyDistance(power=self.power, floor_db=self.floor_db),
            reduction=MeanRmsOverGroups(),
        )

    def check(self, model: Any) -> None:
        nfft = int(model.nfft)
        if self.window > nfft:
            raise ValueError(
                f"{type(self).__name__} window ({self.window}) is longer than the "
                f"model's nfft ({nfft}); there is no decay to read."
            )

    def _surfaces(self, h: torch.Tensor) -> list[torch.Tensor]:
        """Per-direction doubly-cumulated energy (test compat helper)."""
        return [s for s in self.feature(h, 48000.0)]
