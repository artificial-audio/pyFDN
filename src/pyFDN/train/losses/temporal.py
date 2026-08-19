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
