"""Losses on the spectrum of the impulse response."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

from ._targets import _CachedTarget
from .base import ResponseLoss

if TYPE_CHECKING:
    import torch

    from pyFDN.train.response import Response

# How the output channels of |H| are combined before comparing with the target.
ChannelReduction = Literal["sum", "mean", "none"]


def _reduce_channels(magnitude: torch.Tensor, how: ChannelReduction) -> torch.Tensor:
    if how == "sum":
        return magnitude.sum(dim=1)
    if how == "mean":
        return magnitude.mean(dim=1)
    if how == "none":
        return magnitude
    raise ValueError(f"channels must be 'sum', 'mean' or 'none'; got {how!r}")


class FlatMagnitude(ResponseLoss):
    """Mean squared error of :math:`|H|` against a flat target -- *colorless*.

    Fits the magnitude spectrum of the (rectangularly truncated) impulse
    response to a constant, after *Differentiable FDNs for Colorless
    Reverberation* (Dal Santo et al.). The spectrum comes from
    :attr:`Response.spectrum <pyFDN.train.response.Response.spectrum>`, so its
    frequency resolution is the model's ``nfft`` and nothing else.

    Parameters
    ----------
    target : float
        The flat magnitude to fit. The default of 1 matches
        :func:`pyFDN.build_fdn`'s normalized input/output gains, which put the
        initial :math:`|H|` near unity.
    channels : {"sum", "mean", "none"}
        How the output channels are combined before the comparison. ``"sum"``
        (default) reproduces FLAMO's ``mse_loss`` convention. ``"none"`` fits
        each input/output pair to flat on its own, which is the well-posed
        choice for a multi-output FDN.

    Notes
    -----
    The frequency resolution of this loss is the model's ``nfft`` and nothing
    else, which makes the fit sensitive to it -- on the 8-line lossless FDN of
    ``example_train_colorless_FDN``, the flatness reached is 0.61 at
    ``nfft=2**12``, 0.67 at ``2**13`` and 0.70 at ``2**14``. The truncation to
    ``nfft`` samples is a rectangular window, so the peak-to-median range of
    ``|H|`` also grows with ``nfft`` (13 dB at ``2**12``, 30 dB at ``2**15`` for
    a lossless FDN) and the mean squared error weights the tallest modes ever
    more heavily. A multi-resolution spectral loss, whose analysis windows set
    its own resolution, is the way out of both.

    The optimization crosses long plateaus on this objective; ``train_fdn``'s
    default ``patience=10`` stops inside one of them. Raise it (~100) for a fit
    that has actually converged.

    A lossless FDN also has every pole exactly on the unit circle, where the
    frequency-domain evaluation breaks down; :meth:`check` warns if the model
    was built without the ``alias_decay_db`` that avoids it.
    """

    def __init__(
        self, target: float = 1.0, *, channels: ChannelReduction = "sum"
    ) -> None:
        self.target = float(target)
        self.channels = channels
        _reduce_channels_check(channels)

    def __call__(self, response: Response) -> torch.Tensor:
        import torch

        magnitude = _reduce_channels(response.magnitude, self.channels)
        return torch.nn.functional.mse_loss(
            magnitude, torch.full_like(magnitude, self.target)
        )

    def check(self, model: Any) -> None:
        from pyFDN.auxiliary.flamo import core_alias_decay_db
        from pyFDN.train.build import LOSSLESS_ALIAS_DECAY_DB

        if core_alias_decay_db(model.get_core()) > 0.0:
            return
        warnings.warn(
            "FlatMagnitude fits |H| but the model was built with "
            "alias_decay_db=0. A lossless FDN then has its poles exactly on the "
            "unit circle, |H| is unbounded, and the fit shrinks the gains "
            "instead of flattening the response. Rebuild with "
            f"build_fdn(..., alias_decay_db={LOSSLESS_ALIAS_DECAY_DB}) (the "
            "default when rt=None).",
            stacklevel=3,
        )


class FlatSpectrogram(ResponseLoss):
    r"""Flatness measured on multi-resolution smoothed spectra -- *colorless*.

    The multi-scale sibling of :class:`FlatMagnitude`, and the one whose
    frequency resolution is its own business rather than the model's. For each
    analysis window :math:`n`, the short-time magnitudes are averaged over
    frames into a smoothed (Welch) spectral estimate, which is then normalized
    by its own mean and fitted to flat:

    .. math::

        P_n[f] = \sqrt{\big\langle |S_n[t, f]|^2 \big\rangle_t}, \qquad
        \mathcal{L} = \frac{1}{|W|} \sum_{n \in W} \Big\langle \Big(
            \frac{P_n[f]}{\langle P_n \rangle_f} - 1 \Big)^2 \Big\rangle_f

    A short window smooths heavily and constrains the broad spectral tilt; a
    long one resolves individual modes. Because each scale is normalized by its
    own mean, the loss is invariant to overall gain -- it fits spectral *shape*
    only, and needs no assumption that :math:`|H| \approx 1`.

    Parameters
    ----------
    nfft : tuple of int
        STFT window sizes, each no longer than the model's ``nfft``. The default
        spans a factor of eight, which is what makes the objective multi-scale;
        one window alone is just a smoothed :class:`FlatMagnitude`.
    overlap : float
        Fractional overlap between frames (0.75 -> hop of a quarter window).

    Notes
    -----
    Averaging over frames *before* measuring flatness is the whole design. The
    obvious alternative -- asking each individual frame to have a flat spectrum
    -- is actively harmful: an isolated echo inside a short frame already has a
    perfectly flat frame spectrum, so that objective rewards an impulsive,
    comb-filtered impulse response. Trained on the 8-line FDN of
    ``example_train_colorless_FDN``, per-frame flatness drives the spectral
    flatness of the result *below* its random starting point, while the
    time-averaged form above reaches the same flatness as
    :class:`FlatMagnitude` and a considerably denser feedback matrix.

    The loss value is not comparable with :class:`FlatMagnitude`'s: the
    smoothing removes most of the mode-to-mode fluctuation, so it starts and
    ends an order of magnitude smaller. It is, however, far more stable against
    the model's ``nfft`` -- scoring one lossless FDN at ``nfft`` of 2**12, 2**13
    and 2**14 spreads this loss by 1.5x and :class:`FlatMagnitude` by 6.9x,
    which is the point of giving the objective windows of its own.
    """

    def __init__(
        self,
        *,
        nfft: tuple[int, ...] = (256, 512, 1024, 2048),
        overlap: float = 0.75,
    ) -> None:
        self.nfft = tuple(int(n) for n in nfft)
        self.overlap = float(overlap)
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1); got {self.overlap}")
        self._windows: dict[Any, Any] = {}

    def _window(self, n: int, response: Response) -> Any:
        """Hann window for size ``n``, cached per device/dtype across steps."""
        import torch

        key = (n, response.h.device, response.h.dtype)
        if key not in self._windows:
            self._windows[key] = torch.hann_window(
                n, device=response.h.device, dtype=response.h.dtype
            )
        return self._windows[key]

    def __call__(self, response: Response) -> torch.Tensor:
        import torch

        # torch.stft takes (batch, n_samples); fold every input/output pair into
        # the batch so each transfer path is scored on its own flatness.
        signal = response.h.permute(1, 2, 0).reshape(-1, response.n_samples)

        total: Any = None
        for n in self.nfft:
            if n > response.n_samples:
                raise ValueError(
                    f"STFT window {n} is longer than the response "
                    f"({response.n_samples} samples); shorten nfft= or build "
                    "the model with a larger nfft."
                )
            spectrogram = torch.stft(
                signal,
                n_fft=n,
                hop_length=max(1, int(n * (1.0 - self.overlap))),
                window=self._window(n, response),
                center=False,  # no zero-padded edge frames to skew the average
                return_complex=True,
            ).abs()
            # Welch estimate: average power over frames, back to a magnitude.
            smoothed = (spectrogram**2).mean(dim=-1).sqrt()
            level = smoothed.mean(dim=-1, keepdim=True)
            normalized = smoothed / level.clamp_min(torch.finfo(smoothed.dtype).tiny)
            term = torch.nn.functional.mse_loss(normalized, torch.ones_like(normalized))
            total = term if total is None else total + term
        return total / len(self.nfft)


class MatchMagnitude(ResponseLoss):
    """Mean squared error of :math:`|H|` against a reference impulse response.

    The magnitude-only sibling of :class:`MatchImpulseResponse`: fits the
    spectral envelope while ignoring phase.
    """

    def __init__(self, target: Any, *, channels: ChannelReduction = "none") -> None:
        self.channels = channels
        _reduce_channels_check(channels)
        self._target = _CachedTarget(target)

    def __call__(self, response: Response) -> torch.Tensor:
        import torch

        reference = torch.fft.rfft(self._target(response), dim=0).abs()
        return torch.nn.functional.mse_loss(
            _reduce_channels(response.magnitude, self.channels),
            _reduce_channels(reference, self.channels),
        )


class _FlamoSpectrogramLoss(ResponseLoss):
    """Shared plumbing for FLAMO's multi-resolution spectrogram criteria."""

    def __init__(
        self,
        target: Any,
        *,
        nfft: tuple[int, ...] = (256, 512, 1024),
        device: Any = None,
        **kwargs: Any,
    ) -> None:
        self.nfft = tuple(int(n) for n in nfft)
        self.device = device
        self.kwargs = kwargs
        self._target = _CachedTarget(target)
        self._criterion: Any = None

    def _build_criterion(self, response: Response) -> Any:
        raise NotImplementedError

    def __call__(self, response: Response) -> torch.Tensor:
        if self._criterion is None:
            self._criterion = self._build_criterion(response)
        reference = self._target(response)
        # FLAMO's spectrogram losses take (batch, n_samples, n_channels).
        return self._criterion(
            response.flamo_layout(), reference.permute(2, 0, 1).contiguous()
        )


class MatchSpectrogram(_FlamoSpectrogramLoss):
    """Multi-resolution STFT distance to a reference impulse response.

    Wraps FLAMO's ``mss_loss``. ``nfft`` is the tuple of STFT window sizes; the
    loss's own frequency resolution therefore comes from these, independent of
    the model's ``nfft``.
    """

    def _build_criterion(self, response: Response) -> Any:
        from flamo.optimize.loss import mss_loss

        return mss_loss(
            nfft=list(self.nfft),
            sample_rate=int(response.fs),
            device=self.device,
            **self.kwargs,
        )


class MatchMelSpectrogram(_FlamoSpectrogramLoss):
    """Mel-scaled multi-resolution STFT distance (FLAMO's ``mel_mss_loss``)."""

    def _build_criterion(self, response: Response) -> Any:
        from flamo.optimize.loss import mel_mss_loss

        return mel_mss_loss(
            nfft=list(self.nfft),
            sample_rate=int(response.fs),
            device=self.device,
            **self.kwargs,
        )


def _reduce_channels_check(how: ChannelReduction) -> None:
    if how not in ("sum", "mean", "none"):
        raise ValueError(f"channels must be 'sum', 'mean' or 'none'; got {how!r}")
