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


def _reduce_channels_check(how: ChannelReduction) -> None:
    if how not in ("sum", "mean", "none"):
        raise ValueError(f"channels must be 'sum', 'mean' or 'none'; got {how!r}")


def _reduce_channels(magnitude: torch.Tensor, how: ChannelReduction) -> torch.Tensor:
    if how == "sum":
        return magnitude.sum(dim=1)
    if how == "mean":
        return magnitude.mean(dim=1)
    return magnitude


def _warn_if_magnitude_unbounded(model: Any, loss_name: str, consequence: str) -> None:
    """Warn when a flatness loss is asked to fit an unrenderable :math:`|H|`.

    A lossless FDN built with ``alias_decay_db=0`` has its poles exactly on the
    unit circle, where the FFT-domain evaluation is near-singular: ``|H|`` is
    unbounded and the fit chases numerical noise instead of the response.
    """
    from pyFDN.auxiliary.flamo import core_alias_decay_db
    from pyFDN.train.build import LOSSLESS_ALIAS_DECAY_DB

    if core_alias_decay_db(model.get_core()) > 0.0:
        return
    warnings.warn(
        f"{loss_name} fits |H| but the model was built with alias_decay_db=0. "
        "A lossless FDN then has its poles exactly on the unit circle, |H| is "
        f"unbounded, and {consequence} Rebuild with "
        f"build_fdn(..., alias_decay_db={LOSSLESS_ALIAS_DECAY_DB}) (the default "
        "when rt=None).",
        stacklevel=4,
    )


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
        _warn_if_magnitude_unbounded(
            model,
            "FlatMagnitude",
            "the fit shrinks the gains instead of flattening the response.",
        )


class AsymmetricFlatMagnitude(ResponseLoss):
    r"""Flatness that punishes **peaks** far harder than dips -- *colorless*.

    The asymmetric sibling of :class:`FlatMagnitude`, and a sharper statement of
    what "colorless" should mean. A resonant peak in a reverberator rings
    audibly at its own pitch; a dip of the same size is largely inaudible. This
    loss says so, by measuring :math:`|H|` against the response's own RMS level
    and raising the two sides of that deviation to **different powers**:

    .. math::

        d[f] = \frac{|H[f]|}{\sqrt{\langle |H|^2 \rangle_f}} - 1,
        \qquad
        \mathcal{L} = \Big\langle
            \big(d^{+}\big)^{p} + \big(d^{-}\big)^{2}
        \Big\rangle_f,
        \qquad
        d^{+} = \max(d, 0), \; d^{-} = \min(d, 0)

    with ``peak_power`` :math:`p \ge 2`. The exponent, not a weight, is what
    makes this bite: a weight multiplies every peak alike, whereas :math:`p = 4`
    makes a peak twice as tall cost *sixteen* times as much, so the fit spends
    its capacity on the few tallest modes -- the ones that are actually heard.

    Three properties are worth knowing:

    * **Flat is still the unique minimum**, at every ``peak_power`` --
      :math:`\mathcal{L} \ge 0` and it vanishes only at :math:`d \equiv 0`.
      Normalizing by the RMS forces :math:`\langle (1 + d)^2 \rangle = 1`, so
      even the peak term *on its own* cannot be zeroed by anything but a flat
      response: there is no way to buy a peak-free spectrum except by
      flattening it.
    * **Gain-invariant.** :math:`d` is built from a ratio, so this fits spectral
      *shape* only and never touches the overall level -- unlike
      :class:`FlatMagnitude`, which fits :math:`|H|` to an absolute constant and
      so anchors the gain as well. Add :class:`~pyFDN.Energy` if you want the
      level pinned too.
    * **Deliberately not in decibels.** dB is the obvious way to put a peak and
      a dip on equal footing before tilting between them, and it does not work:
      :math:`\partial\,\mathrm{dB}/\partial|H| \propto 1/|H|`, so the deepest
      nulls dominate the gradient however lightly they are weighted. A dB
      version of this loss stalls on a plateau within ~300 steps at *higher*
      peaks than :class:`FlatMagnitude` reaches. In the linear magnitude the
      gradient is :math:`\propto p\,(d^{+})^{p-1}`, largest exactly at the
      tallest peaks, and a dip is bounded at :math:`d = -1` on its own.

    Parameters
    ----------
    peak_power : float
        Exponent on the peak side; dips are always quadratic. Must be at least
        2, which is the symmetric-shape reference (still peak-biased, since a
        peak is unbounded and a dip is not). 4 is the default and the sweet
        spot; 6 is slower but steadier.

    Notes
    -----
    Measured on four 8-line lossless FDNs like the one in
    ``example_train_colorless_FDN``, at ``nfft=2**14``, each trained for up to
    2000 Adam steps at ``lr=1e-2``, ``patience=400``, then given homogeneous
    decay so the colouration can be measured on a fine grid. The number is the
    **tallest mode**, in dB above the response's own median, and the spectral
    flatness:

    ==================  =============  =========
    objective           tallest mode   flatness
    ==================  =============  =========
    ``FlatMagnitude``   17.5 dB        0.32
    ``peak_power=2``    17.1 dB        0.32
    ``peak_power=3``    15.9 dB        0.35
    ``peak_power=4``    13.7 dB        0.46
    ``peak_power=6``    14.2 dB        0.36
    ==================  =============  =========

    So the asymmetry is not simply trading peak height for deeper nulls. Over
    these four the mean 1st-percentile dip was *shallower* at ``peak_power=4``
    (-12.2 dB against -19.9 dB at 2) and plain spectral flatness improved -- but
    that average hides a lot of scatter, and on the single FDN of
    ``example_train_colorless_FDN`` the dip goes marginally the other way. The
    reliable claim is about the peak; treat the rest as FDN-dependent.

    What the exponent costs is steps and steadiness. The gradient vanishes like
    :math:`(d^{+})^{p-1}` near the optimum, so the fit runs long and needs the
    room to: raise ``max_steps`` and ``patience`` (``peak_power=4`` averaged
    1600 steps against 640 at 2, and 6 used the full 2000-step budget on every
    run). The seed-to-seed spread also grows with :math:`p` -- +-2.1 dB at 4
    against +-1.1 dB at 2, so a single run proves little.

    And the advantage is **not** unconditional: repeating the same comparison at
    ``nfft=2**13`` it disappears (16.5 dB at ``peak_power=4`` against 18.5 dB at
    2 with ``nfft=2**14``; 12.4 against 12.3 at ``2**13``, and running four
    times longer does not recover it). Measure your own case before assuming
    a higher exponent helps it.

    The loss value is **not** comparable across different ``peak_power`` -- the
    two terms have different units -- nor with :class:`FlatMagnitude`. Compare
    the responses, not the numbers.

    A lossless FDN has every pole exactly on the unit circle, where the
    frequency-domain evaluation breaks down; :meth:`check` warns if the model
    was built without the ``alias_decay_db`` that avoids it.
    """

    def __init__(self, *, peak_power: float = 4.0) -> None:
        self.peak_power = float(peak_power)
        if self.peak_power < 2.0:
            raise ValueError(
                "peak_power must be at least 2 (2 is the symmetric reference); "
                f"got {self.peak_power}"
            )

    def __call__(self, response: Response) -> torch.Tensor:
        import torch

        # Every input/output path is fitted on its own, each against its own
        # RMS: summing |H| across channels first, as FlatMagnitude does by
        # default, would measure the flatness of a sum rather than of the
        # transfer paths that carry the colouration.
        magnitude = response.magnitude
        rms = (magnitude**2).mean(dim=0, keepdim=True).sqrt()
        deviation = magnitude / rms.clamp_min(torch.finfo(magnitude.dtype).tiny) - 1.0

        peaks = deviation.clamp_min(0.0) ** self.peak_power
        dips = deviation.clamp_max(0.0) ** 2
        return (peaks + dips).mean()

    def check(self, model: Any) -> None:
        _warn_if_magnitude_unbounded(
            model,
            "AsymmetricFlatMagnitude",
            "its peaks are numerical artefacts rather than modes.",
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
