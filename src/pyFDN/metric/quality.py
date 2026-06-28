"""No-reference quality metrics for room impulse responses (RIRs).

**RIR metrics** map a mono RIR (1-D ``array-like``) plus sample rate ``fs`` to a
scalar grading one perceptual artifact:

================================  =====================================  ==========
artifact axis                     metric                                 reference
================================  =====================================  ==========
texture / low echo density        :func:`normalized_echo_density`        Abel & Huang 2006
tail non-Gaussianity              :func:`tail_kurtosis`                  Traer & McDermott 2016
flutter (amplitude modulation)    :func:`envelope_autocorrelation_peak`  AM/envelope perception
flutter (repetition tonality)     :func:`signal_autocorrelation_peak`    Dal Santo et al. 2022; Ando
irregular decay                   :func:`decay_fit_residual`             Schroeder 1965
coloration (spectrum flatness)    :func:`spectral_flatness`              Wiener entropy
coloration (spectrum spread)      :func:`spectral_magnitude_spread`      Dal Santo et al. 2024
================================  =====================================  ==========

Evaluate multichannel RIRs one channel at a time.

**Analytical FDN metrics** (:func:`modal_excitation_magnitudes`,
:func:`modal_excitation_spread`, :func:`modal_excitation_peak`) take a FLAMO FDN
``model`` and measure coloration from its modal-excitation distribution -- the
FDN-domain counterpart of :func:`spectral_magnitude_spread`. Have raw
state-space matrices? Convert with
:func:`pyFDN.translate.dss_to_flamo.dss_to_flamo` first.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import kurtosis

from pyFDN.auxiliary.acoustics import echo_density, edc

_EPS = 1e-12

# Empirical flutter band (Dal Santo et al. 2022): where flutter energy sits.
_FLUTTER_BAND_HZ = (250.0, 2_000.0)

# Autocorrelation period search ranges (ms): signal ACF = fast flutter / ringing,
# envelope ACF = slower modulation. Envelope max stays below ``trend_ms``.
_SIGNAL_PERIOD_MS = (0.1, 50.0)
_ENVELOPE_PERIOD_MS = (5.0, 150.0)


def _prominent_autocorr_peak(
    x: np.ndarray,
    fs: float,
    min_lag: int,
    max_lag: int,
    return_period: bool,
) -> float | tuple[float, float]:
    """Height and lag of the fundamental autocorrelation peak of ``x``.

    The smallest-lag peak within 80 % of the tallest (so harmonic multiples are
    not mistaken for the period); ``0`` when there is no positive periodic peak.
    """
    n = x.size
    # FFT-based linear autocorrelation: O(n log n) vs O(n^2) for np.correlate.
    nfft = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(x, nfft)
    autocorr = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
    if autocorr[0] <= _EPS:
        return (0.0, 0.0) if return_period else 0.0
    autocorr = autocorr / autocorr[0]

    min_lag = max(1, min_lag)
    max_lag = min(n - 1, max_lag)
    if max_lag <= min_lag:
        return (0.0, 0.0) if return_period else 0.0

    segment = autocorr[min_lag : max_lag + 1]
    peaks, _ = find_peaks(segment)
    if peaks.size == 0:
        return (0.0, 0.0) if return_period else 0.0

    heights = segment[peaks]
    h_max = float(heights.max())
    if h_max <= 0.0:  # only negative correlations -> no periodicity
        return (0.0, 0.0) if return_period else 0.0

    fundamental = int(peaks[heights >= 0.8 * h_max].min())
    lag = min_lag + fundamental
    peak = float(segment[fundamental])
    if return_period:
        return peak, lag / fs
    return peak


def _bandpass(
    x: np.ndarray,
    fs: float,
    fmin: float | None,
    fmax: float | None,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band/low/high-pass; identity when both edges None."""
    nyquist = fs / 2.0
    high = None if fmax is None else min(fmax, nyquist * 0.999)
    if fmin is not None and high is not None:
        sos = butter(
            order, [fmin / nyquist, high / nyquist], btype="band", output="sos"
        )
    elif fmin is not None:
        sos = butter(order, fmin / nyquist, btype="high", output="sos")
    elif high is not None:
        sos = butter(order, high / nyquist, btype="low", output="sos")
    else:
        return x
    return np.asarray(sosfiltfilt(sos, x), dtype=float)


def _as_mono(ir: ArrayLike) -> np.ndarray:
    """Validate and return a 1-D float copy of a mono impulse response."""
    arr = np.asarray(ir, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(
            f"Expected a mono (1-D) impulse response, got shape {np.shape(ir)}. "
            "Evaluate multichannel RIRs one channel at a time."
        )
    return arr


def _power_spectrum(
    ir: np.ndarray,
    fs: float | None,
    fmin: float | None,
    fmax: float | None,
) -> np.ndarray:
    """One-sided power spectrum (DC excluded), optionally band-limited."""
    spectrum = np.fft.rfft(ir)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(ir.size, d=1.0 / fs if fs else 1.0)

    # Always drop DC; it carries no audible coloration and breaks the log mean.
    keep = freqs > 0
    if fmin is not None:
        if fs is None:
            raise ValueError("fs is required when fmin/fmax are given")
        keep &= freqs >= fmin
    if fmax is not None:
        if fs is None:
            raise ValueError("fs is required when fmin/fmax are given")
        keep &= freqs <= fmax
    return power[keep]


def spectral_flatness(
    ir: ArrayLike,
    fs: float | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
) -> float:
    """Spectral flatness (Wiener entropy): geometric / arithmetic mean of the
    power spectrum. ``1`` for a flat (white) spectrum, near ``0`` for a tonal /
    coloured one.

    Parameters
    ----------
    ir : array-like, 1-D
        Mono impulse response.
    fs : float, optional
        Sample rate in Hz, only needed with ``fmin``/``fmax``.
    fmin, fmax : float, optional
        Restrict to a frequency band in Hz.

    Returns
    -------
    float
        Spectral flatness in ``[0, 1]``.
    """
    arr = _as_mono(ir)
    power = _power_spectrum(arr, fs, fmin, fmax)
    arithmetic_mean = float(np.mean(power))
    if arithmetic_mean <= _EPS:
        return 0.0
    geometric_mean = float(np.exp(np.mean(np.log(power + _EPS))))
    return geometric_mean / arithmetic_mean


def spectral_magnitude_spread(
    ir: ArrayLike,
    fs: float | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
) -> float:
    """Standard deviation of the RIR magnitude response in dB (coloration).

    Small for a flat (colourless) spectrum, larger with resonances/notches. The
    RIR-domain counterpart of :func:`modal_excitation_spread`.

    Reference: Dal Santo et al. (2024); Heldmann & Schlecht (2021).

    Parameters
    ----------
    ir : array-like, 1-D
        Mono impulse response.
    fs : float, optional
        Sample rate in Hz, only needed with ``fmin``/``fmax``.
    fmin, fmax : float, optional
        Restrict to a frequency band in Hz.

    Returns
    -------
    float
        Magnitude-response std in dB; larger means more coloration.
    """
    arr = _as_mono(ir)
    power = _power_spectrum(arr, fs, fmin, fmax)
    magnitude_db = 10.0 * np.log10(power + _EPS)
    return float(np.std(magnitude_db))


def tail_kurtosis(
    ir: ArrayLike,
    fs: float = 48_000.0,
    start: float = 0.0,
    normalize_decay: bool = True,
    env_win_ms: float = 20.0,
) -> float:
    """Excess kurtosis of the (decay-compensated) late tail (non-Gaussianity).

    A diffuse tail is decaying Gaussian noise -> ~``0`` once the decay is divided
    out; sparse or spiky tails are leptokurtic (``> 0``).

    Reference: Traer & McDermott (2016).

    Parameters
    ----------
    ir : array-like, 1-D
        Mono impulse response.
    fs : float, optional
        Sample rate in Hz (default 48000).
    start : float, optional
        Tail start time in seconds; earlier samples are ignored (default 0).
    normalize_decay : bool, optional
        Divide out the local RMS envelope before measuring (default True).
    env_win_ms : float, optional
        Envelope-estimate window in ms (default 20).

    Returns
    -------
    float
        Excess (Fisher) kurtosis; ``0`` for stationary Gaussian noise.
    """
    arr = _as_mono(ir)
    start_sample = int(round(start * fs))
    tail = arr[start_sample:]
    if tail.size < 4:
        raise ValueError("Tail too short to estimate kurtosis (need >= 4 samples).")

    if normalize_decay:
        win = max(1, int(round(env_win_ms * fs / 1000.0)))
        kernel = np.ones(win) / win
        local_power = np.convolve(tail**2, kernel, mode="same")
        envelope = np.sqrt(local_power)
        tail = tail / (envelope + _EPS)

    return float(kurtosis(tail, fisher=True, bias=False))


def envelope_autocorrelation_peak(
    ir: ArrayLike,
    fs: float = 48_000.0,
    rms_win_ms: float = 5.0,
    trend_ms: float = 200.0,
    bandpass: bool = False,
    return_period: bool = False,
) -> float | tuple[float, float]:
    """Peak of the energy-envelope autocorrelation (amplitude modulation).

    Detects periodic amplitude modulation ("acoustical flutter", beating) from a
    decay-detrended moving-RMS energy envelope. ``0`` for a diffuse tail, towards
    ``1`` for a modulated one; the peak lag is the modulation period. The
    slower-modulation counterpart of :func:`signal_autocorrelation_peak`.

    Reference: temporal-envelope / amplitude-modulation perception; Ando ``tau_e``.

    Parameters
    ----------
    ir : array-like, 1-D
        Mono impulse response.
    fs : float, optional
        Sample rate in Hz (default 48000).
    rms_win_ms : float, optional
        Energy-envelope RMS window in ms (default 5).
    trend_ms : float, optional
        Decay-detrend window in ms (default 200).
    bandpass : bool, optional
        Restrict to the 0.25--2 kHz flutter band first (default False).
    return_period : bool, optional
        If True, also return the modulation period in seconds.

    Returns
    -------
    float or tuple of (float, float)
        Peak autocorrelation in ``[0, 1]``, or ``(peak, period_s)`` when
        ``return_period`` is True.
    """
    arr = _as_mono(ir)
    if bandpass:
        arr = _bandpass(arr, fs, *_FLUTTER_BAND_HZ)

    # Moving-windowed RMS energy envelope, in dB.
    win = max(1, int(round(rms_win_ms * fs / 1000.0)))
    energy = np.convolve(arr**2, np.ones(win) / win, mode="same")
    env_db = 10.0 * np.log10(energy + _EPS)

    # Moving-average high-pass: removes decay & slow drift, keeps modulation.
    trend_win = max(1, int(round(trend_ms * fs / 1000.0)))
    trend = np.convolve(env_db, np.ones(trend_win) / trend_win, mode="same")
    fluctuation = env_db - trend

    min_ms, max_ms = _ENVELOPE_PERIOD_MS
    return _prominent_autocorr_peak(
        fluctuation,
        fs,
        int(round(min_ms * fs / 1000.0)),
        int(round(max_ms * fs / 1000.0)),
        return_period,
    )


def signal_autocorrelation_peak(
    ir: ArrayLike,
    fs: float = 48_000.0,
    bandpass: bool = True,
    return_period: bool = False,
) -> float | tuple[float, float]:
    """Peak of the waveform autocorrelation (flutter / repetition tonality).

    Detects a repetition period ``t_r`` in the (optionally flutter-band)
    late-reverberation waveform. ``0`` for a diffuse tail, towards ``1`` for a
    fluttery/tonal one; the peak lag is ``t_r`` (tonality ``f_0 = 1/t_r``).

    Reference: Dal Santo, Prawda & Välimäki (2022), "Flutter Echo Modeling",
    DAFx; Ando, normalized autocorrelation.

    Parameters
    ----------
    ir : array-like, 1-D
        Mono impulse response.
    fs : float, optional
        Sample rate in Hz (default 48000).
    bandpass : bool, optional
        Restrict to the 0.25--2 kHz flutter band first (default True).
    return_period : bool, optional
        If True, also return the period in seconds.

    Returns
    -------
    float or tuple of (float, float)
        Peak autocorrelation in ``[0, 1]``, or ``(peak, period_s)`` when
        ``return_period`` is True.
    """
    arr = _as_mono(ir)
    if bandpass:
        arr = _bandpass(arr, fs, *_FLUTTER_BAND_HZ)
    arr = arr - np.mean(arr)
    min_ms, max_ms = _SIGNAL_PERIOD_MS
    return _prominent_autocorr_peak(
        arr,
        fs,
        int(round(min_ms * fs / 1000.0)),
        int(round(max_ms * fs / 1000.0)),
        return_period,
    )


def decay_fit_residual(
    ir: ArrayLike,
    fs: float = 48_000.0,
    fit_start_db: float = -5.0,
    fit_end_db: float = -35.0,
) -> float:
    """RMS deviation (dB) of the energy decay curve from a single-slope fit.

    The Schroeder EDC of a clean exponential decay is a straight line in dB;
    kinks, double slopes and truncation inflate the residual. Fitted between
    ``fit_start_db`` and ``fit_end_db`` (avoiding the direct sound and noise
    floor).

    Reference: Schroeder (1965); ISO 3382.

    Parameters
    ----------
    ir : array-like, 1-D
        Mono impulse response.
    fs : float, optional
        Sample rate in Hz (default 48000).
    fit_start_db, fit_end_db : float, optional
        dB bounds of the fit region (default -5, -35).

    Returns
    -------
    float
        RMS residual in dB over the fit region.
    """
    arr = _as_mono(ir)
    energy = edc(arr)
    edc_db = 10.0 * np.log10(energy / (energy[0] + _EPS) + _EPS)

    region = (edc_db <= fit_start_db) & (edc_db >= fit_end_db)
    if np.count_nonzero(region) < 2:
        raise ValueError(
            "EDC does not span the requested fit range "
            f"[{fit_start_db}, {fit_end_db}] dB; provide a longer/cleaner IR "
            "or widen the range."
        )

    t = np.arange(edc_db.size, dtype=float)[region]
    y = edc_db[region]
    slope, intercept = np.polyfit(t, y, 1)
    residual = y - (slope * t + intercept)
    return float(np.sqrt(np.mean(residual**2)))


def normalized_echo_density(
    ir: ArrayLike,
    fs: float = 48_000.0,
    n: int = 1024,
    hop: int = 500,
    start: float = 0.0,
) -> float:
    """Mean normalized echo density (NED) of an RIR.

    ``1`` for a fully mixed (diffuse) field, ``< 1`` for sparse / sputtery
    texture. Wraps :func:`pyFDN.auxiliary.acoustics.echo_density`.

    Reference: Abel & Huang (2006), Proc. 121st AES Convention.

    Parameters
    ----------
    ir : array-like, 1-D
        Mono impulse response.
    fs : float, optional
        Sample rate in Hz (default 48000).
    n : int, optional
        Analysis window length in samples, must be even (default 1024).
    hop : int, optional
        Hop size in samples (default 500).
    start : float, optional
        Start time in seconds for the averaged region (default 0).

    Returns
    -------
    float
        Mean NED over the analysis region.
    """
    arr = _as_mono(ir)
    # The summary uses only the NED profile; the mixing-time search (and its
    # "not found" warning for sparse IRs) is irrelevant here.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mixing time not found")
        _, profile = echo_density(arr, n=n, fs=fs, hop=hop)
    start_sample = int(round(start * fs))
    region = profile[start_sample:]
    if region.size == 0:
        raise ValueError("start is at or beyond the end of the impulse response.")
    return float(np.mean(region))


# ============================================================================
# Analytical FDN coloration metrics (require the FDN, not a measured RIR)
# ============================================================================
def _modal_residue_magnitudes_db(
    residues: np.ndarray, is_conjugate: np.ndarray
) -> np.ndarray:
    """Per-mode excitation magnitudes (dB) from a pole-residue decomposition."""
    # Per-mode magnitude = Frobenius norm of the residue matrix (|rho| for SISO).
    magnitude = np.sqrt(np.sum(np.abs(residues) ** 2, axis=(1, 2)))
    # Expand reduced conjugate pairs (a partner has equal magnitude).
    counts = np.where(np.asarray(is_conjugate, dtype=bool), 2, 1)
    magnitude = np.repeat(magnitude, counts)
    return 20.0 * np.log10(magnitude + _EPS)


def modal_excitation_magnitudes(model: Any) -> np.ndarray:
    """Per-mode excitation magnitudes (dB) of an FDN, ``20 log10|rho_k|``.

    The residue magnitudes of the FDN's modal decomposition; their spread governs
    coloration (narrow -> white, wide -> ringing). Conjugate pairs are expanded,
    so the length equals the system order.

    Reference: Schlecht & Habets (2019); Heldmann & Schlecht (2021).

    Parameters
    ----------
    model : FLAMO model
        A FLAMO FDN model (e.g. from
        :func:`pyFDN.translate.dss_to_flamo.dss_to_flamo`). Have raw state-space
        matrices? Convert with ``dss_to_flamo`` first.

    Returns
    -------
    np.ndarray, shape ``(n_modes,)``
        Modal excitation magnitudes in dB. The :func:`modal_excitation_spread` /
        :func:`modal_excitation_peak` summaries are scale-invariant.
    """
    # Lazy import keeps the rest of this module (pure numpy/scipy) free of the
    # torch + FLAMO import cost unless a modal-excitation metric is used.
    from pyFDN.translate.flamo_to_pr import flamo_to_pr

    residues, _poles, _direct, is_conjugate, _meta = flamo_to_pr(model, verbose=False)
    return _modal_residue_magnitudes_db(residues, is_conjugate)


def modal_excitation_spread(model: Any) -> float:
    """Std (dB) of the FDN modal-excitation distribution (coloration).

    Small -> uniformly excited modes (white), large -> uneven (coloured).
    Scale-invariant. Takes a FLAMO ``model``.

    Reference: Heldmann & Schlecht (2021); Dal Santo et al. (2024).

    Returns
    -------
    float
        Std of ``20 log10|rho_k|`` in dB.
    """
    return float(np.std(modal_excitation_magnitudes(model)))


def modal_excitation_peak(model: Any) -> float:
    """Prominence (dB) of the loudest FDN mode: ``max - median`` of the modal
    excitations.

    Large when one dominant mode rings; ``0`` for a uniform distribution.
    Scale-invariant. Takes a FLAMO ``model``.

    Returns
    -------
    float
        Peak-to-median modal excitation in dB.
    """
    db = modal_excitation_magnitudes(model)
    return float(np.max(db) - np.median(db))
