"""Acoustics and RT related functions."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import firwin2, sosfreqz
from scipy.special import erfc

from pyFDN.auxiliary.utils import db_to_lin, hertz_to_unit, lin_to_db

# The filter designs live in pyFDN.eq, which is where every EQ design in the
# package now lives. They are re-exported here because that is where they used
# to be, and because this module's rt_to_slope is what turns a reverberation
# time into the dB target they take. Imported inside a function there, not at
# module level, so the two modules do not cycle.
from pyFDN.eq.first_order import first_order_absorption as first_order_absorption
from pyFDN.eq.first_order import first_order_shelving_eq as first_order_shelving_eq
from pyFDN.eq.one_pole import one_pole_absorption as one_pole_absorption


def rt_to_slope(rt: ArrayLike, fs: float) -> np.ndarray:
    """Convert reverb time (RT, seconds) to energy decay slope (dB per sample)."""

    rt_arr = np.asarray(rt, dtype=float)
    return -60.0 / (rt_arr * fs)


def slope_to_rt(slope: ArrayLike, fs: float) -> np.ndarray:
    """Convert slope (dB/sample) to reverb time in seconds."""
    slope_arr = np.asarray(slope, dtype=float)
    return -60.0 / (slope_arr * fs)


def rt_to_gain_per_sample(rt: float, fs: float) -> float:
    """Convert reverb time (seconds) to gain coefficient per sample.

    The gain g satisfies g^(rt*fs) = 10^(-3), i.e. about -30 dB after rt seconds.
    """
    return 10 ** (-3 / (rt * fs))


def edc(ir: ArrayLike, axis: int = 0) -> np.ndarray:
    """Energy decay curve: backward cumulative sum of squared signal along an axis.

    EDC(t) = sum(ir[t:]^2), so the curve decreases from total energy to zero.
    Typically used with impulse responses with shape (n_samples, n_channels).

    Parameters
    ----------
    ir : array-like
        Signal(s). If 1D, EDC of that signal. If 2D (e.g. samples x channels),
        EDC is computed along the time axis for each channel.
    axis : int, optional
        Axis along which time runs (default 0). EDC is computed along this axis.

    Returns
    -------
    np.ndarray
        Same shape as ir. Values are non-negative and non-increasing along axis.
    """
    ir = np.asarray(ir, dtype=float)
    rev = np.flip(ir, axis=axis)
    cum = np.cumsum(rev**2, axis=axis)
    return np.flip(cum, axis=axis)


def absorption_filters(
    frequency: ArrayLike,
    target_rt: np.ndarray,
    filterOrder: int,
    delays: ArrayLike,
    fs: float,
) -> np.ndarray:
    """
    Generate FIR absorption filters for each channel.
    frequency: [freq_points]
    target_rt: shape (freq_points, channels)
    delays: array of length channels
    """
    delays_arr = np.asarray(delays, dtype=float)
    num_channels = len(delays_arr)
    unit_freq = hertz_to_unit(frequency, fs)
    FIR = np.zeros((num_channels, filterOrder + 1))

    if filterOrder == 0:
        rt = target_rt[0, :]
        db = delays_arr * rt_to_slope(rt, fs)
        FIR[:, 0] = db_to_lin(db)
    else:
        for ch in range(num_channels):
            rt = target_rt[:, ch]
            delay = delays_arr[ch] + int(np.ceil(filterOrder / 2))
            db = delay * rt_to_slope(rt, fs)
            target_amp = db_to_lin(db)
            # firwin2 expects normalized [0..1] freqs and gain values
            FIR[ch, :] = firwin2(filterOrder + 1, unit_freq, target_amp)
    return FIR


def absorption_to_rt(
    filterCoeffs: np.ndarray,
    delays: ArrayLike,
    nfft: int,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reverb time from recursive absorption filter with delay."""
    delays_arr = np.asarray(delays, dtype=float)
    filterLen = filterCoeffs.shape[1]
    response = np.fft.fft(filterCoeffs, nfft, axis=1)
    freq = np.linspace(0, fs / 2, nfft // 2, endpoint=False)

    response = response[:, : nfft // 2]
    freq = freq[: nfft // 2]

    totalDelay = delays_arr[:, None] + filterLen / 2
    decayPerSample = lin_to_db(np.abs(response)) / totalDelay
    rt = slope_to_rt(decayPerSample, fs)
    return rt.T, freq  # shape: (freq_points, channels)


def echo_density(
    ir: ArrayLike,
    n: int = 1024,
    fs: float = 48000.0,
    pre_delay: int = 0,
    mixing_thresh: float = 1.0,
    hop: int = 500,
) -> tuple[float, np.ndarray]:
    """Echo density and mixing time (Abel & Huang 2006).

    Computes the transition time between early reflections and stochastic
    reverberation assuming sound pressure in a reverberant field is
    Gaussian distributed.

    Reference: Abel & Huang (2006), "A simple, robust measure of
    reverberation echo density", Proc. 121st AES Convention, San Francisco.

    Parameters
    ----------
    ir : array-like
        Impulse response (1 channel only). Converted to 1D.
    n : int, optional
        Window length (must be even). Default 1024.
    fs : float, optional
        Sampling rate in Hz. Default 48000.
    pre_delay : int, optional
        Onset delay in samples for mixing time. Default 0.
    mixing_thresh : float, optional
        Normalized echo density threshold for mixing time (Abel & Huang use 1).
        Default 1.0.
    hop : int, optional
        Hop size in samples for sparse analysis. Default 500.

    Returns
    -------
    t_abel : float
        Mixing time in milliseconds (time at which echo density first
        exceeds mixing_thresh, relative to pre_delay). 0 if not found.
    echo_dens : np.ndarray
        Echo density vector (length = len(ir)), normalized; interpolated
        from sparse analysis.
    """
    ir_arr = np.asarray(ir, dtype=float).ravel()
    len_ir = len(ir_arr)
    if n % 2 != 0:
        raise ValueError("Window length n must be even.")
    if len_ir < n:
        raise ValueError(
            f"IR length {len_ir} is shorter than analysis window {n}. "
            "Provide at least an IR of some 100 ms."
        )
    half_win = n // 2
    w_tau = np.hanning(n)
    w_tau = w_tau / np.sum(w_tau)

    sparse_ind = np.arange(0, len_ir, hop, dtype=int)
    if sparse_ind[-1] != len_ir - 1 and len_ir - 1 not in sparse_ind:
        sparse_ind = np.append(sparse_ind, len_ir - 1)
    echo_dens_sparse = np.zeros(len(sparse_ind))

    for ii, n_center in enumerate(sparse_ind):
        if n_center <= half_win:
            h_tau = ir_arr[0 : n_center + half_win]
            w_t = w_tau[-(n_center + half_win) :]
        elif n_center <= len_ir - half_win - 1:
            h_tau = ir_arr[n_center - half_win : n_center + half_win]
            w_t = w_tau.copy()
        else:
            h_tau = ir_arr[n_center - half_win : len_ir]
            w_t = w_tau[: len(h_tau)].copy()

        s = np.sqrt(np.sum(w_t * (h_tau**2)))
        tip_ct = (np.abs(h_tau) > s).astype(float)
        echo_dens_sparse[ii] = np.sum(w_t * tip_ct)

    echo_dens_sparse = echo_dens_sparse / erfc(1.0 / np.sqrt(2))
    echo_dens = np.interp(
        np.arange(len_ir, dtype=float),
        sparse_ind.astype(float),
        echo_dens_sparse,
    )

    d = np.where(echo_dens > mixing_thresh)[0]
    if d.size > 0:
        first_idx = int(d[0])
        t_abel = (first_idx - pre_delay) / fs * 1000.0
        if t_abel < 0:
            t_abel = 0.0
    else:
        t_abel = 0.0
        warnings.warn(
            "Mixing time not found within given limits.", UserWarning, stacklevel=2
        )

    return float(t_abel), echo_dens


def octave_bands(
    fc: float = 1000.0,
    start: float = -4.0,
    n: int = 8,
    fs: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Octave band edges and centre frequencies.

    Centre frequencies are ``fc * 2**k`` for ``k = start … start + n - 1``; the
    band edges are the centre frequency divided and multiplied by ``sqrt(2)``.

    Parameters
    ----------
    fc : float
        Reference centre frequency in Hz (default 1000).
    start : float
        Octave offset of the lowest band relative to ``fc`` (default -4 → 62.5 Hz).
    n : int
        Number of octave bands (default 8).
    fs : float, optional
        If given, bands whose upper edge reaches Nyquist (``fs/2``) are dropped.

    Returns
    -------
    bands : (n_bands, 2) ndarray
        Lower and upper edge of each band in Hz.
    f_centre : (n_bands,) ndarray
        Centre frequency of each band in Hz.
    """
    f_centre = fc * 2.0 ** np.arange(start, start + n)
    fd = np.sqrt(2.0)
    bands = np.stack((f_centre / fd, f_centre * fd), axis=1)

    if fs is not None:
        valid = bands[:, 1] < fs / 2
        bands = bands[valid]
        f_centre = f_centre[valid]

    return bands, f_centre


def octave_band_filterbank(
    bands: np.ndarray, fs: float, filter_order: int = 8
) -> list[np.ndarray]:
    """Butterworth bandpass filters (SOS) for the given band edges.

    Parameters
    ----------
    bands : (n_bands, 2) array
        Lower and upper band edges in Hz, e.g. from :func:`octave_bands`.
    fs : float
        Sampling rate in Hz.
    filter_order : int
        Order of the bandpass filters (default 8). The Butterworth prototype
        order is ``filter_order // 2``, so the value counts poles of the
        bandpass, not of the lowpass prototype.

    Returns
    -------
    list of (n_sections, 6) ndarray
        One SOS array per band.
    """
    from scipy.signal import butter

    nyquist = fs / 2.0
    sos_bank = []
    for lower, upper in np.asarray(bands, dtype=float):
        if lower >= nyquist:
            raise ValueError(
                f"Band edge {lower} Hz is at or above Nyquist {nyquist} Hz"
            )
        # the top band is truncated just below Nyquist, where the filter design
        # would otherwise be ill-conditioned
        edges = np.minimum(0.99, np.array([lower, upper]) / nyquist)
        sos_bank.append(
            butter(filter_order // 2, edges, btype="bandpass", output="sos")
        )
    return sos_bank


def _rt_from_ir(ir: np.ndarray, fs: float, decay_db: float) -> float:
    """RT from a linear fit to the Schroeder decay curve of a single signal.

    The fit starts at -5 dB (skipping the direct sound) and spans ``decay_db``
    dB, or the available dynamic range if that is smaller. Its slope is
    extrapolated to a 60 dB decay. Returns 0 if the decay never reaches -5 dB.
    """
    energy = edc(ir)
    positive = np.flatnonzero(energy > 0)
    if positive.size < 2:
        return 0.0
    energy_db = 10.0 * np.log10(energy[: positive[-1] + 1])
    energy_db -= energy_db[0]

    # shrink the fit range if the decay curve does not span decay_db + 5 dB
    dynamic_range = -np.min(energy_db)
    if dynamic_range - 5.0 < decay_db:
        decay_db = dynamic_range

    below_5db = np.flatnonzero(energy_db < -5.0)
    if below_5db.size == 0:
        return 0.0
    i_start = int(below_5db[0])
    below_end = np.flatnonzero(energy_db < energy_db[i_start] - decay_db)
    i_end = int(below_end[0]) if below_end.size else len(energy_db)

    segment = energy_db[i_start:i_end]
    if segment.size < 2:
        return 0.0

    time = np.arange(segment.size) / fs
    slope = np.polyfit(time, segment - segment[0], 1)[0]
    if slope >= 0:
        return 0.0
    return float(-60.0 / slope)


def estimate_rt_bands(
    ir: ArrayLike,
    fs: float,
    fc: float = 1000.0,
    start: float = -4.0,
    n: int = 8,
    filter_order: int = 8,
    decay_db: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate RT in octave bands via Butterworth bandpass filtering.

    Filters the impulse response into octave bands (:func:`octave_bands`,
    :func:`octave_band_filterbank`), then fits a line to the Schroeder decay
    curve of each band: the fit starts at -5 dB and spans ``decay_db``, and its
    slope is extrapolated to a 60 dB decay.

    Assumes a **single-slope** decay per band. For multi-exponential decays
    (coupled rooms) estimate the slopes with a dedicated multi-slope estimator
    and convert its amplitudes with :func:`slope_amplitude_to_level`.

    Default bands: 63, 125, 250, 500, 1000, 2000, 4000, 8000 Hz (``start=-4, n=8``).
    Bands whose upper edge reaches ``fs/2`` are dropped.

    Parameters
    ----------
    ir : array-like, 1-D
        Impulse response.
    fs : float
        Sampling rate in Hz.
    fc : float
        Octave-band reference centre frequency in Hz (default 1000).
    start : float
        Octave offset of the lowest band relative to ``fc`` (default -4 → 62.5 Hz).
    n : int
        Number of octave bands (default 8).
    filter_order : int
        Butterworth filter order (default 8).
    decay_db : float
        Decay range in dB used for the linear fit. The default 30 dB fit is
        extrapolated to a 60 dB reverberation time.

    Returns
    -------
    rt : (n_bands,) ndarray
        Estimated RT in seconds per band.
    f_centre : (n_bands,) ndarray
        Centre frequencies in Hz corresponding to each RT value.
    """
    from scipy.signal import sosfilt

    ir = np.asarray(ir, dtype=float).ravel()
    bands, f_centre = octave_bands(fc=fc, start=start, n=n, fs=fs)
    sos_bank = octave_band_filterbank(bands, fs, filter_order)

    rt = np.zeros(len(f_centre))
    for k, sos in enumerate(sos_bank):
        rt[k] = _rt_from_ir(sosfilt(sos, ir), fs, decay_db)

    return rt, f_centre


def slope_amplitude_to_level(
    amplitude: ArrayLike, decay_time: ArrayLike, fs: float
) -> np.ndarray:
    """Initial amplitude of an exponential decay from its energy (EDC amplitude).

    A decay with initial amplitude ``L`` and reverberation time ``T``, i.e. the
    envelope ``L * 10**(-3 t / T)``, carries the energy
    ``E = L**2 * T * fs / (6 ln 10)``, so ``L = sqrt(6 ln(10) E / (T fs))``.

    ``E`` is the amplitude of the energy decay curve at ``t = 0``, which is what
    multi-slope estimators (DecayFitNet, Bayesian decay analysis) report as the
    slope amplitude ``A`` — one value per slope and band. The conversion is
    therefore how a multi-slope estimate becomes a set of per-slope FDN levels.
    Note that estimators usually normalise the EDC to 0 dB, in which case the
    amplitudes must be multiplied by the reported normalisation value first.

    Slopes with ``decay_time == 0`` are inactive and map to level 0.

    Parameters
    ----------
    amplitude : array-like
        Energy of the decay, i.e. the EDC amplitude of the slope. Any shape.
    decay_time : array-like
        Reverberation time in seconds, broadcastable against ``amplitude``.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    ndarray
        Initial level (linear amplitude), broadcast shape of the inputs.

    See Also
    --------
    estimate_initial_level_bands : single-slope band levels straight from an IR.
    """
    amplitude_arr, decay_arr = np.broadcast_arrays(
        np.asarray(amplitude, dtype=float), np.asarray(decay_time, dtype=float)
    )
    active = decay_arr > 0
    level = np.zeros(amplitude_arr.shape, dtype=float)
    np.sqrt(
        6.0
        * np.log(10.0)
        * np.where(active, amplitude_arr, 0.0)
        / np.where(active, decay_arr * fs, 1.0),
        out=level,
    )
    return level


def estimate_initial_level_bands(
    ir: ArrayLike,
    rt: ArrayLike,
    fs: float,
    fc: float = 1000.0,
    start: float = -4.0,
    n: int = 8,
    filter_order: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the initial level of the exponential decay per octave band.

    Companion to :func:`estimate_rt_bands` (same octave filterbank). Models
    the squared band-filtered impulse response as ``L^2 * 10^(-6 t / T)`` and
    matches the total band energy, see :func:`slope_amplitude_to_level`. This
    replaces the DecayFitNet initial-level estimate used in the MATLAB
    ``example_RIR2FDN``.

    Parameters
    ----------
    ir : array-like, 1-D
        Impulse response, starting at the onset.
    rt : array-like
        RT in seconds per band, as returned by :func:`estimate_rt_bands`
        with the same band parameters.
    fs : float
        Sampling rate in Hz.
    fc, start, n, filter_order
        Octave filterbank parameters, see :func:`estimate_rt_bands`.

    Returns
    -------
    level : (n_bands,) ndarray
        Initial level (linear amplitude) per band.
    f_centre : (n_bands,) ndarray
        Centre frequencies in Hz corresponding to each level.
    """
    from scipy.signal import sosfilt

    ir = np.asarray(ir, dtype=float).ravel()
    rt = np.asarray(rt, dtype=float).ravel()
    bands, f_centre = octave_bands(fc=fc, start=start, n=n, fs=fs)
    if rt.size != len(f_centre):
        raise ValueError("rt must have one entry per octave band")

    sos_bank = octave_band_filterbank(bands, fs, filter_order)
    energy = np.array([np.sum(sosfilt(sos, ir) ** 2) for sos in sos_bank])

    return slope_amplitude_to_level(energy, rt, fs), f_centre


def sos_gain_per_sample_curves(
    sos: np.ndarray,
    delays: ArrayLike,
    nfft: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude response (gain per sample vs angle) for a per-channel SOS bank.

    Evaluates :math:`|H(e^{j\\omega})|` at ``nfft`` angles from 0 to :math:`\\pi`
    (Nyquist) for each channel's SOS cascade, then scales by delay length so that
    the result is gain per sample: for channel j with delay m_j, the curve is
    :math:`|H|^{1/m_j}`, so that after m_j samples the effective gain is
    :math:`|H|`. Useful for plotting absorption/gain curves (e.g. on a pole plot).

    Parameters
    ----------
    sos : (n_sections, 6, N) array
        Per-channel SOS bank; section rows are ``[b0, b1, b2, a0, a1, a2]``. Same
        format as :func:`one_pole_absorption` / :func:`first_order_absorption`
        return.
    delays : (N,) array-like
        Delay lengths in samples, one per channel. Used to scale gain to per-sample.
    nfft : int
        Number of frequency points (default 512).

    Returns
    -------
    angles : (nfft,) array
        Angles in rad/sample, 0 to pi.
    magnitude : (nfft, N) array
        Gain per sample (linear), i.e. :math:`|H(e^{j\\omega})|^{1/m}` per channel.
    """
    sos = np.asarray(sos, dtype=np.float64)
    delays_arr = np.asarray(delays, dtype=np.float64).ravel()
    if sos.ndim != 3 or sos.shape[1] != 6:
        raise ValueError("sos must have shape (n_sections, 6, N)")
    N = sos.shape[2]
    if delays_arr.shape[0] != N:
        raise ValueError("delays must have length N (number of channels in sos)")
    if np.any(delays_arr < 1):
        raise ValueError("delays must be >= 1")
    magnitude = np.zeros((nfft, N), dtype=np.float64)
    angles = np.zeros(nfft, dtype=np.float64)
    for ch in range(N):
        angles, h = sosfreqz(sos[:, :, ch], worN=nfft)
        magnitude[:, ch] = np.power(np.abs(h), 1.0 / delays_arr[ch])
    return angles, magnitude
