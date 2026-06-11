"""Acoustics and RT60 related functions."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import firwin2, freqz
from scipy.special import erfc

from pyFDN.auxiliary.utils import db_to_lin, hertz_to_unit, lin_to_db


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


def estimate_rt_bands(
    ir: ArrayLike,
    fs: float,
    fc: float = 1000.0,
    start: float = -4.0,
    n: int = 8,
    filter_order: int = 8,
    decay_db: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate RT60 in octave bands via Butterworth bandpass filtering.

    Filters the impulse response into octave bands using
    ``pyroomacoustics.bandpass_filterbank``, then estimates RT60 per band
    using ``pyroomacoustics.measure_rt60`` (extrapolated from ``decay_db``).

    Default bands: 63, 125, 250, 500, 1000, 2000, 4000, 8000 Hz (``start=-4, n=8``).
    Bands whose upper edge exceeds ``fs/2`` are dropped.

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
        Decay range in dB used for the linear fit (default 30, i.e. RT30→RT60).

    Returns
    -------
    rt : (n_bands,) ndarray
        Estimated RT60 in seconds per band.
    f_centre : (n_bands,) ndarray
        Centre frequencies in Hz corresponding to each RT value.
    """
    try:
        import pyroomacoustics as pra
    except ImportError as exc:
        raise ImportError(
            "estimate_rt_bands requires pyroomacoustics (pip install pyroomacoustics)"
        ) from exc

    from scipy.signal import sosfilt

    ir = np.asarray(ir, dtype=float).ravel()
    bands, f_centre = pra.octave_bands(fc=fc, start=start, n=n)

    # drop bands whose upper edge is at or above Nyquist
    valid = bands[:, 1] < fs / 2
    bands = bands[valid]
    f_centre = f_centre[valid]

    sos_bank = pra.bandpass_filterbank(bands, fs=fs, order=filter_order, output="sos")
    rt = np.zeros(len(f_centre))
    for k, sos in enumerate(sos_bank):
        ir_band = sosfilt(sos, ir)
        rt[k] = pra.measure_rt60(ir_band, fs=fs, decay_db=decay_db)

    return rt, f_centre


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
    matches the total band energy: ``E = L^2 * T * fs / (6 ln 10)``, hence
    ``L = sqrt(6 ln(10) E / (T fs))``. This replaces the DecayFitNet
    initial-level estimate used in the MATLAB ``example_RIR2FDN``.

    Parameters
    ----------
    ir : array-like, 1-D
        Impulse response, starting at the onset.
    rt : array-like
        RT60 in seconds per band, as returned by :func:`estimate_rt_bands`
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
    try:
        import pyroomacoustics as pra
    except ImportError as exc:
        raise ImportError(
            "estimate_initial_level_bands requires pyroomacoustics "
            "(pip install pyroomacoustics)"
        ) from exc

    from scipy.signal import sosfilt

    ir = np.asarray(ir, dtype=float).ravel()
    rt = np.asarray(rt, dtype=float).ravel()
    bands, f_centre = pra.octave_bands(fc=fc, start=start, n=n)

    valid = bands[:, 1] < fs / 2
    bands = bands[valid]
    f_centre = f_centre[valid]
    if rt.size != len(f_centre):
        raise ValueError("rt must have one entry per octave band")

    sos_bank = pra.bandpass_filterbank(bands, fs=fs, order=filter_order, output="sos")
    level = np.zeros(len(f_centre))
    for k, sos in enumerate(sos_bank):
        ir_band = sosfilt(sos, ir)
        energy = np.sum(ir_band**2)
        level[k] = np.sqrt(6.0 * np.log(10.0) * energy / (rt[k] * fs))

    return level, f_centre


def one_pole_absorption(
    rt_dc: float, rt_ny: float, delays: ArrayLike, fs: float
) -> np.ndarray:
    """Design one-pole absorption filters according to specified reverb time.

    Returns SOS format: shape (6, N) with [b0, b1, b2, a0, a1, a2] per channel.
    """
    delays_arr = np.asarray(delays, dtype=float)

    # Calculate target gains
    slope_dc = rt_to_slope(rt_dc, fs)
    slope_ny = rt_to_slope(rt_ny, fs)

    # Convert to linear magnitude
    h_dc = db_to_lin(delays_arr * slope_dc)
    h_ny = db_to_lin(delays_arr * slope_ny)

    # Design filters
    r = h_dc / h_ny
    a1 = (1.0 - r) / (1.0 + r)
    b0 = (1.0 - a1) * h_ny

    num_filters = h_dc.size
    sos = np.zeros((6, num_filters))
    sos[0, :] = b0  # b0
    sos[3, :] = 1.0  # a0
    sos[4, :] = a1  # a1

    return sos


def sos_gain_per_sample_curves(
    sos_6n: np.ndarray,
    delays: ArrayLike,
    nfft: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude response (gain per sample vs angle) for SOS filters in (6, N) format.

    Evaluates :math:`|H(e^{j\\omega})|` at nfft angles from 0 to :math:`2\\pi` for each
    channel, then scales by delay length so that the result is gain per sample: for
    channel j with delay m_j, the curve is :math:`|H|^{1/m_j}`, so that after m_j
    samples the effective gain is :math:`|H|`. Useful for plotting absorption/gain
    curves (e.g. on a pole plot).

    Parameters
    ----------
    sos_6n : (6, N) array
        SOS coefficients with rows [b0, b1, b2, a0, a1, a2] per column (channel).
        Same format as :func:`one_pole_absorption` returns.
    delays : (N,) array-like
        Delay lengths in samples, one per channel. Used to scale gain to per-sample.
    nfft : int
        Number of frequency points (default 512).

    Returns
    -------
    angles : (nfft,) array
        Angles in rad/sample, 0 to 2*pi.
    magnitude : (nfft, N) array
        Gain per sample (linear), i.e. :math:`|H(e^{j\\omega})|^{1/m}` per channel.
    """
    sos_6n = np.asarray(sos_6n, dtype=np.float64)
    delays_arr = np.asarray(delays, dtype=np.float64).ravel()
    if sos_6n.shape[0] != 6:
        raise ValueError("sos_6n must have shape (6, N)")
    N = sos_6n.shape[1]
    if delays_arr.shape[0] != N:
        raise ValueError("delays must have length N (number of channels in sos_6n)")
    if np.any(delays_arr < 1):
        raise ValueError("delays must be >= 1")
    magnitude = np.zeros((nfft, N), dtype=np.float64)
    for ch in range(N):
        b = sos_6n[0:3, ch]
        a = sos_6n[3:6, ch]
        angles, h = freqz(b, a, worN=nfft)
        mag = np.abs(h)
        magnitude[:, ch] = np.power(mag, 1.0 / delays_arr[ch])
    return angles, magnitude
