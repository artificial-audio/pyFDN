"""Acoustics and RT60 related functions."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import firwin2, sosfreqz
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

    Returns a one-section per-channel SOS bank of shape ``(1, 6, N)`` (the
    canonical SOS bank layout; section rows are ``[b0, b1, b2, a0, a1, a2]``).
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
    sos = np.zeros((1, 6, num_filters))
    sos[0, 0, :] = b0  # b0
    sos[0, 3, :] = 1.0  # a0
    sos[0, 4, :] = a1  # a1

    return sos


def first_order_absorption(
    rt_dc: float,
    rt_ny: float,
    delays: ArrayLike,
    fs: float,
    crossover_frequency: float | None = None,
) -> np.ndarray:
    """Design first-order shelving absorption filters according to specified reverb time.

    Each delay line gets a first-order shelving filter whose gain matches the
    target decay (rt_dc at DC, rt_ny at Nyquist) for its delay length, with the
    shelf transition at crossover_frequency.

    Reference: Jot, J. M., "Proportional parametric equalizers - Application to
    digital reverberation and environmental audio processing", AES 2015.

    Parameters
    ----------
    rt_dc : float
        Reverberation time in seconds at DC.
    rt_ny : float
        Reverberation time in seconds at Nyquist.
    delays : array-like
        Delay lengths in samples, one per channel.
    fs : float
        Sampling rate in Hz.
    crossover_frequency : float, optional
        Shelf crossover frequency in Hz. Defaults to fs/8, the midpoint of the
        warped (bilinear) frequency axis. Values above fs/5 are clamped to fs/5
        since a too high crossover leads to an unstable filter (fs/4 is the limit).

    Returns
    -------
    np.ndarray
        One-section per-channel SOS bank of shape ``(1, 6, N)`` (the canonical
        SOS bank layout); section rows are ``[b0, b1, b2, a0, a1, a2]``
        (b2 = a2 = 0 for these first-order filters).
    """
    delays_arr = np.asarray(delays, dtype=float)

    h_dc = db_to_lin(delays_arr * rt_to_slope(rt_dc, fs))
    h_ny = db_to_lin(delays_arr * rt_to_slope(rt_ny, fs))
    return _first_order_shelf(h_dc, h_ny, fs, crossover_frequency)


def _first_order_shelf(
    h_dc: np.ndarray,
    h_ny: np.ndarray,
    fs: float,
    crossover_frequency: float | None = None,
) -> np.ndarray:
    """First-order shelving SOS bank from target linear gains at DC and Nyquist.

    Shared core of :func:`first_order_absorption` and
    :func:`first_order_shelving_eq`. ``h_dc`` and ``h_ny`` are linear-magnitude
    gains (one per channel); returns a one-section per-channel SOS bank of shape
    ``(1, 6, N)``.
    """
    h_dc = np.asarray(h_dc, dtype=float)
    h_ny = np.asarray(h_ny, dtype=float)

    if crossover_frequency is None:
        crossover_frequency = fs / 8.0
    crossover_frequency = min(crossover_frequency, fs / 5.0)
    omega = crossover_frequency / fs * 2.0 * np.pi

    t = np.tan(omega)
    sqrt_k = np.sqrt(h_dc / h_ny)

    b0 = (t * sqrt_k + 1.0) * h_ny
    b1 = (t * sqrt_k - 1.0) * h_ny
    a0 = t / sqrt_k + 1.0
    a1 = t / sqrt_k - 1.0

    sos = np.zeros((1, 6, h_dc.size))
    sos[0, 0, :] = b0 / a0
    sos[0, 1, :] = b1 / a0
    sos[0, 3, :] = 1.0
    sos[0, 4, :] = a1 / a0
    return sos


def first_order_shelving_eq(
    db_dc: ArrayLike,
    db_nyquist: ArrayLike,
    fs: float,
    crossover_frequency: float | None = None,
) -> np.ndarray:
    """Design first-order shelving EQ filters from gains in dB at DC and Nyquist.

    Unlike :func:`first_order_absorption` (whose gains are derived from a
    reverberation time and a delay length), the shelf endpoints are specified
    directly as decibel gains. Useful as a per-output tone correction (post EQ).

    Parameters
    ----------
    db_dc : array-like
        Gain in dB at DC, scalar or one value per channel.
    db_nyquist : array-like
        Gain in dB at Nyquist, scalar or one value per channel. Broadcast
        against ``db_dc`` to a common number of channels.
    fs : float
        Sampling rate in Hz.
    crossover_frequency : float, optional
        Shelf crossover frequency in Hz. Defaults to fs/8; clamped to fs/5.

    Returns
    -------
    np.ndarray
        One-section per-channel SOS bank of shape ``(1, 6, N)`` (canonical SOS
        bank layout); section rows are ``[b0, b1, b2, a0, a1, a2]``.
    """
    db_dc_arr, db_ny_arr = np.broadcast_arrays(
        np.asarray(db_dc, dtype=float).ravel(),
        np.asarray(db_nyquist, dtype=float).ravel(),
    )
    return _first_order_shelf(
        db_to_lin(db_dc_arr), db_to_lin(db_ny_arr), fs, crossover_frequency
    )


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
