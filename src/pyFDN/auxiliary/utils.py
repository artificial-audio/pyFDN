"""General utility functions."""
from __future__ import annotations
import warnings
import numpy as np
from numpy.typing import ArrayLike
from numpy.linalg import svd
from scipy.interpolate import interp1d
from scipy.signal import freqz, group_delay

def ensure_3d(matrix: ArrayLike) -> np.ndarray:
    """Ensure the matrix has a trailing polynomial dimension."""

    arr = np.asarray(matrix)
    if arr.ndim == 2:
        return arr[:, :, np.newaxis]
    if arr.ndim == 3:
        return arr
    raise ValueError("Expected a 2-D or 3-D array for polynomial matrices")


def last_nonzero_indices(mat: np.ndarray) -> np.ndarray:
    """Return 1-based indices of the last non-zero element along axis 2."""

    arr = ensure_3d(mat)
    nonzero = np.abs(arr) > 0
    if not np.any(nonzero):
        return np.zeros(arr.shape[:2], dtype=int)
    reversed_nonzero = nonzero[:, :, ::-1]
    first_true = np.argmax(reversed_nonzero, axis=2)
    has_nonzero = np.any(nonzero, axis=2)
    last = np.zeros_like(first_true)
    last[has_nonzero] = arr.shape[2] - first_true[has_nonzero]
    return last


def mag2db(magnitude: ArrayLike) -> np.ndarray:
    """Convert magnitudes to decibels with numerical guard."""

    mag = np.asarray(magnitude, dtype=float)
    tiny = np.finfo(float).tiny
    return 20.0 * np.log10(np.maximum(np.abs(mag), tiny))


def db2mag(db: ArrayLike) -> np.ndarray:
    """Convert decibel values to linear magnitude."""

    db_arr = np.asarray(db, dtype=float)
    return np.power(10.0, db_arr / 20.0)


def hertz2unit(hz: ArrayLike, fs: float) -> np.ndarray:
    """Convert frequency (Hz) to normalized frequency (0–1)."""
    return np.asarray(hz) / fs * 2


def is_bounding_curve(x_points, y_points, x_curve, y_curve, bound_type):
    """
    Check if all value points are bounded by the curve.
    Args:
        x_points: x-coordinates of data points (1D array)
        y_points: y-coordinates of data points (1D array)
        x_curve: x-coordinates of curve points (1D array)
        y_curve: y-coordinates of curve points (1D array)
        bound_type: 'upper' or 'lower'
    Returns:
        all_bounded: bool, whether all data points are bounded
        is_bounded: boolean array, whether each data point is bounded
    """
    # Spline interpolation with extrapolation
    interp = interp1d(x_curve, y_curve, kind="cubic", fill_value="extrapolate")
    y_curve_interp = interp(x_points)

    if bound_type == "upper":
        is_bounded = y_curve_interp >= y_points
    elif bound_type == "lower":
        is_bounded = y_curve_interp <= y_points
    else:
        raise ValueError("bound_type must be 'upper' or 'lower'")

    all_bounded = np.all(is_bounded)
    return all_bounded, is_bounded


def pole_boundaries(delays, absorption, feedback_matrix, fs, nfft=2**12):
    """
    Find upper and lower pole boundaries for FDN loop.
    Args:
        delays: 1D array of delays in samples (length N)
        absorption: object with .b and .a attributes, each shape (N, 1, len)
        feedback_matrix: 3D numpy array (N, N, len)
        fs: sampling frequency
        nfft: number of frequency bins (default: 4096)
    Returns:
        MinCurve: lower bound of pole magnitude (shape: nfft)
        MaxCurve: upper bound of pole magnitude (shape: nfft)
        f: frequency points (Hz, shape: nfft)
    """
    N = len(delays)
    # Compute frequency points
    w = np.linspace(0, np.pi, nfft)
    # FFT along the third axis
    FeedbackMatrix = np.fft.fft(feedback_matrix, n=nfft * 2, axis=2)
    FeedbackMatrix = FeedbackMatrix[:, :, :nfft]

    Min = np.zeros(nfft)
    Max = np.zeros(nfft)
    for it in range(nfft):
        s = svd(FeedbackMatrix[:, :, it], compute_uv=False)
        Min[it] = np.min(np.abs(s)) ** (1 / np.min(delays))
        Max[it] = np.max(np.abs(s)) ** (1 / np.max(delays))

    # Combine with absorption
    b = np.transpose(absorption.b, (0, 2, 1))  # shape (N, len, 1)
    a = np.transpose(absorption.a, (0, 2, 1))  # shape (N, len, 1)
    b = b.squeeze(-1)  # shape (N, len)
    a = a.squeeze(-1)  # shape (N, len)

    H = np.zeros((nfft, N), dtype=complex)
    G = np.zeros((nfft, N))
    for it in range(N):
        # freqz expects (b, a) as 1D arrays
        H[:, it], w = freqz(b[it, :], a[it, :], nfft)
        # group_delay returns (w, gd)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _, gd = group_delay((b[it, :], a[it, :]), nfft)
        G[:, it] = gd

    # delays: shape (N,)
    # G: shape (nfft, N)
    # d: shape (nfft, N)
    d = np.abs(H) ** (1.0 / (delays + G))
    dMin = np.min(d, axis=1)
    dMax = np.max(d, axis=1)

    MinCurve = dMin * Min
    MaxCurve = dMax * Max
    f = w / np.pi * fs / 2

    return MinCurve, MaxCurve, f

