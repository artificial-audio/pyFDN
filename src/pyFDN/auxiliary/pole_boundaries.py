import numpy as np
from numpy.linalg import svd
from scipy.signal import freqz, group_delay


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
    FeedbackMatrix = np.fft.fft(feedback_matrix, n=nfft*2, axis=2)
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
