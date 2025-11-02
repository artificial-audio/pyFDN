import numpy as np

from pyFDN.auxiliary.mag2db import mag2db
from pyFDN.auxiliary.slope_to_rt60 import slope_to_rt60

def absorption_to_t60(filterCoeffs, delays, nfft, fs):
    """Compute T60 from recursive absorption filter with delay."""
    filterLen = filterCoeffs.shape[1]
    response = np.fft.fft(filterCoeffs, nfft, axis=1)
    freq = np.linspace(0, fs, nfft)

    response = response[:, :nfft // 2]
    freq = freq[:nfft // 2]

    totalDelay = delays[:, None] + filterLen / 2
    decayPerSample = mag2db(np.abs(response)) / totalDelay
    T60 = slope_to_rt60(decayPerSample, fs)
    return T60.T, freq  # shape: (freq_points, channels)
