import numpy as np
from scipy.signal import firwin2

from pyFDN.auxiliary.RT602slope import RT602slope
from pyFDN.auxiliary.db2mag import db2mag
from pyFDN.auxiliary.hertz2unit import hertz2unit


def absorptionFilters(frequency, targetRT60, filterOrder, delays, fs):
    """
    Generate FIR absorption filters for each channel.
    frequency: [freq_points]
    targetRT60: shape (freq_points, channels)
    delays: array of length channels
    """
    num_channels = len(delays)
    unit_freq = hertz2unit(frequency, fs)
    FIR = np.zeros((num_channels, filterOrder + 1))

    if filterOrder == 0:
        rt60 = targetRT60[0, :]
        db = delays * RT602slope(rt60, fs)
        FIR = db2mag(db)[:, None]
    else:
        for ch in range(num_channels):
            rt60 = targetRT60[:, ch]
            delay = delays[ch] + int(np.ceil(filterOrder / 2))
            db = delay * RT602slope(rt60, fs)
            target_amp = db2mag(db)
            # firwin2 expects normalized [0..1] freqs and gain values
            FIR[ch, :] = firwin2(filterOrder + 1, unit_freq, target_amp)
    return FIR
