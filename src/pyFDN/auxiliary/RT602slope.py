def RT602slope(RT60, fs):
    """Convert T60 in seconds to slope in dB/sample."""
    return -60.0 / (RT60 * fs)
