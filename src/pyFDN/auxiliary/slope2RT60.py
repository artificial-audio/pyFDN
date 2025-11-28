def slope2RT60(slope, fs):
    """Convert slope (dB/sample) to T60 in seconds."""
    return -60.0 / (slope * fs)