import numpy as np

def mag2db(x):
    return 20 * np.log10(np.maximum(x, 1e-20))  # avoid log(0)