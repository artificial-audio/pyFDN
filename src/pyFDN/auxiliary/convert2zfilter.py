import numpy as np

# ---------------- Placeholder Classes ---------------- #
class ZFilter:
    """Base class for zFilter objects."""
    def __init__(self):
        pass

class ZScalar(ZFilter):
    """Wraps a numeric matrix as a zFilter."""
    def __init__(self, mat):
        super().__init__()
        self.mat = np.array(mat)
        self.m, self.n = self.mat.shape

    def filter(self, x):
        """Matrix multiplication for filtering."""
        return x @ self.mat.T

class ZFIR(ZFilter):
    """Wraps a numeric 1D array (FIR) as a zFilter."""
    def __init__(self, coeffs):
        super().__init__()
        self.coeffs = np.array(coeffs)

    def filter(self, x):
        """1D FIR filtering along axis 0 (time)."""
        from scipy.signal import lfilter
        return lfilter(self.coeffs, [1.0], x, axis=0)


# ---------------- convert2zFilter ---------------- #
def convert2zFilter(m):
    """
    Convert numeric input to ZFilter object if needed.
    
    Parameters
    ----------
    m : ndarray, list, or ZFilter
        Numeric matrix/array or ZFilter object.

    Returns
    -------
    zF : ZFilter
        Wrapped ZFilter object (ZScalar or ZFIR) or the input if already ZFilter.
    """
    if isinstance(m, (np.ndarray, list)):
        m = np.array(m)
        if m.ndim == 2:
            zF = ZScalar(m)
        else:
            zF = ZFIR(m)
    elif isinstance(m, ZFilter):
        zF = m
    else:
        raise TypeError("Type not defined for convert2zFilter")
    
    return zF
