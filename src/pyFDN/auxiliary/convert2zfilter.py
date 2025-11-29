import numpy as np

from pyFDN.auxiliary.zscalar import ZScalar
from pyFDN.auxiliary.zfir import ZFIR
from pyFDN.auxiliary.zfilter import ZFilter


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
