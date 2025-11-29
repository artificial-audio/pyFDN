from __future__ import annotations

import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike

from pyFDN.auxiliary.mgrpdelay import mgrpdelay
from pyFDN.auxiliary.outer_sum_approximation import outer_sum_approximation

def matrix_delay_approximation(matrix: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Rank-1 approximation of matrix group delay."""

    GD, _ = mgrpdelay(matrix)
    GD[np.isinf(GD)] = np.nan
    matrix_delay = np.nanmean(GD, axis=2)

    gdl, gdr = outer_sum_approximation(matrix_delay)
    approximation = gdl + gdr
    approximation_error = gdl[:, np.newaxis] + gdr[np.newaxis, :] - matrix_delay
    return approximation, approximation_error
