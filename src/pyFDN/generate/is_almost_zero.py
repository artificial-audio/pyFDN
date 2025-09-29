from __future__ import annotations
import numpy as np

def is_almost_zero(A: np.ndarray, tol: float = 1e-12) -> bool:
    """
    Test whether matrix/vector is almost zero in absolute values.
    
    Args:
        A: Numerical values to be tested
        tol: Tolerance value for max deviation from 0
        
    Returns:
        isZ: Boolean whether all values in A are almost 0
    """
    max_val = np.max(np.abs(A))
    return max_val < tol