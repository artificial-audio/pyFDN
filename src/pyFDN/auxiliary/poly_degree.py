from __future__ import annotations
import numpy as np

def poly_degree(polynomial: np.ndarray, var: str, tol_db: float = -200) -> int:
    """
    Polynomial degree with tolerance and exponent sign.
    
    Args:
        polynomial: Vector of polynomial coefficients
        var: Either 'z^1' or 'z^-1'
        tol_db: Tolerance in dB
        
    Returns:
        deg: Degree of the polynomial
    """
    poly_db = 20 * np.log10(np.abs(polynomial) + np.finfo(float).eps)
    max_coefficient = np.max(poly_db)
    
    if var == 'z^-1':
        valid_indices = np.where((poly_db - max_coefficient) > tol_db)[0]
        deg = valid_indices[-1] if len(valid_indices) > 0 else 0
    elif var == 'z^1':
        valid_indices = np.where((poly_db - max_coefficient) > tol_db)[0]
        deg = len(polynomial) - valid_indices[0] if len(valid_indices) > 0 else 0
    else:
        raise ValueError('Variable type not defined')
    
    return deg