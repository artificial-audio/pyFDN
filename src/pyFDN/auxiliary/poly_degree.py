import numpy as np


def mag2db(x):
    return 20 * np.log10(np.maximum(np.abs(x), np.finfo(float).eps))

def poly_degree(polynomial, var, tol=None):
    """
    Determine the degree of a polynomial with tolerance and exponent sign.
    Args:
        polynomial: 1D numpy array of polynomial coefficients
        var: 'z^1' or 'z^-1'
        tol: tolerance in dB (optional, default: mag2db(eps))
    Returns:
        deg: degree of the polynomial
    """
    if tol is None:
        tol = mag2db(np.finfo(float).eps)

    poly_db = mag2db(np.abs(polynomial))
    max_coefficient = np.max(poly_db)

    if var == 'z^-1':
        indices = np.where((poly_db - max_coefficient) > tol)[0]
        deg = indices[-1] if len(indices) > 0 else 0
    elif var == 'z^1':
        indices = np.where((poly_db - max_coefficient) > tol)[0]
        deg = len(polynomial) - indices[0] - 1 if len(indices) > 0 else 0
    else:
        raise ValueError("var must be 'z^1' or 'z^-1'")

    return deg

