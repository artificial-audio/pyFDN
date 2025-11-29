from __future__ import annotations
import numpy as np

from pyFDN.auxiliary.polyder_rational import polyder_rational


def negpolyder(b: np.ndarray, a: np.ndarray, dont_truncate: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Derivative of rational polynomial with negative exponents.
    
    Args:
        b: Numerator coefficients
        a: Denominator coefficients
        dont_truncate: Leading zeros are not truncated
        
    Returns:
        q: Numerator coefficients of derivative
        p: Denominator coefficients of derivative
    """
    # Flip for substitution x = z^-1
    b_flip = np.flip(b)
    a_flip = np.flip(a)
    
    # Compute derivative
    q, p = polyder_rational(b_flip, a_flip)
    
    # Flip for back substitution x^-1 = z
    q = np.flip(q)
    p = np.flip(p)
    
    # Multiply with -1/z^2
    q = np.convolve(q, np.array([0, 0, -1]))
    
    # Restore full length if truncation is not desired
    if dont_truncate:
        qq = np.zeros(len(a) + len(b) - 1)
        pp = np.zeros(len(a) + len(a) - 1)
        qq[:len(q)] = q
        pp[:len(p)] = p
        q = qq
        p = pp
    
    return q, p
