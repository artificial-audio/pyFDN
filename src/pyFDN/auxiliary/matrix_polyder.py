"""Matrix polynomial operations."""
from __future__ import annotations
import numpy as np

from pyFDN.auxiliary.polyder_rational import polyder_rational
from pyFDN.auxiliary.negpolyder import negpolyder


def matrix_polyder(B: np.ndarray, A: np.ndarray, var: str = 'z^1') -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper function for polynomial derivative of filter matrices.
    
    Args:
        B: Numerator [FIR, N, M]
        A: Denominator [FIR, N, M]
        var: Variable type {'z^1', 'z^-1'}
        
    Returns:
        Q: Numerator of derivative
        P: Denominator of derivative
    """
    Q = np.zeros((1, B.shape[1], B.shape[2]))
    P = np.zeros((1, A.shape[1], A.shape[2]))
    
    for it1 in range(B.shape[1]):
        for it2 in range(B.shape[2]):
            if var == 'z^1':
                q, p = polyder_rational(B[:, it1, it2], A[:, it1, it2])
                Q[0, it1, it2] = q[0] if len(q) > 0 else 0
                P[0, it1, it2] = p[0] if len(p) > 0 else 0
            elif var == 'z^-1':
                q, p = negpolyder(B[:, it1, it2], A[:, it1, it2])
                Q_len = min(len(q), Q.shape[0])
                P_len = min(len(p), P.shape[0])
                Q[:Q_len, it1, it2] = q[:Q_len]
                P[:P_len, it1, it2] = p[:P_len]
    
    return Q, P
