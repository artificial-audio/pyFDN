from __future__ import annotations
import numpy as np

def polydiag(p: np.ndarray) -> np.ndarray:
    """
    Convert array of polynomials to diagonal polynomial matrix.
    
    Args:
        p: Array of polynomials [N, FIR] or [N, M, FIR]
        
    Returns:
        d: Diagonal polynomial matrix [N, N, FIR]
    """
    if p.ndim == 2:
        N, L = p.shape
    elif p.ndim == 3:
        N, M, L = p.shape
        # For 3D case, reshape to 2D
        p = p.reshape(N, -1)
        L = p.shape[1]
    else:
        raise ValueError(f"Unsupported number of dimensions: {p.ndim}")
    
    d = np.zeros((N, N, L))
    for i in range(N):
        d[i, i, :] = p[i, :]
    return d