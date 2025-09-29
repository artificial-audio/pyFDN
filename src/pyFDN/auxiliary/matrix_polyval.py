"""Matrix polynomial operations."""
from __future__ import annotations
import numpy as np

def matrix_polyval(P: np.ndarray, z: complex | np.ndarray) -> np.ndarray:
    """
    Evaluate matrix polynomial at z.
    
    Args:
        P: Polynomial matrix [N, M, FIR]
        z: Evaluation point [scalar]
        
    Returns:
        Y: Output matrix [N, M]
    """
    degree = P.shape[2]
    exponents = np.arange(degree - 1, -1, -1)
    z_powers = np.power(z, exponents)
    z_powers = z_powers.reshape(1, 1, -1)
    
    Y = np.sum(P * z_powers, axis=2)
    return Y