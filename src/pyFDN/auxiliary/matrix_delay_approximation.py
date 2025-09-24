import numpy as np

from pyFDN.auxiliary.mgrpdelay import mgrpdelay
from pyFDN.auxiliary.outer_sum_approximation import outer_sum_approximation

def matrix_delay_approximation(matrix):
    """
    Rank 1 approximation of matrix group delay.
    
    Args:
        matrix (ndarray): 3D filter matrix (N, M, FIR)
        
    Returns:
        approximation (ndarray): Approximation of group delay in matrix (M, N)
        approximation_error (ndarray): Error matrix of same shape as matrixDelay
    """
    # Compute group delay of each filter in matrix
    GD, _ = mgrpdelay(matrix)
    
    # Replace infinities with NaN
    GD = np.where(np.isinf(GD), np.nan, GD)
    
    # Average over the third dimension (FIR) ignoring NaNs
    matrix_delay = np.nanmean(GD, axis=2)  # shape (N, M)
    
    # Rank-1 approximation of the group delay
    gdl, gdr = outer_sum_approximation(matrix_delay)
    
    # Approximation matrix
    approximation = (gdl[:, None] + gdr[None, :]).T  # Transpose to match MATLAB output
    
    # Approximation error
    approximation_error = (gdr[None, :] + gdl[:, None]) - matrix_delay

    return approximation, approximation_error
