from __future__ import annotations
import numpy as np


def matrix_convolution(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Matrix polynomial multiplication by convolution.
    
    Args:
        A: Matrix polynomial of size [m, n, order]
        B: Matrix polynomial of size [n, k, order]
        
    Returns:
        C: Matrix polynomial of size [m, k, order] with C(z) = A(z)*B(z)
    """
    sz_A = A.shape
    sz_B = B.shape
    
    if sz_A[1] != sz_B[0]:
        raise ValueError('Invalid matrix dimension.')
    
    C = np.zeros((sz_A[0], sz_B[1], sz_A[2] + sz_B[2] - 1))
    
    A_perm = np.transpose(A, (2, 0, 1))
    B_perm = np.transpose(B, (2, 0, 1))
    C_perm = np.transpose(C, (2, 0, 1))
    
    for row in range(sz_A[0]):
        for col in range(sz_B[1]):
            for it in range(sz_A[1]):
                C_perm[:, row, col] += np.convolve(A_perm[:, row, it], B_perm[:, it, col])
    
    C = np.transpose(C_perm, (1, 2, 0))
    return C