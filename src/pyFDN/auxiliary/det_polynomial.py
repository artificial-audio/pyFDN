from __future__ import annotations
import numpy as np

from pyFDN.auxiliary.poly_degree import poly_degree

def det_polynomial(polynomial_matrix: np.ndarray, var: str) -> np.ndarray:
    """
    Determinant of a polynomial matrix.
    
    Args:
        polynomial_matrix: numpy array of shape (N, N, degree) containing the polynomial coefficients
        var: 'z^1' or 'z^-1'    Returns:
        determinant: Determinant polynomial
    """
    tol_db = -200
    N = polynomial_matrix.shape[1]
    length = polynomial_matrix.shape[2]
    fft_size = length * N
    
    # Computation
    if var == 'z^-1':
        freq_mat = np.fft.fft(polynomial_matrix, fft_size, axis=2)
    elif var == 'z^1':
        freq_mat = np.fft.fft(np.flip(polynomial_matrix, axis=2), fft_size, axis=2)
    else:
        raise ValueError('Variable type not defined')
    
    freq_det = np.zeros(fft_size, dtype=complex)
    for it in range(fft_size):
        freq_det[it] = np.linalg.det(freq_mat[:, :, it])
    
    determinant = np.fft.ifft(freq_det, fft_size).real
    determinant = determinant[:-(N-1)]
    
    # Shorten the determinant numerically
    if var == 'z^-1':
        degree = poly_degree(determinant, var, tol_db)
        determinant = determinant[:degree+1]
    elif var == 'z^1':
        determinant = np.flip(determinant)
        degree = poly_degree(determinant, var, tol_db)
        determinant = determinant[-degree-1:] if degree >= 0 else determinant
    
    return determinant
