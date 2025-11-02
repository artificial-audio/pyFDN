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


def polyder_rational(b: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derivative of rational polynomial using quotient rule."""
    # Remove leading zeros
    b = np.trim_zeros(b, 'f')
    a = np.trim_zeros(a, 'f')
    
    if len(b) == 0:
        b = np.array([0.0])
    if len(a) == 0:
        a = np.array([1.0])
    
    # Compute derivatives of numerator and denominator
    db = np.polyder(b) if len(b) > 1 else np.array([0.0])
    da = np.polyder(a) if len(a) > 1 else np.array([0.0])
    
    # Apply quotient rule: (b/a)' = (b'*a - b*a') / a^2
    if len(db) == 0:
        db = np.array([0.0])
    if len(da) == 0:
        da = np.array([0.0])
        
    num1 = np.convolve(db, a)
    num2 = np.convolve(b, da)
    
    # Pad to same length
    max_len = max(len(num1), len(num2))
    if len(num1) < max_len:
        num1 = np.pad(num1, (max_len - len(num1), 0))
    if len(num2) < max_len:
        num2 = np.pad(num2, (max_len - len(num2), 0))
    
    q = num1 - num2
    p = np.convolve(a, a)
    
    # Remove leading zeros from result
    q = np.trim_zeros(q, 'f')
    p = np.trim_zeros(p, 'f')
    
    if len(q) == 0:
        q = np.array([0.0])
    if len(p) == 0:
        p = np.array([1.0])
    
    return q, p


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


def det_polynomial(polynomial_matrix: np.ndarray, var: str) -> np.ndarray:
    """
    Determinant of a polynomial matrix.
    
    Args:
        polynomial_matrix: Polynomial Matrix of size [N, N, degree]
        var: Either 'z^1' or 'z^-1'
        
    Returns:
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