import numpy as np

from pyFDN.auxiliary import poly_degree


def det_polynomial(polynomial_matrix, var):
    """
    Compute the determinant of a polynomial matrix.
    Args:
        polynomial_matrix: numpy array of shape (N, N, degree) containing the polynomial coefficients
        var: 'z^1' or 'z^-1'
    Returns:
        determinant: 1D numpy array of the determinant polynomial
    """
    tol = -200  # in dB
    N, _, length = polynomial_matrix.shape
    fft_size = length * N

    # FFT along the polynomial axis
    if var == "z^-1":
        freq_mat = np.fft.fft(polynomial_matrix, n=fft_size, axis=2)
    elif var == "z^1":
        freq_mat = np.fft.fft(np.flip(polynomial_matrix, axis=2), n=fft_size, axis=2)
    else:
        raise ValueError("var must be 'z^1' or 'z^-1'")

    # Compute determinant at each frequency bin
    freq_det = np.zeros(fft_size, dtype=complex)
    for it in range(fft_size):
        freq_det[it] = np.linalg.det(freq_mat[:, :, it])

    determinant = np.fft.ifft(freq_det, n=fft_size)
    determinant = determinant[: (len(determinant) - (N - 1))]

    # Shorten the determinant numerically
    if var == "z^-1":
        degree = poly_degree(determinant, var, tol)
        determinant = determinant[: degree + 1]
    elif var == "z^1":
        determinant = np.flipud(determinant)
        degree = poly_degree(determinant, var, tol)
        determinant = determinant[-(degree + 1) :]

    # If the result is complex but imaginary part is negligible, return real part
    if np.all(np.abs(determinant.imag) < 1e-10):
        determinant = determinant.real

    return determinant
