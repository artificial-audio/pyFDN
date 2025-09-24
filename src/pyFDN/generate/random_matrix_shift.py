import numpy as np

def shift_matrix(matrix, shifts, direction='left'):
    """
    Shift each row of a matrix by the corresponding value in shifts array.

    Args:
        matrix (ndarray): 2D array of shape (N, M)
        shifts (array-like): array of shifts of length N
        direction (str): 'left' or 'right'

    Returns:
        ndarray: shifted matrix
    """
    N, M = matrix.shape
    shifted = np.zeros_like(matrix)
    
    for i in range(N):
        s = shifts[i] % M  # Ensure shift does not exceed row length
        if direction == 'left':
            shifted[i] = np.roll(matrix[i], -s)
        elif direction == 'right':
            shifted[i] = np.roll(matrix[i], s)
        else:
            raise ValueError("Direction must be 'left' or 'right'")
    return shifted


def random_matrix_shift(max_shift, matrix, matrix_rev=None):
    """
    Shift polynomial matrix entries randomly in time.

    Args:
        max_shift (int): maximum shift in samples
        matrix (ndarray): 2D array (N x M)
        matrix_rev (ndarray, optional): reversed matrix to also shift

    Returns:
        tuple: (shifted_matrix, shifted_matrix_rev, rand_left_shift, rand_right_shift)
    """
    N = matrix.shape[0]

    if max_shift >= N:
        rand_left_shift = np.random.permutation(max_shift)[:N]
        rand_right_shift = np.random.permutation(max_shift)[:N]
    elif max_shift <= 0:
        rand_left_shift = np.zeros(N, dtype=int)
        rand_right_shift = np.zeros(N, dtype=int)
    else:
        rand_left_shift = np.random.randint(0, max_shift + 1, size=N)
        rand_right_shift = np.random.randint(0, max_shift + 1, size=N)

    # Normalize shifts to start from 0
    rand_left_shift -= rand_left_shift.min()
    rand_right_shift -= rand_right_shift.min()

    # Apply shifts
    matrix = shift_matrix(matrix, rand_left_shift, 'left')
    matrix = shift_matrix(matrix, rand_right_shift, 'right')

    if matrix_rev is not None:
        matrix_rev = shift_matrix(matrix_rev, rand_right_shift, 'left')
        matrix_rev = shift_matrix(matrix_rev, rand_left_shift, 'right')
        return matrix, matrix_rev, rand_left_shift, rand_right_shift

    return matrix, None, rand_left_shift, rand_right_shift
