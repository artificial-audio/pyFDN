import numpy as np

def matrix_convolution(A, B):
    """
    Matrix polynomial multiplication (convolution along the last axis).
    A: shape (m, n, orderA)
    B: shape (n, k, orderB)
    Returns:
        C: shape (m, k, orderA + orderB - 1)
    """
    szA = A.shape
    szB = B.shape

    if szA[1] != szB[0]:
        raise ValueError("Invalid matrix dimension.")

    m, n, orderA = szA
    n2, k, orderB = szB
    orderC = orderA + orderB - 1

    C = np.zeros((m, k, orderC), dtype=A.dtype)

    # Permute to (order, m, n) for easier indexing
    A_perm = np.transpose(A, (2, 0, 1))
    B_perm = np.transpose(B, (2, 0, 1))

    for row in range(m):
        for col in range(k):
            for it in range(n):
                # Convolve the polynomials for (row, it) and (it, col)
                conv_result = np.convolve(A_perm[:, row, it], B_perm[:, it, col])
                C[row, col, :len(conv_result)] += conv_result

    return C
