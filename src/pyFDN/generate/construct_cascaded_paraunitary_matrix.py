from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.linalg import hadamard
import math

from pyFDN.auxiliary.matrix_convolution import matrix_convolution
from pyFDN.generate.random_orthogonal import random_orthogonal
from pyFDN.generate.shift_matrix import shift_matrix
from pyFDN.generate.shift_matrix_distribute import shift_matrix_distribute
from pyFDN.helpers.utils import ensure_3d

def construct_cascaded_paraunitary_matrix(
    n: int,
    k: int,
    *,
    sparsity: float = 1.0,
    matrix_type: str = "Hadamard",
    gain_per_sample: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a paraunitary matrix and its reverse response."""

    matrix_type_lower = matrix_type.lower()
    if matrix_type_lower == "hadamard":
        if n & (n - 1) != 0:
            raise ValueError("Hadamard construction requires n to be a power of two")
        generate_matrix = lambda size: hadamard(size) / math.sqrt(size)
    elif matrix_type_lower == "random":
        generate_matrix = random_orthogonal
    else:
        raise ValueError("matrix_type must be 'Hadamard' or 'random'")

    sparsity_vector = np.concatenate(([sparsity], np.ones(max(k - 1, 0))))
    matrix = ensure_3d(generate_matrix(n))
    rev_matrix = ensure_3d(np.linalg.inv(matrix[:, :, 0]))

    pulse_size = 1
    for stage in range(k):
        shift_left = shift_matrix_distribute(matrix, sparsity_vector[stage], pulse_size=pulse_size)
        gain_diag = np.diag(np.power(gain_per_sample, shift_left))
        R1 = ensure_3d(generate_matrix(n) @ gain_diag)

        matrix = shift_matrix(matrix, shift_left, "left")
        matrix = matrix_convolution(R1, matrix)

        rev_matrix = shift_matrix(rev_matrix, shift_left, "right")
        R1_inv = ensure_3d(np.linalg.inv(R1[:, :, 0]))
        rev_matrix = matrix_convolution(rev_matrix, R1_inv)

        pulse_size = max(int(pulse_size * n * sparsity_vector[stage]), 1)

    return matrix, rev_matrix
