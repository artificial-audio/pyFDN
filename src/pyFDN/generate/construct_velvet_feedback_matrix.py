from __future__ import annotations
from typing import Tuple
import numpy as np

from pyFDN.generate.construct_cascaded_paraunitary_matrix import construct_cascaded_paraunitary_matrix

def construct_velvet_feedback_matrix(n: int, stages: int, sparsity: float) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper for ``construct_cascaded_paraunitary_matrix`` using Hadamard stages."""

    return construct_cascaded_paraunitary_matrix(n, stages, sparsity=sparsity, matrix_type="Hadamard")
