"""
Tiny rotation matrix generator for FDN feedback matrices.

Translation of tinyRotationMatrix.m from fdnToolbox.
Original MATLAB code: (c) Sebastian Jiro Schlecht, 2020
Python translation: Facundo Franchino, 2025
"""

import torch

from pyFDN.generate.random_orthogonal import random_orthogonal


def rotation_matrix_from_angles(
    angles: torch.Tensor,
    n: int | None = None,
) -> torch.Tensor:
    """
    Generate orthogonal matrix with prescribed eigenvalue angles.

    Builds a block-diagonal matrix of 2x2 Givens rotations, one block per
    angle, so the eigenvalues are exp(+-1j * angles). For odd matrix sizes,
    a single eigenvalue at 1 is appended.

    Args:
        angles: Eigenvalue angles in radians, one per conjugate pair
        n: Matrix size; either 2 * len(angles) or 2 * len(angles) + 1
            (default 2 * len(angles))

    Returns:
        rotation_matrix: Orthogonal matrix of shape (n, n)

    Example:
        >>> angles = torch.tensor([0.1, 0.2], dtype=torch.float64)
        >>> R = rotation_matrix_from_angles(angles, n=5)
        >>> R.shape
        torch.Size([5, 5])
        >>> torch.allclose(R @ R.T, torch.eye(5, dtype=R.dtype), atol=1e-12)
        True
    """
    angles = torch.as_tensor(angles).reshape(-1)
    if not angles.dtype.is_floating_point:
        angles = angles.to(torch.get_default_dtype())

    num_pairs = angles.numel()
    if n is None:
        n = 2 * num_pairs
    if n not in (2 * num_pairs, 2 * num_pairs + 1):
        raise ValueError(f"Matrix size n={n} requires {n // 2} angles, got {num_pairs}")

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rotation_matrix = torch.zeros(n, n, dtype=angles.dtype)
    for k in range(num_pairs):
        i = 2 * k
        rotation_matrix[i, i] = cos[k]
        rotation_matrix[i, i + 1] = -sin[k]
        rotation_matrix[i + 1, i] = sin[k]
        rotation_matrix[i + 1, i + 1] = cos[k]
    if n % 2 == 1:
        rotation_matrix[-1, -1] = 1

    return rotation_matrix


def tiny_rotation_matrix(
    n: int,
    delta: float,
    spread: float = 0.1,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Generate orthogonal matrix with small eigenvalue angles.

    Creates a rotation matrix suitable for use in FDN feedback structures,
    where small eigenvalue angles help control the density of the impulse
    response. The eigenvalue angles are delta * pi, randomly spread by the
    spread factor. The matrix is constructed from 2x2 Givens rotations with
    these angles, pre- and post-multiplied by a random orthogonal matrix so
    that the result is dense but keeps the prescribed eigenvalues. For odd
    matrix sizes, one eigenvalue is at 1.

    Args:
        n: Matrix size
        delta: Mean normalized eigenvalue angle
        spread: Spreading of eigenvalue angle (default 0.1)
        dtype: Floating point dtype of the result (default torch default)

    Returns:
        rotation_matrix: Orthogonal matrix of shape (n, n)

    Example:
        >>> R = tiny_rotation_matrix(6, 12)
        >>> R.shape
        torch.Size([6, 6])
        >>> torch.allclose(R @ R.T, torch.eye(6), atol=1e-5)
        True
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    num_pairs = n // 2
    frequency_spread = 2 * (torch.rand(num_pairs, dtype=dtype) - 0.5) * spread + 1
    angles = delta * torch.pi * frequency_spread

    givens = rotation_matrix_from_angles(angles, n)
    q = torch.from_numpy(random_orthogonal(n)).to(dtype)

    return q @ givens @ q.T
