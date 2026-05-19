"""
Tiny rotation matrix generator for FDN feedback matrices.

Translation of tinyRotationMatrix.m from fdnToolbox.
Original MATLAB code: (c) Sebastian Jiro Schlecht, 2020
Python translation: Facundo Franchino, 2025
"""

import torch


def tiny_rotation_matrix(
    n: int,
    delta: float,
    spread: float = 0.1,
    log_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Generate orthogonal matrix with small eigenvalue angles.

    Creates a rotation matrix suitable for use in FDN feedback structures,
    where small eigenvalue angles help control the density of the impulse
    response.

    Args:
        n: Matrix size
        delta: Mean normalized eigenvalue angle
        spread: Spreading of eigenvalue angle (default 0.1)
        log_matrix: Initial logarithm matrix (default random)

    Returns:
        rotation_matrix: Orthogonal matrix of shape (n, n)

    Example:
        >>> torch.manual_seed(42)
        >>> R = tiny_rotation_matrix(6, 12)
        >>> R.shape
        torch.Size([6, 6])
        >>> torch.allclose(R @ R.T, torch.eye(6), atol=1e-5)
        True
    """
    if log_matrix is None:
        log_matrix = torch.randn(n, n)

    # Generate skew symmetric matrix
    skew_symmetric = (log_matrix - log_matrix.T) / 2

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eig(skew_symmetric)

    # Pair each eigenvalue with its complex-conjugate partner. A real
    # skew-symmetric matrix has purely imaginary eigenvalues in conjugate
    # pairs (plus a possible real/zero eigenvalue when n is odd). Matching on
    # eigenvalues is exact and phase-independent; matching eigenvectors is not,
    # because torch.linalg.eig returns eigenvectors with an arbitrary phase.
    conj_distances = torch.abs(
        eigenvalues.unsqueeze(1) - torch.conj(eigenvalues).unsqueeze(0)
    )
    idx = torch.argmin(conj_distances, dim=1)

    # Generate frequency spread
    frequency_spread = 2 * (torch.rand(n) - 0.5) * spread + 1

    # Average spreads for conjugate pairs
    frequency_spread = (frequency_spread[idx] + frequency_spread) / 2

    # Detect real eigenvalues: where IDX[i] == i (self-conjugate)
    # MATLAB: frequencySpread( IDX - 1:N == 0 ) = 0
    # This translates to: idx == torch.arange(n)
    real_eigenval_mask = idx == torch.arange(n)
    frequency_spread[real_eigenval_mask] = 0

    # Scale and spread eigenvalues
    eigenvals_normalized = eigenvalues / (torch.abs(eigenvalues) + 1e-12)
    new_eigenvals = eigenvals_normalized * frequency_spread * delta * torch.pi

    # Create new skew-symmetric matrix
    new_skew = torch.real(
        eigenvectors @ torch.diag(new_eigenvals) @ eigenvectors.conj().T
    )
    new_skew = (new_skew - new_skew.T) / 2

    # Matrix exponential to get orthogonal matrix
    rotation_matrix = torch.matrix_exp(new_skew)

    return rotation_matrix
