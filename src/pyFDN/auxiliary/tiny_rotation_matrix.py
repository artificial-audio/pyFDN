"""
Tiny rotation matrix generator for FDN feedback matrices.

Translation of tinyRotationMatrix.m from fdnToolbox.
Original MATLAB code: (c) Sebastian Jiro Schlecht, 2020
Python translation: Facundo Franchino, 2025
"""

import torch


def tiny_rotation_matrix(n, delta, spread=0.1, log_matrix=None):
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

    # Find nearest neighbors: for each eigenvector, find closest among conjugates
    # The MATLAB nearestneighbour(v, conj(v)) finds for each column of v,
    # the closest column in conj(v) using Euclidean distance

    # Calculate pairwise distances between eigenvectors and their conjugates
    # Each eigenvector is a column, so we transpose for proper broadcasting
    v_conj = torch.conj(eigenvectors)

    # Compute distance matrix: |v_i - conj(v_j)|^2 for all i,j
    distances = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            # Distance between eigenvector i and conjugate of eigenvector j
            diff = eigenvectors[:, i] - v_conj[:, j]
            distances[i, j] = torch.norm(diff).real

    # Find nearest neighbor index for each eigenvector
    idx = torch.argmin(distances, dim=1)

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
