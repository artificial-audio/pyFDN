"""Tests for train features."""

import pytest

pytest.importorskip("torch")

import torch
from pyFDN.train.features.spatial import mimo_rir_eigenvalues_per_frequency


def test_mimo_eigenvalues_identity_system():
    """An identity MIMO impulse response must have eigenvalues of 1 across all bins."""
    n_samples, n_out, n_in = 512, 3, 3
    n_fft = 512

    # Dirac-MIMO-System: H(z) = I
    ir = torch.zeros(n_samples, n_out, n_in, dtype=torch.float32)
    for ch in range(n_out):
        ir[0, ch, ch] = 1.0

    eigvals = mimo_rir_eigenvalues_per_frequency(ir, n_fft=n_fft)

    # Shape: (n_fft // 2 + 1, n_out)
    assert eigvals.shape == (n_fft // 2 + 1, n_out)
    assert eigvals.is_complex()

    # All Eigenvalues must be 1.0 + 0j
    expected = torch.ones((n_fft // 2 + 1, n_out), dtype=torch.complex64)
    torch.testing.assert_close(eigvals, expected, atol=1e-6, rtol=1e-6)


def test_mimo_eigenvalues_shape_and_fft_length():
    """Verify arbitrary FFT lengths and shape preservation."""
    n_samples, n_ch = 1024, 4
    n_fft = 2048
    ir = torch.randn(n_samples, n_ch, n_ch, dtype=torch.float32)

    eigvals = mimo_rir_eigenvalues_per_frequency(ir, n_fft=n_fft)
    assert eigvals.shape == (n_fft // 2 + 1, n_ch)
    assert torch.all(torch.isfinite(eigvals.real))
    assert torch.all(torch.isfinite(eigvals.imag))


def test_mimo_eigenvalues_rejects_non_square_channels():
    """Non-square channel geometries (n_out != n_in) must raise ValueError."""
    ir_rect = torch.randn(512, 3, 2, dtype=torch.float32)
    with pytest.raises(ValueError, match="Expected a square MIMO system"):
        mimo_rir_eigenvalues_per_frequency(ir_rect)


def test_mimo_eigenvalues_rejects_invalid_dimensions():
    """Tensors that are not 3D must raise ValueError."""
    ir_1d = torch.randn(512, dtype=torch.float32)
    with pytest.raises(ValueError, match="Expected a 3D tensor"):
        mimo_rir_eigenvalues_per_frequency(ir_1d)


def test_mimo_eigenvalues_rejects_non_float_or_invalid_type():
    """Non-tensor types must raise TypeError."""
    with pytest.raises(TypeError):
        mimo_rir_eigenvalues_per_frequency([1, 2, 3])