"""Tests for train features."""

import pytest

pytest.importorskip("torch")

import torch
from pyFDN import mimo_rir_eigenvalues_per_frequency
from pyFDN import energy_decay_curve



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


def test_energy_decay_curve_exponential():
    """An exponential decay must produce a linear downward slope in dB."""
    fs = 48000.0
    t = torch.arange(48000, dtype=torch.float32) / fs
    rt60 = 1.0

    # Synthetic decaying response: 60 dB drop in 1.0 second
    decay_rate = 3.0 * torch.log(torch.tensor(10.0)) / rt60
    ir = torch.exp(-decay_rate * t).unsqueeze(-1).unsqueeze(-1)

    edc = energy_decay_curve(ir, dim=0, db=True, normalize=True)

    # Initial value must be 0 dB
    assert float(edc[0, 0, 0]) == pytest.approx(0.0, abs=1e-4)

    # EDC must be monotonically non-increasing
    diffs = edc[1:, 0, 0] - edc[:-1, 0, 0]
    assert torch.all(diffs <= 1e-6)

    # At t = 0.5 s, the decay should be approximately -30 dB
    half_sec_idx = int(0.5 * fs)
    assert float(edc[half_sec_idx, 0, 0]) == pytest.approx(-30.0, abs=1.0)


def test_energy_decay_curve_linear_mode():
    """Verify linear mode (db=False, normalize=False) returns raw integrated power."""
    # Unit impulse at t=0
    ir = torch.zeros(100, 1, 1, dtype=torch.float32)
    ir[0, 0, 0] = 2.0  # Energy = 4.0

    edc = energy_decay_curve(ir, dim=0, db=False, normalize=False)

    # The entire tail before and at t=0 contains 4.0; after t=0 it is 0.0
    assert float(edc[0, 0, 0]) == pytest.approx(4.0, abs=1e-6)
    assert float(edc[1, 0, 0]) == pytest.approx(0.0, abs=1e-6)