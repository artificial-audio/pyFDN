#!/usr/bin/env python3
"""
Non-regression test for coupled rooms example.

what this tests:
- The tinyRotationMatrix translation produces orthogonal matrices
- The coupled rooms FDN runs without errors
- The output has expected properties (shape, magnitude, etc.)
- Results are reproducible across runs
"""

import os
import sys

import numpy as np
import torch

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pyFDN.examples.example_coupled_rooms import (
    create_coupled_rooms_fdn,
    tiny_rotation_matrix,
)


def test_tiny_rotation_matrix():
    """Test the tinyRotationMatrix translation."""
    print("Testing tinyRotationMatrix translation...")

    # set reproducible seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Test parameters
    n = 6
    delta = 12
    spread = 0.1

    # generate rotation matrix
    R = tiny_rotation_matrix(n, delta, spread)

    # first test, check it's actually a matrix of correct size
    assert R.shape == (n, n), f"Expected shape ({n}, {n}), got {R.shape}"

    # 2nd test check orthogonality (R^T * R should be identity)
    orthogonality_check = torch.matmul(R.T, R)
    identity = torch.eye(n)
    max_deviation = torch.max(torch.abs(orthogonality_check - identity))

    print(f"  Max deviation from orthogonality: {max_deviation:.2e}")
    assert max_deviation < 1e-5, f"Matrix not orthogonal, max deviation: {max_deviation}"

    # third test, check determinant is ±1 (should be +1 for rotation matrix)
    det = torch.det(R)
    print(f"  Determinant: {det:.6f}")
    assert abs(abs(det) - 1) < 1e-4, f"Determinant should be ±1, got {det}"

    # fourth tes, reproducibility. same seed should give same result
    torch.manual_seed(42)
    np.random.seed(42)
    R2 = tiny_rotation_matrix(n, delta, spread)

    max_diff = torch.max(torch.abs(R - R2))
    print(f"  Reproducibility check: {max_diff:.2e}")
    assert max_diff < 1e-15, f"Results not reproducible, max difference: {max_diff}"

    # fifth test, different seeds should give different results
    torch.manual_seed(123)
    np.random.seed(123)
    R3 = tiny_rotation_matrix(n, delta, spread)

    max_diff = torch.max(torch.abs(R - R3))
    print(f"  Different seed produces different result: {max_diff:.6f}")
    assert max_diff > 1e-6, "Different seeds should produce different matrices"

    print("  ✓ tinyRotationMatrix tests passed")


def test_coupled_rooms_fdn():
    """Test the complete coupled rooms FDN example."""
    print("\nTesting coupled rooms FDN...")

    # set reproducible seeds
    torch.manual_seed(5)
    np.random.seed(5)

    # run the example
    ir, fs, feedback_matrix, delays = create_coupled_rooms_fdn()

    # Check impulse response properties
    assert isinstance(ir, np.ndarray), "IR should be numpy array"
    print(f"  IR shape: {ir.shape}")

    # The actual length depends on nfft size used in the implementation
    # For testing, just check it's a reasonable length (should be at least a few thousand samples)
    assert ir.shape[0] >= 8192, f"IR too short: {ir.shape[0]} samples"
    assert ir.shape[0] <= 100000, f"IR too long: {ir.shape[0]} samples"

    # Should be stereo (2 channels)
    assert ir.shape[1] == 2, f"Expected 2 channels, got {ir.shape[1]}"

    # Check IR magnitude is reasonable
    max_amp = np.max(np.abs(ir))
    print(f"  Max amplitude: {max_amp:.6f}")
    assert 0.001 < max_amp < 10, f"IR amplitude seems unreasonable: {max_amp}"

    # check feedback matrix is orthogonal
    assert isinstance(feedback_matrix, np.ndarray), "Feedback matrix should be numpy array"
    assert feedback_matrix.shape == (12, 12), f"Expected 12x12 feedback matrix, got {feedback_matrix.shape}"

    ortho_check = feedback_matrix.T @ feedback_matrix
    max_deviation = np.max(np.abs(ortho_check - np.eye(12)))
    print(f"  Feedback matrix orthogonality: {max_deviation:.2e}")
    assert max_deviation < 2e-5, f"Feedback matrix not orthogonal: {max_deviation}"

    # check delay values
    assert isinstance(delays, np.ndarray), "Delays should be numpy array"
    assert len(delays) == 12, f"Expected 12 delays, got {len(delays)}"
    assert np.all(delays > 0), "All delays should be positive"
    print(f"  Delay range: {delays.min():.0f} - {delays.max():.0f} samples")

    # Check for NaN or inf values
    assert not np.any(np.isnan(ir)), "IR contains NaN values"
    assert not np.any(np.isinf(ir)), "IR contains infinite values"
    assert not np.any(np.isnan(feedback_matrix)), "Feedback matrix contains NaN values"
    assert not np.any(np.isinf(feedback_matrix)), "Feedback matrix contains infinite values"

    print("  ✓ Coupled rooms FDN tests passed")


def test_reproducibility():
    """Test that the complete example is reproducible."""
    print("\nTesting reproducibility...")

    # Run 1
    torch.manual_seed(5)
    np.random.seed(5)
    ir1, fs1, fb1, delays1 = create_coupled_rooms_fdn()

    # Run 2 with same seed
    torch.manual_seed(5)
    np.random.seed(5)
    ir2, fs2, fb2, delays2 = create_coupled_rooms_fdn()

    # Check all outputs are identical
    ir_diff = np.max(np.abs(ir1 - ir2))
    fb_diff = np.max(np.abs(fb1 - fb2))
    delay_diff = np.max(np.abs(delays1 - delays2))

    print(f"  IR difference: {ir_diff:.2e}")
    print(f"  Feedback matrix difference: {fb_diff:.2e}")
    print(f"  Delay difference: {delay_diff:.2e}")

    assert ir_diff < 1e-15, f"IR not reproducible: {ir_diff}"
    assert fb_diff < 1e-15, f"Feedback matrix not reproducible: {fb_diff}"
    assert delay_diff < 1e-15, f"Delays not reproducible: {delay_diff}"
    assert fs1 == fs2, "Sample rate not reproducible"

    print("  ✓ Reproducibility tests passed")


def test_energy_decay():
    """Test that the impulse response shows proper energy decay."""
    print("\nTesting energy decay characteristics...")

    torch.manual_seed(5)
    np.random.seed(5)
    ir, fs, _, _ = create_coupled_rooms_fdn()

    # Compute energy decay curve (Schroeder integral)
    for channel in range(ir.shape[1]):
        energy = np.cumsum(ir[::-1, channel]**2)[::-1]
        energy_db = 10 * np.log10(energy / (energy[0] + 1e-12))

        # Check that energy generally decreases
        # Allow some fluctuation but overall trend should be downward
        mid_point = len(energy_db) // 2
        end_point = int(len(energy_db) * 0.9)  # 90% of the way through

        decay_amount = energy_db[0] - energy_db[end_point]
        print(f"  Channel {channel+1} decay amount: {decay_amount:.1f} dB")

        # Should have at least 10dB of decay (relaxed due to shorter IR length in test)
        assert decay_amount > 10, f"Insufficient decay in channel {channel+1}: {decay_amount:.1f} dB"

        # Should not decay too fast (not more than 80dB in first half)
        mid_decay = energy_db[0] - energy_db[mid_point]
        assert mid_decay < 80, f"Too fast decay in channel {channel+1}: {mid_decay:.1f} dB"

    print("  ✓ Energy decay tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Coupled Rooms Non-Regression Tests")
    print("=" * 60)

    try:
        test_tiny_rotation_matrix()
        test_coupled_rooms_fdn()
        test_reproducibility()
        test_energy_decay()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("The coupled rooms implementation is working correctly.")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit(main())
