"""Integration tests for complete recursive DSP systems."""

import pytest
import torch
import numpy as np
from pyFDN.recursive import (
    DelayRead, DelayWrite, Biquads,
    FeedbackMix, InputTap, OutputTap, RecursionCore
)


class TestSimpleSystems:
    """Test simple complete systems."""
    
    def test_pure_delay_system(self):
        """Test system with only delay (no feedback or filtering)."""
        delay_length = 16
        num_lines = 1
        
        stages = [
            DelayRead(delay_length=delay_length, num_lines=num_lines),
            InputTap(input_matrix=torch.eye(num_lines)),  # Add input to lines
            DelayWrite(),
            OutputTap(num_lines=num_lines, num_outputs=num_lines),
        ]
        core = RecursionCore(stages)
        
        # Impulse input
        input_signal = torch.zeros(64, num_lines)
        input_signal[0, :] = 1.0
        
        output = core.process(input_signal, block_size=8)  # block_size < delay_length
        
        # With InputTap, output[0] includes direct input + zero delay = 1
        # output[delay_length] includes input + delayed input = 0 + 1 = 1
        assert torch.allclose(output[0], torch.ones(num_lines), atol=1e-6)
        assert torch.allclose(output[delay_length], torch.ones(num_lines), atol=1e-6)
    
    def test_simple_feedthrough(self):
        """Test system with direct path (no delay loop)."""
        num_lines = 2
        
        # Simple system: input goes through delay, gets output via output tap
        C = torch.ones(1, num_lines) / num_lines  # Average lines
        
        stages = [
            DelayRead(delay_length=16, num_lines=num_lines),
            InputTap(input_matrix=torch.eye(num_lines, 1).repeat(1, 1)),  # Feed input to first line only
            DelayWrite(),
            OutputTap(
                output_matrix=C,
                num_lines=num_lines,
                num_outputs=1,
            ),
        ]
        core = RecursionCore(stages)
        
        input_signal = torch.randn(32, 1)
        output = core.process(input_signal, block_size=8)
        
        # Output includes contribution from lines (which include delayed + current input)
        # Just verify system runs and produces finite output
        assert torch.all(torch.isfinite(output))
        assert output.shape == (32, 1)


class TestFeedbackComb:
    """Test feedback comb filter systems."""
    
    def test_feedback_comb_filter(self):
        """Test simple feedback comb filter."""
        delay_length = 16  # Increased to be > block_size
        feedback_gain = 0.7
        
        # Single line with feedback
        stages = [
            DelayRead(delay_length=delay_length, num_lines=1),
            FeedbackMix(feedback_matrix=torch.tensor([[feedback_gain]])),
            InputTap(input_matrix=torch.ones(1, 1)),
            DelayWrite(),
            OutputTap(num_lines=1, num_outputs=1),
        ]
        core = RecursionCore(stages)
        
        # Impulse input
        input_signal = torch.zeros(100, 1)
        input_signal[0, 0] = 1.0
        
        output = core.process(input_signal, block_size=8)  # block_size < delay_length
        
        # Expected impulse response: impulses at 0, delay, 2*delay, 3*delay, ...
        # with amplitudes 1, g, g^2, g^3, ...
        expected_peaks = [0, delay_length, 2*delay_length, 3*delay_length]
        expected_amps = [1.0, feedback_gain, feedback_gain**2, feedback_gain**3]
        
        for peak_idx, expected_amp in zip(expected_peaks[:3], expected_amps[:3]):
            if peak_idx < len(output):
                actual_amp = output[peak_idx, 0].item()
                assert abs(actual_amp - expected_amp) < 0.1, \
                    f"Peak at {peak_idx}: expected {expected_amp}, got {actual_amp}"
    
    def test_comb_with_block_size_invariance(self):
        """Test that different block sizes give same result."""
        delay_length = 32  # Large enough for various block sizes
        feedback_gain = 0.5
        
        stages_template = lambda: [
            DelayRead(delay_length=delay_length, num_lines=1),
            FeedbackMix(feedback_matrix=torch.tensor([[feedback_gain]])),
            InputTap(input_matrix=torch.ones(1, 1)),
            DelayWrite(),
            OutputTap(num_lines=1, num_outputs=1),
        ]
        
        input_signal = torch.randn(64, 1)
        
        # Process with different block sizes (all < delay_length)
        core_4 = RecursionCore(stages_template())
        output_4 = core_4.process(input_signal.clone(), block_size=4)
        
        core_8 = RecursionCore(stages_template())
        output_8 = core_8.process(input_signal.clone(), block_size=8)
        
        core_16 = RecursionCore(stages_template())
        output_16 = core_16.process(input_signal.clone(), block_size=16)
        
        # All should give same result
        assert torch.allclose(output_4, output_8, atol=1e-5)
        assert torch.allclose(output_8, output_16, atol=1e-5)


class TestFDNSystems:
    """Test FDN-like multi-line reverb systems."""
    
    def test_basic_fdn(self):
        """Test basic FDN structure."""
        num_lines = 4
        delay_length = 16
        
        # Create Householder feedback matrix (unitary)
        v = torch.randn(num_lines, 1)
        v = v / torch.norm(v)
        A = torch.eye(num_lines) - 2 * v @ v.T
        
        stages = [
            DelayRead(delay_length=delay_length, num_lines=num_lines),
            FeedbackMix(feedback_matrix=A),
            InputTap(
                input_matrix=torch.ones(num_lines, 1),
                num_lines=num_lines,
                num_inputs=1
            ),
            DelayWrite(),
            OutputTap(
                output_matrix=torch.ones(1, num_lines) / num_lines,
                num_lines=num_lines,
                num_outputs=1
            ),
        ]
        core = RecursionCore(stages)
        
        input_signal = torch.randn(128, 1)
        output = core.process(input_signal, block_size=32)
        
        assert output.shape == (128, 1)
        # FDN should not blow up (unitary matrix preserves energy)
        assert torch.all(torch.isfinite(output))
        assert output.abs().max() < 100  # Reasonable bound
    
    def test_fdn_with_absorption_inside_loop(self):
        """Test FDN with absorption filters inside feedback loop."""
        num_lines = 4
        delay_length = 16
        
        # Lowpass absorption (0.9 one-pole)
        absorption_coeffs = torch.tensor([[[1.0, -0.9, 0.0, 0.1, 0.0, 0.0]]]).repeat(num_lines, 1, 1)  # [a0, a1, a2, b0, b1, b2]
        
        # Hadamard matrix for feedback
        A = torch.tensor([
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1]
        ], dtype=torch.float32) * 0.5
        
        stages = [
            DelayRead(delay_length=delay_length, num_lines=num_lines),
            FeedbackMix(feedback_matrix=A),
            Biquads(num_lines=num_lines, biquad_coeffs=absorption_coeffs),
            InputTap(input_matrix=torch.ones(num_lines, 1)),
            DelayWrite(),
            OutputTap(output_matrix=torch.ones(1, num_lines) / num_lines),
        ]
        core = RecursionCore(stages)
        
        # Impulse response
        input_signal = torch.zeros(200, 1)
        input_signal[0, 0] = 1.0
        
        output = core.process(input_signal, block_size=8)  # block_size < delay_length
        
        # Should decay over time due to absorption
        # Check that overall energy decreases from early to very late
        early_energy = output[16:48].pow(2).sum()
        late_energy = output[150:182].pow(2).sum()
        # Just verify some decay happens (absorption affects the system)
        assert torch.all(torch.isfinite(output))
        assert late_energy < early_energy  # Some decay
    
    def test_fdn_with_absorption_outside_loop(self):
        """Test FDN with absorption filters after feedback loop."""
        num_lines = 4
        delay_length = 16
        
        # Lowpass absorption (0.8 one-pole)
        absorption_coeffs = torch.tensor([[[1.0, -0.8, 0.0, 0.2, 0.0, 0.0]]]).repeat(num_lines, 1, 1)  # [a0, a1, a2, b0, b1, b2]
        
        A = torch.eye(num_lines) * 0.9  # Simple diagonal feedback
        
        stages = [
            DelayRead(delay_length=delay_length, num_lines=num_lines),
            FeedbackMix(feedback_matrix=A),
            InputTap(input_matrix=torch.ones(num_lines, 1)),
            DelayWrite(),
            Biquads(num_lines=num_lines, biquad_coeffs=absorption_coeffs),
            OutputTap(output_matrix=torch.ones(1, num_lines) / num_lines),
        ]
        core = RecursionCore(stages)
        
        input_signal = torch.zeros(200, 1)
        input_signal[0, 0] = 1.0
        
        output = core.process(input_signal, block_size=32)
        
        # Should still have some decay (absorption affects output only)
        assert torch.all(torch.isfinite(output))
        assert output.abs().max() < 100


class TestStageOrdering:
    """Test that stage ordering affects behavior correctly."""
    
    def test_absorption_position_affects_output(self):
        """Test that moving absorption filter changes the sound."""
        num_lines = 2
        delay_length = 8
        
        # Strong lowpass
        absorption_coeffs = torch.tensor([[[1.0, -0.5, 0.0, 0.5, 0.0, 0.0]]]).repeat(num_lines, 1, 1)  # [a0, a1, a2, b0, b1, b2]
        A = torch.eye(num_lines) * 0.8
        
        # Version 1: Absorption inside loop
        stages_inside = [
            DelayRead(delay_length=delay_length, num_lines=num_lines),
            FeedbackMix(feedback_matrix=A),
            Biquads(num_lines=num_lines, biquad_coeffs=absorption_coeffs),
            InputTap(input_matrix=torch.ones(num_lines, 1)),
            DelayWrite(),
            OutputTap(output_matrix=torch.ones(1, num_lines) / num_lines),
        ]
        
        # Version 2: Absorption outside loop
        stages_outside = [
            DelayRead(delay_length=delay_length, num_lines=num_lines),
            FeedbackMix(feedback_matrix=A),
            InputTap(input_matrix=torch.ones(num_lines, 1)),
            DelayWrite(),
            Biquads(num_lines=num_lines, biquad_coeffs=absorption_coeffs),
            OutputTap(output_matrix=torch.ones(1, num_lines) / num_lines),
        ]
        
        input_signal = torch.randn(64, 1)
        
        core_inside = RecursionCore(stages_inside)
        output_inside = core_inside.process(input_signal.clone(), block_size=16)
        
        core_outside = RecursionCore(stages_outside)
        output_outside = core_outside.process(input_signal.clone(), block_size=16)
        
        # Outputs should be different
        assert not torch.allclose(output_inside, output_outside, atol=0.01)


class TestBatchProcessing:
    """Test batch processing with multiple parallel signals."""
    
    def test_batch_independence(self):
        """Test that batch elements are processed independently."""
        num_lines = 2
        delay_length = 8
        batch_size = 3
        
        stages = [
            DelayRead(delay_length=delay_length, num_lines=num_lines),
            FeedbackMix(feedback_matrix=torch.eye(num_lines) * 0.5),
            InputTap(input_matrix=torch.ones(num_lines, 1)),
            DelayWrite(),
            OutputTap(output_matrix=torch.ones(1, num_lines) / num_lines),
        ]
        
        # Process batch: [B, N, T]
        core_batch = RecursionCore(stages)
        input_batch = torch.randn(batch_size, 1, 64)  # [B, N_in, T]
        output_batch = core_batch.process(input_batch, block_size=16)
        
        # Process individually
        for i in range(batch_size):
            core_single = RecursionCore([
                DelayRead(delay_length=delay_length, num_lines=num_lines),
                FeedbackMix(feedback_matrix=torch.eye(num_lines) * 0.5),
                InputTap(input_matrix=torch.ones(num_lines, 1)),
                DelayWrite(),
                OutputTap(output_matrix=torch.ones(1, num_lines) / num_lines),
            ])
            output_single = core_single.process(input_batch[i:i+1], block_size=16)  # [1, N, T]
            
            # Should match: [B, N_out, T]
            assert torch.allclose(output_batch[i:i+1], output_single, atol=1e-6)
