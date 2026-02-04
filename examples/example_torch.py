"""
Recursive DSP Framework Demonstration
======================================

This script demonstrates the usage of the recursive DSP framework
with various example systems.
"""

import torch
import matplotlib.pyplot as plt
from pyFDN.recursive import (
    DelayRead,
    DelayWrite,
    Biquads,
    FeedbackMix,
    InputTap,
    OutputTap,
    RecursionCore,
    build_ffm_stages,
)


def create_impulse(length: int, num_channels: int = 1) -> torch.Tensor:
    """Create an impulse signal."""
    signal = torch.zeros(length, num_channels)
    signal[0, :] = 1.0
    return signal


def plot_signal(signal: torch.Tensor, title: str, max_samples: int = 500):
    """Plot a (potentially multichannel) signal."""
    plt.figure(figsize=(12, 4))
    signal_np = signal.detach().cpu().numpy()  # shape [samples, channels] or [samples]
    if signal_np.ndim == 1:
        # Single channel
        plt.plot(signal_np[:max_samples])
    else:
        # Multiple channels: plot each with its own label
        num_channels = signal_np.shape[1]
        for ch in range(num_channels):
            plt.plot(signal_np[:max_samples, ch], label=f"Channel {ch+1}")
        plt.legend()
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def example_1_pure_delay():
    """Example 1: Simple pure delay (no feedback)."""
    print("\n" + "="*60)
    print("Example 1: Pure Delay System")
    print("="*60)
    
    delay_lengths = [30, 40]
    num_lines = 2
    
    stages = [
        DelayRead(delay_lengths=torch.tensor(delay_lengths, dtype=torch.long), num_lines=num_lines),
        OutputTap(output_matrix=torch.eye(num_lines, num_lines)),
        InputTap(num_lines=num_lines, num_inputs=1),
        DelayWrite(),
    ]
    
    core = RecursionCore(stages, block_size=30)
    print(f"\nSystem structure:\n{core}")
    
    # Process impulse
    input_signal = create_impulse(500, 1)
    output = core.process(input_signal)
    
    print(f"\nInput shape: {input_signal.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Delay length: {delay_lengths} samples")
    
    plot_signal(output, f"Pure Delay - {delay_lengths} samples")
    return output


def example_2_feedback_comb():
    """Example 2: Feedback comb filter."""
    print("\n" + "="*60)
    print("Example 2: Feedback Comb Filter")
    print("="*60)
    
    delay_lengths = [64]
    feedback_gain = 0.7
    
    stages = [
        DelayRead(delay_lengths=torch.tensor(delay_lengths, dtype=torch.long), num_lines=1),
        OutputTap(output_matrix=torch.eye(1, 1)),
        InputTap(num_lines=1, num_inputs=1),
        FeedbackMix(feedback_matrix=torch.tensor([[feedback_gain]])),
        DelayWrite(),
    ]
    
    core = RecursionCore(stages, block_size=64)
    print(f"\nSystem structure:\n{core}")
    
    # Process impulse
    input_signal = create_impulse(500, 1)
    output = core.process(input_signal)
    
    print(f"\nDelay: {delay_lengths} samples")
    print(f"Feedback gain: {feedback_gain}")
    print(f"Expected periodicity: {delay_lengths} samples")
    
    plot_signal(output, f"Feedback Comb Filter (delay={delay_lengths}, g={feedback_gain})")
    return output


def example_3_fdn_absorption():
    """Example 3: FDN with absorption inside feedback loop."""
    print("\n" + "="*60)
    print("Example 3: FDN with Absorption Inside Loop")
    print("="*60)
    
    num_lines = 4
    delay_lengths = [64, 80, 100, 121]
    
    # Create Hadamard feedback matrix
    A = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=torch.float32)
    
    # Lowpass absorption filters (one-pole, a=0.85)
    # Format: [a0, a1, a2, b0, b1, b2] where a0=1.0, a1=-0.85, b0=0.15
    absorption_coeffs = torch.zeros(num_lines, 1, 6)
    absorption_coeffs[0, 0] = torch.tensor([1.0, -0.85, 0.0, 0.15, 0.0, 0.0])
    for i in range(1, num_lines):
        absorption_coeffs[i, 0] = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # Only a0=1 for other lines

    stages = [
        DelayRead(delay_lengths=torch.tensor(delay_lengths, dtype=torch.long), num_lines=num_lines),
        OutputTap(output_matrix=torch.eye(num_lines, num_lines)),
        InputTap(input_matrix=torch.Tensor([[1.0], [0.0], [0.0], [0.0]])),
        Biquads(num_lines=num_lines, biquad_coeffs=absorption_coeffs),
        FeedbackMix(feedback_matrix=A),
        DelayWrite(),
    ]
    
    core = RecursionCore(stages, block_size=50)
    print(f"\nSystem structure:\n{core}")
    
    # Process impulse
    input_signal = create_impulse(1000, 1)
    output = core.process(input_signal)
    
    print(f"\nNumber of delay lines: {num_lines}")
    print(f"Delay length: {delay_lengths} samples")
    print(f"Feedback matrix: circular shift matrix")
    print(f"Absorption: One-pole lowpass (a=0.85)")
    
    plot_signal(output, "FDN with Absorption Inside Loop", max_samples=1000)
    return output


def example_4_block_size_comparison():
    """Example 4: Verify block size invariance."""
    print("\n" + "="*60)
    print("Example 4: Block Size Invariance")
    print("="*60)
    
    num_lines = 2
    delay_lengths = [300, 400]
    
    stages_template = lambda: [
        DelayRead(delay_lengths=torch.tensor(delay_lengths, dtype=torch.long), num_lines=num_lines),
        FeedbackMix(feedback_matrix=torch.eye(num_lines) * 0.7),
        InputTap(input_matrix=torch.eye(num_lines, 1)),
        DelayWrite(),
        OutputTap(output_matrix=torch.eye(1, num_lines)),
    ]
    
    input_signal = torch.randn(2048, 1)
    
    block_sizes = [8, 32, 64, 256]
    outputs = {}
    
    for bs in block_sizes:
        core = RecursionCore(stages_template(), block_size=bs)
        output = core.process(input_signal.clone())
        outputs[bs] = output
        print(f"\nBlock size {bs:3d}: Output shape {output.shape}")
    
    # Compare outputs
    print("\nComparing outputs (max absolute difference):")
    reference = outputs[block_sizes[0]]
    for bs in block_sizes[1:]:
        diff = (outputs[bs] - reference).abs().max().item()
        print(f"  Block size {block_sizes[0]} vs {bs}: {diff:.2e}")
    
    return outputs


def example_5_ffm_lossless():
    """Example 5: Lossless cascaded Filter Feedback Matrix (FFM) inside the loop."""
    print("\n" + "=" * 60)
    print("Example 5: Lossless Cascaded FFM (no attenuation)")
    print("=" * 60)

    torch.manual_seed(0)

    num_lines = 4
    main_delays = [61, 89, 101, 127]

    # Orthonormal mixing matrices U_k (Hadamard + two random orthogonals).
    U0 = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ],
        dtype=torch.float32,
    ) * 0.5

    def random_orthogonal(n: int) -> torch.Tensor:
        a = torch.randn(n, n, dtype=torch.float32)
        q, r = torch.linalg.qr(a)
        d = torch.sign(torch.diag(r))
        d[d == 0] = 1.0
        return q * d

    U1 = random_orthogonal(num_lines)
    U2 = random_orthogonal(num_lines)

    # Diagonal per-line delays m_k (can be smaller than block_size; includes 0).
    m1 = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    m2 = torch.tensor([0, 4, 8, 12], dtype=torch.long)

    ffm_stages = build_ffm_stages([U0, U1, U2], [m1, m2], state_prefix="ffm")

    # Input/output: excite all lines and tap average of delay outputs (v) with direct path.
    B = torch.ones(num_lines, 1, dtype=torch.float32) / num_lines
    C = torch.ones(1, num_lines, dtype=torch.float32) / num_lines
    D = torch.ones(1, 1, dtype=torch.float32)

    stages = [
        DelayRead(delay_lengths=torch.tensor(main_delays, dtype=torch.long), num_lines=num_lines, state_key="delay"),
        OutputTap(output_matrix=C, direct_matrix=D),
        InputTap(input_matrix=B),
        *ffm_stages,
        DelayWrite(state_key="delay"),
    ]

    core = RecursionCore(stages, block_size=32)
    print(f"\nSystem structure:\n{core}")
    print(f"\nMain delays (samples): {main_delays}")
    print(f"FFM diagonal delays m1: {m1.tolist()}")
    print(f"FFM diagonal delays m2: {m2.tolist()}")
    print("Note: This is lossless (no attenuation). Add a scalar gain or absorption filters to get decay.")

    input_signal = create_impulse(2000, 1)
    output = core.process(input_signal)
    plot_signal(output, "Lossless Cascaded FFM (tapped at delay outputs)", max_samples=1200)
    return output


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("RECURSIVE DSP FRAMEWORK DEMONSTRATION")
    print("="*60)
    
    # Run examples
    try:
        example_1_pure_delay()
        example_2_feedback_comb()
        example_3_fdn_absorption()
        example_4_block_size_comparison()
        example_5_ffm_lossless()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
        # Show all plots
        plt.show()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
