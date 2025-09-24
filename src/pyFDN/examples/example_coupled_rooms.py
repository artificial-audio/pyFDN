"""
Translation of example_coupledRooms.m to Python using FLAMO and FLARE

This example demonstrates coupled room acoustics using Feedback Delay Networks (FDN).
Based on ideas from:
    Das, O., Abel, J. S. & Canfield-Dafilou, E. K. Delay Network
    Architectures For Room And Coupled Space Modeling. in Proceedings of
    the 23rdInternational Conference on Digital Audio Effects (DAFx2020)
    (2020).

Original MATLAB code: (c) Sebastian Jiro Schlecht, Monday, 7. December 2020
Python translation: Facundo Franchino, September 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import soundfile as sf

# FLAMO imports
from flamo.processor import dsp, system
from flamo.functional import signal_gallery

# Set random seed for reproducibility (matches MATLAB rng(5))
torch.manual_seed(5)
np.random.seed(5)

def create_coupled_rooms_fdn():
    """
    Create a coupled rooms FDN matching the MATLAB implementation exactly.
    
    Returns:
        ir: Impulse response (numpy array)
        fs: Sample rate
        feedback_matrix: The feedback matrix used
        delay_lengths: The delay lengths used
    """
    
    # Parameters (matching MATLAB exactly)
    fs = 48000
    impulse_response_length = fs * 2  # 2 seconds
    nfft = 16384  # Reduced for efficiency while testing
    
    # FDN configuration
    N = 12  # Total number of delay lines
    N_per_room = N // 2  # 6 delay lines per room
    num_input = 1
    num_output = 2
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    alias_decay_db = 0  # No anti-aliasing for exact reproduction
    
    # Exact delay values from MATLAB with rng(5)
    # These were captured from running the MATLAB code
    delays_room1 = torch.tensor([411, 736, 403, 760, 544, 606], dtype=torch.float32)
    delays_room2 = torch.tensor([2532, 2037, 1593, 1375, 1161, 2477], dtype=torch.float32)
    delay_lengths = torch.cat([delays_room1, delays_room2])
    
    print(f"Delay values: {delay_lengths.int().tolist()}")
    
    # Coupling parameter (exact from MATLAB)
    coupling = 0.3
    
    # Exact feedback matrices from MATLAB tinyRotationMatrix(6, 12)
    # These were captured from the MATLAB output
    A1 = torch.tensor([
        [-0.3317,  0.1721, -0.2895,  0.3210,  0.5599, -0.6001],
        [ 0.1655, -0.6906,  0.5125, -0.1606,  0.3038, -0.3392],
        [-0.7027, -0.4035,  0.1191,  0.4512, -0.3147,  0.1630],
        [ 0.3934, -0.5077, -0.6866,  0.2267, -0.2243, -0.1199],
        [ 0.1033,  0.2301,  0.2475,  0.0899, -0.6489, -0.6678],
        [ 0.4509,  0.1415,  0.3268,  0.7798,  0.1543,  0.1947]
    ], dtype=torch.float32)
    
    A2 = torch.tensor([
        [ 0.1120,  0.4493,  0.2820,  0.5641, -0.5814,  0.2231],
        [-0.5886,  0.1512, -0.0437,  0.5752,  0.5456,  0.0137],
        [-0.0576,  0.1039,  0.5691, -0.0287,  0.0054, -0.8131],
        [-0.5400, -0.6441, -0.0199,  0.1149, -0.5250, -0.0654],
        [ 0.5869, -0.5341, -0.0065,  0.5621,  0.1914, -0.1330],
        [-0.0408, -0.2536,  0.7708, -0.1448,  0.2278,  0.5167]
    ], dtype=torch.float32)
    
    print(f"\nA1 matrix (first 3x3):\n{A1[:3, :3]}")
    print(f"\nA2 matrix (first 3x3):\n{A2[:3, :3]}")
    
    # Use the exact square root matrices from MATLAB
    # These were captured from sqrtm(A1) and sqrtm(A2) in MATLAB
    A1_sqrt = torch.tensor([
        [ 0.4363, -0.3556, -0.0379,  0.0913,  0.7694, -0.2854],
        [ 0.6143,  0.2620,  0.5911, -0.1438, -0.2962, -0.3102],
        [-0.4151, -0.4892,  0.7049,  0.3011, -0.0025, -0.0291],
        [ 0.2606, -0.2261, -0.3266,  0.7317, -0.4367, -0.2196],
        [-0.3465,  0.6565,  0.0671,  0.3898,  0.2869, -0.4585],
        [ 0.2684,  0.2890,  0.2027,  0.4392,  0.2176,  0.7504]
    ], dtype=torch.float32)
    
    A2_sqrt = torch.tensor([
        [ 0.7440,  0.3316,  0.1310,  0.3899, -0.3842,  0.1407],
        [-0.3770,  0.7573, -0.0141,  0.3999,  0.3471,  0.0613],
        [-0.0616,  0.0329,  0.8850, -0.0297,  0.0125, -0.4592],
        [-0.3805, -0.4243,  0.0145,  0.7457, -0.3447, -0.0091],
        [ 0.3856, -0.3425, -0.0128,  0.3575,  0.7717, -0.1029],
        [-0.0848, -0.1347,  0.4462, -0.0569,  0.1322,  0.8688]
    ], dtype=torch.float32)
    
    # Build the exact coupled feedback matrix from MATLAB
    cos_c = torch.cos(torch.tensor(coupling))
    sin_c = torch.sin(torch.tensor(coupling))
    
    # Create the block matrix structure
    feedback_matrix = torch.zeros(N, N)
    feedback_matrix[:N_per_room, :N_per_room] = cos_c * A1
    feedback_matrix[:N_per_room, N_per_room:] = sin_c * torch.matmul(A1_sqrt, A2_sqrt)
    feedback_matrix[N_per_room:, :N_per_room] = -sin_c * torch.matmul(A2_sqrt, A1_sqrt)
    feedback_matrix[N_per_room:, N_per_room:] = cos_c * A2
    
    print(f"\nFeedback matrix (top-left 3x3):\n{feedback_matrix[:3, :3]}")
    print(f"Feedback matrix (top-right 3x3):\n{feedback_matrix[:3, 6:9]}")
    
    # Verify orthogonality
    ortho_check = torch.matmul(feedback_matrix.T, feedback_matrix)
    max_deviation = torch.max(torch.abs(ortho_check - torch.eye(N)))
    print(f"\nMax deviation from orthogonality: {max_deviation:.6e}")
    
    ## Build FDN using FLAMO components
    
    # Input gain: source only in first room (matches MATLAB exactly)
    input_gain = dsp.Gain(
        size=(N, num_input),
        nfft=nfft,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=device,
    )
    input_gain_values = torch.zeros(N, num_input)
    input_gain_values[:N_per_room, :] = 1.0  # First 6 delays get input
    input_gain.assign_value(input_gain_values)
    
    # Output gain: block diagonal [ones(1,6), zeros; zeros, ones(1,6)]
    output_gain = dsp.Gain(
        size=(num_output, N),
        nfft=nfft,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=device,
    )
    output_gain_values = torch.zeros(num_output, N)
    output_gain_values[0, :N_per_room] = 1.0  # Room 1 to left channel
    output_gain_values[1, N_per_room:] = 1.0   # Room 2 to right channel
    output_gain.assign_value(output_gain_values)
    
    # Create delay lines
    delays = dsp.parallelDelay(
        size=(N,),
        max_len=int(delay_lengths.max()),
        nfft=nfft,
        isint=True,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=device,
    )
    delays.assign_value(delays.sample2s(delay_lengths.int()))
    
    # Create mixing matrix with the exact feedback matrix
    mixing_matrix = dsp.Matrix(
        size=(N, N),
        nfft=nfft,
        matrix_type="random",  # We'll override with our values
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=device,
    )
    mixing_matrix.assign_value(feedback_matrix)
    
    # Create attenuation filters
    # T60 values from MATLAB (at 1kHz)
    shortT60 = torch.tensor([0.5, 0.5, 0.55, 0.575, 0.525, 0.375, 0.275, 0.2, 0.175, 0.175])
    longT60 = torch.tensor([4.0, 4.0, 4.4, 4.6, 4.2, 3.0, 2.2, 1.6, 1.4, 1.4])
    
    print(f"\nShort T60 (1kHz): {shortT60[4].item():.3f}s")
    print(f"Long T60 (1kHz): {longT60[4].item():.3f}s")
    
    # For simplicity, use frequency-independent attenuation at 1kHz band
    attenuation = dsp.parallelGain(
        size=(N,),
        nfft=nfft,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=device,
    )
    
    # Calculate attenuation coefficients from T60
    def t60_to_gain_per_sample(t60, fs):
        """Convert T60 to gain coefficient per sample"""
        return 10 ** (-3 / (t60 * fs))
    
    # Use the 1kHz band T60 values
    short_t60_1khz = shortT60[4].item()  # 0.525 seconds
    long_t60_1khz = longT60[4].item()    # 4.2 seconds
    
    attenuation_values = torch.zeros(N)
    # Room 1 (short T60)
    g_short = t60_to_gain_per_sample(short_t60_1khz, fs)
    for i in range(N_per_room):
        attenuation_values[i] = g_short ** delay_lengths[i]
    
    # Room 2 (long T60)
    g_long = t60_to_gain_per_sample(long_t60_1khz, fs)
    for i in range(N_per_room, N):
        attenuation_values[i] = g_long ** delay_lengths[i]
    
    attenuation.assign_value(attenuation_values)
    
    # Create feedback path
    feedback = system.Series(
        OrderedDict({
            "mixing_matrix": mixing_matrix,
            "attenuation": attenuation
        })
    )
    
    # Create recursion (feedback loop)
    feedback_loop = system.Recursion(fF=delays, fB=feedback)
    
    # Complete FDN
    fdn = system.Series(
        OrderedDict({
            "input_gain": input_gain,
            "feedback_loop": feedback_loop,
            "output_gain": output_gain,
        })
    )
    
    # No direct path (D matrix is zeros)
    # This matches the MATLAB: direct = zeros(numOutput,numInput)
    
    # Create shell with FFT/iFFT
    input_layer = dsp.FFT(nfft)
    output_layer = dsp.iFFT(nfft)
    
    model = system.Shell(
        core=fdn,
        input_layer=input_layer,
        output_layer=output_layer
    )
    
    # Generate impulse response
    print("\nGenerating impulse response...")
    with torch.no_grad():
        # Use direct impulse processing (faster and more reliable)
        impulse = torch.zeros(1, nfft, 1)
        impulse[0, 0, 0] = 1.0
        ir = model(impulse).squeeze().cpu().numpy()
        
        # correct shape
        if ir.ndim == 1:
            ir = ir.reshape(-1, 1)
        
        # Trim to desired length
        ir = ir[:impulse_response_length, :]
        
        print(f"IR shape: {ir.shape}")
        print(f"Max amplitude room 1: {np.max(np.abs(ir[:, 0])):.6f}")
        if ir.shape[1] > 1:
            print(f"Max amplitude room 2: {np.max(np.abs(ir[:, 1])):.6f}")
        
        # Print first 10 samples for verification
        print(f"\nFirst 10 samples of room 1 IR:")
        print(ir[:10, 0])
        if ir.shape[1] > 1:
            print(f"\nFirst 10 samples of room 2 IR:")
            print(ir[:10, 1])
    
    return ir, fs, feedback_matrix.numpy(), delay_lengths.numpy()


def plot_results(ir, fs, feedback_matrix):
    """
    Plot the impulse responses, feedback matrix, and energy decay curves.
    
    Args:
        ir: Impulse response array
        fs: Sample rate
        feedback_matrix: The feedback matrix
    """
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1, Impulse responses (using samples like MATLAB)
    ax1 = plt.subplot(1, 3, 1)
    samples = np.arange(len(ir))
    ax1.plot(samples, ir[:, 0], label='Short Room', alpha=0.7, linewidth=0.5)
    if ir.shape[1] > 1:
        # Offset for visibility (matching MATLAB plot)
        ax1.plot(samples, ir[:, 1] - 2, label='Long Room', alpha=0.7, linewidth=0.5)
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Coupled Rooms Impulse Response')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, len(ir)])
    
    # Plot 2, Feedback matrix
    ax2 = plt.subplot(1, 3, 2)
    im = ax2.imshow(feedback_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title('Feedback Matrix')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    
    # Add grid lines to show room separation
    ax2.axhline(y=5.5, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax2.axvline(x=5.5, color='black', linewidth=1, linestyle='--', alpha=0.5)
    
    # Plot 3, Energy decay curves (instead of pole plot from MATLAB's dss2pr)
    ax3 = plt.subplot(1, 3, 3)
    
    # Compute backward energy integration (Schroeder integral)
    t = np.arange(len(ir)) / fs  # Time in seconds for EDC
    edc1 = np.cumsum(ir[::-1, 0]**2)[::-1]
    edc1_db = 10 * np.log10(edc1 / (edc1[0] + 1e-12))
    ax3.plot(t, edc1_db, label='Short Room', alpha=0.8)
    
    if ir.shape[1] > 1:
        edc2 = np.cumsum(ir[::-1, 1]**2)[::-1]
        edc2_db = 10 * np.log10(edc2 / (edc2[0] + 1e-12))
        ax3.plot(t, edc2_db, label='Long Room', alpha=0.8)
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Energy (dB)')
    ax3.set_title('Energy Decay Curves\n(Note: MATLAB shows poles, Python shows EDC)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, min(2, len(ir)/fs)])
    ax3.set_ylim([-60, 0])
    
    plt.tight_layout()
    plt.savefig('coupled_rooms_python.png', dpi=150)
    plt.show()
    
    # Save audio file
    if ir.shape[1] > 1:
        # Stereo output
        ir_normalized = ir / np.max(np.abs(ir))
    else:
        # Mono output, duplicate to stereo
        ir_normalized = np.column_stack([ir, ir]) / np.max(np.abs(ir))
    
    sf.write('coupled_rooms_python.wav', ir_normalized, fs)
    print(f"\nSaved audio to 'coupled_rooms_python.wav'")
    print(f"Saved plot to 'coupled_rooms_python.png'")


if __name__ == "__main__":
    print("=" * 60)
    print("Coupled Rooms FDN Example")
    print("Python Translation by Facundo Franchino")
    print("Using FLAMO library")
    print("=" * 60)
    print("\nOriginal MATLAB code by Sebastian J. Schlecht")
    print("Based on: Das, Abel & Canfield-Dafilou (DAFx 2020)")
    print("=" * 60)
    
    # Generate coupled rooms FDN
    ir, fs, feedback_matrix, delays = create_coupled_rooms_fdn()
    
    # Plot results
    plot_results(ir, fs, feedback_matrix)
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("\nImplementation Notes:")
    print("- Modal decomposition (dss2pr) has been omitted as requested")
    print("- Using exact delay values and matrices from MATLAB with rng(5)")
    print("- Frequency-independent attenuation for simplicity")
    print("  (Can be upgraded to use parallelFDNAccurateGEQ for full frequency-dependent absorption)")
    print("- The coupling is implemented through the block matrix structure")
    print("=" * 60)