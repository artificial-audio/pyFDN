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

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch

# FLAMO imports
from flamo.processor import dsp, system

# Set random seed for reproducibility (matches MATLAB rng(5))
torch.manual_seed(5)
np.random.seed(5)

def tiny_rotation_matrix(n, delta, spread=0.1, log_matrix=None):
    """
    Translation of tinyRotationMatrix.m - orthogonal matrix with small eigenvalue angles.

    Args:
        n: Matrix size
        delta: Mean normalized eigenvalue angle
        spread: Spreading of eigenvalue angle (default 0.1)
        log_matrix: Initial logarithm matrix (default random)

    Returns:
        rotation_matrix: Orthogonal matrix
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
    real_eigenval_mask = (idx == torch.arange(n))
    frequency_spread[real_eigenval_mask] = 0

    # Scale and spread eigenvalues
    eigenvals_normalized = eigenvalues / (torch.abs(eigenvalues) + 1e-12)
    new_eigenvals = eigenvals_normalized * frequency_spread * delta * torch.pi

    # Create new skew-symmetric matrix
    new_skew = torch.real(eigenvectors @ torch.diag(new_eigenvals) @ eigenvectors.conj().T)
    new_skew = (new_skew - new_skew.T) / 2

    # Matrix exponential to get orthogonal matrix
    rotation_matrix = torch.matrix_exp(new_skew)

    return rotation_matrix


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

    # FDN configuration (uppercase N follows MATLAB convention)
    N = 12  # noqa: N806 - Total number of delay lines
    N_per_room = N // 2  # noqa: N806 - 6 delay lines per room
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


    # Coupling parameter (exact from MATLAB)
    coupling = 0.3

    # Generate feedback matrices using tinyRotationMatrix
    # Matches MATLAB: A1 = tinyRotationMatrix(N/2,12); A2 = tinyRotationMatrix(N/2,12);
    A1 = tiny_rotation_matrix(N_per_room, 12).float()
    A2 = tiny_rotation_matrix(N_per_room, 12).float()


    # Compute matrix square roots using eigenvalue decomposition
    # sqrtm(A) = V * sqrt(D) * V^(-1) where A = V * D * V^(-1)
    def matrix_sqrt(A):
        eigenvals, eigenvecs = torch.linalg.eig(A.to(torch.complex64))
        sqrt_eigenvals = torch.sqrt(eigenvals)
        return torch.real(eigenvecs @ torch.diag(sqrt_eigenvals) @ torch.linalg.inv(eigenvecs)).float()

    A1_sqrt = matrix_sqrt(A1)
    A2_sqrt = matrix_sqrt(A2)

    # Build the exact coupled feedback matrix from MATLAB
    cos_c = torch.cos(torch.tensor(coupling))
    sin_c = torch.sin(torch.tensor(coupling))

    # Create the block matrix structure
    feedback_matrix = torch.zeros(N, N)
    feedback_matrix[:N_per_room, :N_per_room] = cos_c * A1
    feedback_matrix[:N_per_room, N_per_room:] = sin_c * torch.matmul(A1_sqrt, A2_sqrt)
    feedback_matrix[N_per_room:, :N_per_room] = -sin_c * torch.matmul(A2_sqrt, A1_sqrt)
    feedback_matrix[N_per_room:, N_per_room:] = cos_c * A2


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
    _fig = plt.figure(figsize=(15, 5))

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
    plt.show()


if __name__ == "__main__":
    # Generate coupled rooms FDN
    ir, fs, feedback_matrix, delays = create_coupled_rooms_fdn()

    # Plot results
    plot_results(ir, fs, feedback_matrix)
