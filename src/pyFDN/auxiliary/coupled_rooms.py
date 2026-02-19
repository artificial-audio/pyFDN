"""
Coupled rooms FDN builder.

Translation of example_coupledRooms.m from fdnToolbox.
Original MATLAB code: (c) Sebastian Jiro Schlecht, 2020
Python translation: Facundo Franchino, 2025
"""

from collections import OrderedDict

import numpy as np
import torch

from flamo.processor import dsp, system

from pyFDN.auxiliary.acoustics import rt_to_gain_per_sample
from pyFDN.auxiliary.tiny_rotation_matrix import tiny_rotation_matrix


def create_coupled_rooms_fdn():
    """
    Create a coupled rooms FDN matching the MATLAB implementation.

    This function builds a 12-delay-line FDN modeling two acoustically
    coupled rooms with different reverberation characteristics.

    Returns:
        ir: Impulse response (numpy array, shape [samples, 2])
        fs: Sample rate (48000 Hz)
        feedback_matrix: The 12x12 feedback matrix used
        delay_lengths: The 12 delay lengths in samples

    Example:
        >>> torch.manual_seed(5)
        >>> np.random.seed(5)
        >>> ir, fs, fb, delays = create_coupled_rooms_fdn()
        >>> ir.shape
        (96000, 2)
    """
    # Parameters (matching MATLAB exactly)
    fs = 48000
    impulse_response_length = fs * 2  # 2 seconds
    nfft = 16384

    # FDN configuration
    N = 12  # Total number of delay lines
    N_per_room = N // 2  # 6 delay lines per room
    num_input = 1
    num_output = 2

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alias_decay_db = 0  # No anti-aliasing for exact reproduction

    # Exact delay values from MATLAB with rng(5)
    delays_room1 = torch.tensor([411, 736, 403, 760, 544, 606], dtype=torch.float32)
    delays_room2 = torch.tensor(
        [2532, 2037, 1593, 1375, 1161, 2477], dtype=torch.float32
    )
    delay_lengths = torch.cat([delays_room1, delays_room2])

    # Coupling parameter (exact from MATLAB)
    coupling = 0.3

    # Generate feedback matrices using tinyRotationMatrix
    A1 = tiny_rotation_matrix(N_per_room, 12).float()
    A2 = tiny_rotation_matrix(N_per_room, 12).float()

    # Compute matrix square roots using eigenvalue decomposition
    def matrix_sqrt(A):
        eigenvals, eigenvecs = torch.linalg.eig(A.to(torch.complex64))
        sqrt_eigenvals = torch.sqrt(eigenvals)
        return torch.real(
            eigenvecs @ torch.diag(sqrt_eigenvals) @ torch.linalg.inv(eigenvecs)
        ).float()

    A1_sqrt = matrix_sqrt(A1)
    A2_sqrt = matrix_sqrt(A2)

    # Build the coupled feedback matrix
    cos_c = torch.cos(torch.tensor(coupling))
    sin_c = torch.sin(torch.tensor(coupling))

    feedback_matrix = torch.zeros(N, N)
    feedback_matrix[:N_per_room, :N_per_room] = cos_c * A1
    feedback_matrix[:N_per_room, N_per_room:] = sin_c * torch.matmul(A1_sqrt, A2_sqrt)
    feedback_matrix[N_per_room:, :N_per_room] = -sin_c * torch.matmul(A2_sqrt, A1_sqrt)
    feedback_matrix[N_per_room:, N_per_room:] = cos_c * A2

    # Build FDN using FLAMO components
    input_gain = dsp.Gain(
        size=(N, num_input),
        nfft=nfft,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=device,
    )
    input_gain_values = torch.zeros(N, num_input)
    input_gain_values[:N_per_room, :] = 1.0
    input_gain.assign_value(input_gain_values)

    output_gain = dsp.Gain(
        size=(num_output, N),
        nfft=nfft,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=device,
    )
    output_gain_values = torch.zeros(num_output, N)
    output_gain_values[0, :N_per_room] = 1.0
    output_gain_values[1, N_per_room:] = 1.0
    output_gain.assign_value(output_gain_values)

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

    mixing_matrix = dsp.Matrix(
        size=(N, N),
        nfft=nfft,
        matrix_type="random",
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=device,
    )
    mixing_matrix.assign_value(feedback_matrix)

    # T60 values from MATLAB (at 1kHz)
    short_rt = torch.tensor(
        [0.5, 0.5, 0.55, 0.575, 0.525, 0.375, 0.275, 0.2, 0.175, 0.175]
    )
    long_rt = torch.tensor([4.0, 4.0, 4.4, 4.6, 4.2, 3.0, 2.2, 1.6, 1.4, 1.4])

    attenuation = dsp.parallelGain(
        size=(N,),
        nfft=nfft,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=device,
    )

    short_rt_1khz = short_rt[4].item()
    long_rt_1khz = long_rt[4].item()

    attenuation_values = torch.zeros(N)
    g_short = rt_to_gain_per_sample(short_rt_1khz, fs)
    for i in range(N_per_room):
        attenuation_values[i] = g_short ** delay_lengths[i]

    g_long = rt_to_gain_per_sample(long_rt_1khz, fs)
    for i in range(N_per_room, N):
        attenuation_values[i] = g_long ** delay_lengths[i]

    attenuation.assign_value(attenuation_values)

    feedback = system.Series(
        OrderedDict({"mixing_matrix": mixing_matrix, "attenuation": attenuation})
    )

    feedback_loop = system.Recursion(fF=delays, fB=feedback)

    fdn = system.Series(
        OrderedDict(
            {
                "input_gain": input_gain,
                "feedback_loop": feedback_loop,
                "output_gain": output_gain,
            }
        )
    )

    input_layer = dsp.FFT(nfft)
    output_layer = dsp.iFFT(nfft)

    model = system.Shell(
        core=fdn, input_layer=input_layer, output_layer=output_layer
    )

    # Generate impulse response
    with torch.no_grad():
        impulse = torch.zeros(1, nfft, 1)
        impulse[0, 0, 0] = 1.0
        ir = model(impulse).squeeze().cpu().numpy()

        if ir.ndim == 1:
            ir = ir.reshape(-1, 1)

        ir = ir[:impulse_response_length, :]

    return ir, fs, feedback_matrix.numpy(), delay_lengths.numpy()
