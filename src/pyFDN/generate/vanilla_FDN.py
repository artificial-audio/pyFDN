"""Build a vanilla FDN (delays + feedback matrix + first-order absorption) using FLAMO."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from flamo.processor import dsp, system

from pyFDN.auxiliary.acoustics import first_order_absorption
from pyFDN.generate.random_orthogonal import random_orthogonal

if TYPE_CHECKING:
    pass  # FLAMO Shell type not annotated here


def vanilla_FDN(
    n: int,
    *,
    rt_dc: float = 2.0,
    rt_ny: float = 0.5,
    fs: float = 48000.0,
    n_fft: int = 2**16,
    device: str = "cpu",
    num_input: int = 1,
    num_output: int = 1,
    delay_min: int = 400,
    delay_max: int = 1200,
) -> Any:
    """Build a vanilla FDN (delays + feedback matrix + first-order absorption) in FLAMO.

    Delay lengths are drawn uniformly at random in [delay_min, delay_max) per channel.
    Uses a random orthogonal feedback matrix and first-order shelving absorption per
    delay (rt_dc at DC, rt_ny at Nyquist).

    Parameters
    ----------
    n : int
        Number of delay lines (and matrix size).
    rt_dc : float, optional
        Target reverberation time (T60) at DC in seconds (default 2.0).
    rt_ny : float, optional
        Target reverberation time (T60) at Nyquist in seconds (default 0.5).
    fs : float, optional
        Sampling rate in Hz (default 48000).
    n_fft : int, optional
        FFT size for FLAMO (default 2**16).
    device : str, optional
        Device for FLAMO (default "cpu").
    num_input : int, optional
        Number of inputs (default 1).
    num_output : int, optional
        Number of outputs (default 1).
    delay_min : int, optional
        Minimum delay length in samples (default 400).
    delay_max : int, optional
        Maximum delay length in samples, exclusive (default 1200).

    Returns
    -------
    model
        FLAMO Shell instance (core = direct + FDN, with FFT/iFFT layers).
    """

    delays_arr = np.random.randint(delay_min, delay_max, size=n).astype(np.float64)
    delays_torch = torch.tensor(delays_arr, dtype=torch.float32, device=device)
    delay_module = dsp.parallelDelay(
        size=(n,),
        max_len=int(delays_torch.max().item()),
        nfft=n_fft,
        isint=True,
        unit=1,
        device=device,
    )
    delay_module.assign_value(delay_module.sample2s(delays_torch))

    input_gain = dsp.Gain(size=(n, num_input), nfft=n_fft, device=device)
    input_gain.assign_value(torch.ones(n, num_input, device=device))

    output_gain = dsp.Gain(size=(num_output, n), nfft=n_fft, device=device)
    output_gain.assign_value(torch.ones(num_output, n, device=device))

    # Random orthogonal feedback matrix
    feedback_matrix = random_orthogonal(n).astype(np.float32)
    feedback_torch = torch.tensor(feedback_matrix, dtype=torch.float32, device=device)
    mixing_matrix = dsp.Matrix(
        size=(n, n), nfft=n_fft, matrix_type="random", device=device
    )
    mixing_matrix.assign_value(feedback_torch)

    sos = first_order_absorption(rt_dc, rt_ny, delays_arr, fs=float(fs))
    absorption_coeff = torch.tensor(
        sos[np.newaxis, ...], dtype=torch.float32, device=device
    )
    absorption = dsp.parallelSOSFilter(
        size=(n,), n_sections=1, nfft=n_fft, device=device
    )
    absorption.assign_value(absorption_coeff)
    delay_chain = system.Series(
        OrderedDict({"delay": delay_module, "absorption": absorption})
    )

    feedback_loop = system.Recursion(fF=delay_chain, fB=mixing_matrix)
    fdn = system.Series(
        OrderedDict(
            {
                "input_gain": input_gain,
                "feedback_loop": feedback_loop,
                "output_gain": output_gain,
            }
        )
    )

    direct_gain = dsp.Gain(size=(num_output, num_input), nfft=n_fft, device=device)
    direct_gain.assign_value(torch.ones(num_output, num_input, device=device))

    complete_system = system.Parallel(brA=direct_gain, brB=fdn, sum_output=True)
    model = system.Shell(
        core=complete_system,
        input_layer=dsp.FFT(n_fft),
        output_layer=dsp.iFFT(n_fft),
    )
    return model
