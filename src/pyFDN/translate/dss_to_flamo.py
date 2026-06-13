"""
Convert delay state-space (A, B, C, D, m) to a FLAMO model for rendering.

Uses gain_module and delay_module from pyFDN.auxiliary.flamo.
Optionally place an allpass (or other) filter behind the delays in the loop.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from pyFDN.auxiliary.flamo import delay_module, gain_module

try:
    from flamo.processor import dsp, system

    _HAS_FLAMO = True
except ImportError:
    _HAS_FLAMO = False


def dss_to_flamo(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    m: np.ndarray,
    Fs: float,
    nfft: int = 2**16,
    device: Any = None,
    *,
    shell: bool = True,
    dtype: Any = None,
    sos_filter: np.ndarray | None = None,
    output_filter: np.ndarray | None = None,
    post_delay_module: Any = None,
) -> Any:
    """
    Build a FLAMO model from delay state-space (A, B, C, D, m).

    Signal flow: input -> B -> [recursion: delay -> (optional filter/module) -> A] -> C -> output,
    with direct path D summed in parallel.

    Parameters
    ----------
    A : (N, N) or (N, N, L) array
        Feedback matrix. A 3-D array is a polynomial (FIR) matrix in z^{-1}
        convention (e.g. paraunitary) and is placed as a FLAMO Filter module.
    B : (N, num_in) array
        Input gain.
    C : (num_out, N) array
        Output gain.
    D : (num_out, num_in) array
        Direct gain.
    m : (N,) array
        Delay lengths in samples (one per delay line).
    Fs : float
        Sampling rate in Hz.
    nfft : int
        FFT size for FLAMO (default 2**16).
    device : torch device or None
        Device; default is cuda if available else cpu.
    shell : bool
        If True (default), wrap the core in a Shell with FFT/iFFT for get_time_response().
        If False, return only the core (e.g. for use as post_delay_module in another dss_to_flamo).
    dtype : torch.dtype or None
        Optional dtype for FLAMO delay/gain/filter modules (e.g., torch.float64).
        If None, wrapper defaults are used.
    sos_filter : (n_sections, 6, N) array or None
        Optional SOS filter in the loop after delays.
    output_filter : (n_sections, 6, num_out) array or None
        Optional SOS filter cascade applied per output channel after the
        output gain C (e.g. an output equalizer), matching the output
        filters of the MATLAB ``dss2impz``.
    post_delay_module : FLAMO module or None
        Optional module to append after the delay in the recursion (e.g. a Schroeder allpass core).
        Must have input/output size N. Loop becomes: delay -> post_delay_module -> A.

    Returns
    -------
    model : flamo.processor.system.Shell or core
        If shell=True, FLAMO Shell (use model.get_time_response() for IR).
        If shell=False, the core module (same I/O as B.shape[1] / C.shape[0]).
    """
    if not _HAS_FLAMO:
        raise ImportError("dss_to_flamo requires flamo (pip install flamo)")

    import torch

    from pyFDN.auxiliary.flamo import fir_matrix_module, sos_filter_module

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64).ravel()
    N = A.shape[0]
    if m.shape[0] != N:
        raise ValueError("m must have length N (number of delay lines)")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Delays: convert samples to seconds for FLAMO
    lengths_sec = m / float(Fs)
    delays = delay_module(lengths_sec, nfft, Fs=Fs, device=device, dtype=dtype)
    if A.ndim == 3:
        gain_A = fir_matrix_module(A, nfft, device=device, dtype=dtype)
    else:
        gain_A = gain_module(A, nfft, device=device, dtype=dtype)
    gain_B = gain_module(B, nfft, device=device, dtype=dtype)
    gain_C = gain_module(C, nfft, device=device, dtype=dtype)
    gain_D = gain_module(D, nfft, device=device, dtype=dtype)

    if sos_filter is not None:
        filter_module = sos_filter_module(sos_filter, nfft, device=device, dtype=dtype)
        delay_chain = system.Series(
            OrderedDict({"delay": delays, "filter": filter_module})
        )
    else:
        delay_chain = delays

    if post_delay_module is not None:
        delay_chain = system.Series(
            OrderedDict({"delay": delay_chain, "post_delay_module": post_delay_module})
        )

    feedback_loop = system.Recursion(fF=delay_chain, fB=gain_A)
    fdn_modules = OrderedDict(
        {
            "input_gain": gain_B,
            "feedback_loop": feedback_loop,
            "output_gain": gain_C,
        }
    )
    if output_filter is not None:
        fdn_modules["output_filter"] = sos_filter_module(
            output_filter, nfft, device=device, dtype=dtype
        )
    fdn_branch = system.Series(fdn_modules)
    core = system.Parallel(brA=fdn_branch, brB=gain_D, sum_output=True)

    if shell:
        torch_dtype = torch.float32 if dtype is None else dtype
        return system.Shell(
            core=core,
            input_layer=dsp.FFT(nfft, dtype=torch_dtype),
            output_layer=dsp.iFFT(nfft, dtype=torch_dtype),
        )
    return core
