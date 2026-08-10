"""
Convert delay state-space (A, B, C, D, m) to a FLAMO model for rendering.

Uses gain_module and delay_module from pyFDN.auxiliary.flamo.
Optionally place an allpass (or other) filter behind the delays in the loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pyFDN.auxiliary.flamo import delay_module, gain_module

if TYPE_CHECKING:
    from pyFDN.generate.fdn_matrix_gallery import FDNBuild

try:
    import flamo.processor  # noqa: F401

    _HAS_FLAMO = True
except ImportError:
    _HAS_FLAMO = False


def dss_to_flamo(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray | None,
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
    D : (num_out, num_in) array or None
        Direct gain. ``None`` resolves to a zero ``(num_out, num_in)`` gain.
    m : (N,) array
        Delay lengths in samples (one per delay line).
    Fs : float
        Sampling rate in Hz.
    nfft : int
        FFT size for FLAMO (default 2**16).
    device : torch device or None
        Device; default is cuda if available else cpu.
    shell : bool
        If True (default), wrap the core in a Shell with FFT/iFFT. Use
        :func:`pyFDN.flamo_time_response` to obtain a NumPy impulse response.
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
        If shell=True, FLAMO Shell. Use :func:`pyFDN.flamo_time_response` for
        a NumPy impulse response.
        If shell=False, the core module (same I/O as B.shape[1] / C.shape[0]).
    """
    if not _HAS_FLAMO:
        raise ImportError("dss_to_flamo requires flamo (pip install flamo)")

    import torch

    from pyFDN.auxiliary.flamo import (
        assemble_fdn_core,
        fir_matrix_module,
        sos_filter_module,
        wrap_fdn_shell,
    )

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    D = (
        np.zeros((C.shape[0], B.shape[1]), dtype=np.float64)
        if D is None
        else np.asarray(D, dtype=np.float64)
    )
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
    loop_filter = (
        sos_filter_module(sos_filter, nfft, device=device, dtype=dtype)
        if sos_filter is not None
        else None
    )
    out_filter = (
        sos_filter_module(output_filter, nfft, device=device, dtype=dtype)
        if output_filter is not None
        else None
    )

    # Wiring is delegated to the shared assembler so the render path here and the
    # training builder (pyFDN.train) stay byte-for-byte identical in topology.
    core = assemble_fdn_core(
        input_gain=gain_B,
        feedback=gain_A,
        delays=delays,
        output_gain=gain_C,
        direct=gain_D,
        loop_filter=loop_filter,
        output_filter=out_filter,
        post_delay_module=post_delay_module,
    )

    if shell:
        return wrap_fdn_shell(core, nfft=nfft, dtype=dtype)
    return core


def build_to_flamo(
    build: FDNBuild,
    nfft: int = 2**16,
    device: Any = None,
    *,
    shell: bool = True,
    dtype: Any = None,
    post_delay_module: Any = None,
) -> Any:
    """
    Build a FLAMO model from a complete :class:`FDNBuild` config.

    Thin wrapper over :func:`dss_to_flamo` that unpacks an
    :class:`~pyFDN.generate.fdn_matrix_gallery.FDNBuild` (as returned by
    :func:`pyFDN.fdn_build_gallery`) into its state-space arguments, mapping the
    in-loop absorption ``build.filters`` to ``sos_filter`` and the per-output
    ``build.post_eq`` to ``output_filter``.

    Parameters
    ----------
    build : FDNBuild
        Complete FDN parameters (``A``, ``B``, ``C``, ``D``, ``delays``,
        ``fs``, optional ``filters`` and ``post_eq``), e.g. from
        :func:`pyFDN.fdn_build_gallery`.
    nfft : int
        FFT size for FLAMO (default 2**16).
    device : torch device or None
        Device; default is cuda if available else cpu.
    shell : bool
        If True (default), wrap the core in a Shell with FFT/iFFT. Use
        :func:`pyFDN.flamo_time_response` to obtain a NumPy impulse response.
        If False, return only the core.
    dtype : torch.dtype or None
        Optional dtype for FLAMO delay/gain/filter modules (e.g., torch.float64).
        If None, wrapper defaults are used.
    post_delay_module : FLAMO module or None
        Optional module to append after the delay in the recursion (e.g. a
        Schroeder allpass core). Must have input/output size N. Loop becomes:
        delay -> post_delay_module -> A.

    Returns
    -------
    model : flamo.processor.system.Shell or core
        If shell=True, a FLAMO Shell. Use :func:`pyFDN.flamo_time_response` for
        a NumPy impulse response. If shell=False, the core module.
    """
    return dss_to_flamo(
        build.A,
        build.B,
        build.C,
        build.D,
        build.delays,
        build.fs,
        nfft=nfft,
        device=device,
        shell=shell,
        dtype=dtype,
        sos_filter=build.filters,
        output_filter=build.post_eq,
        post_delay_module=post_delay_module,
    )
