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
    D: np.ndarray,
    m: np.ndarray,
    fs: float,
    nfft: int = 2**16,
    device: Any = None,
    *,
    shell: bool = True,
    dtype: Any = None,
    post_delay: Any = None,
    post_matrix: Any = None,
    post_output: Any = None,
) -> Any:
    """
    Build a FLAMO model from delay state-space (A, B, C, D, m).

    Signal flow: input -> B -> [recursion: delay -> (post_delay); fB = A -> (post_matrix)]
    -> C -> (post_output) -> output, with direct path D summed in parallel.

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
    fs : float
        Sampling rate in Hz.
    nfft : int
        FFT size for FLAMO (default 2**16).
    device : torch device or None
        Device; default is cuda if available else cpu.
    shell : bool
        If True (default), wrap the core in a Shell with FFT/iFFT. Use
        :func:`pyFDN.flamo_time_response` to obtain a NumPy impulse response.
        If False, return only the core (e.g. for use as post_delay in another dss_to_flamo).
    dtype : torch.dtype or None
        Optional dtype for FLAMO delay/gain/filter modules (e.g., torch.float64).
        If None, wrapper defaults are used.
    post_delay : array, FLAMO module, sequence, or None
        In-loop filter applied to the delay output, inside the recursion -- the
        same hook :func:`pyFDN.process_fdn` calls ``post_delay``. An
        ``(n_sections, 6, N)`` SOS bank, a FLAMO module of input/output size N
        (e.g. a Schroeder allpass core from ``shell=False``), or a sequence of
        both applied in order. See :func:`pyFDN.hook_module`.
    post_matrix : array, FLAMO module, sequence, or None
        Filter applied to the feedback path after ``A``.
    post_output : array, FLAMO module, sequence, or None
        Per-output filter applied to the wet signal after ``C``; an
        ``(n_sections, 6, num_out)`` SOS bank, or a module.

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
        hook_module,
        wrap_fdn_shell,
    )

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
    lengths_sec = m / float(fs)
    delays = delay_module(lengths_sec, nfft, fs=fs, device=device, dtype=dtype)
    if A.ndim == 3:
        gain_A = fir_matrix_module(A, nfft, device=device, dtype=dtype)
    else:
        gain_A = gain_module(A, nfft, device=device, dtype=dtype)
    gain_B = gain_module(B, nfft, device=device, dtype=dtype)
    gain_C = gain_module(C, nfft, device=device, dtype=dtype)
    gain_D = gain_module(D, nfft, device=device, dtype=dtype)
    hooks = {
        name: hook_module(value, nfft, name=name, device=device, dtype=dtype)
        for name, value in (
            ("post_delay", post_delay),
            ("post_matrix", post_matrix),
            ("post_output", post_output),
        )
    }

    # Wiring is delegated to the shared assembler so the render path here and the
    # training builder (pyFDN.train) stay byte-for-byte identical in topology.
    core = assemble_fdn_core(
        input_gain=gain_B,
        feedback=gain_A,
        delays=delays,
        output_gain=gain_C,
        direct=gain_D,
        **hooks,
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
    post_delay: Any = None,
    post_matrix: Any = None,
    post_output: Any = None,
) -> Any:
    """
    Build a FLAMO model from a complete :class:`FDNBuild` config.

    Thin wrapper over :func:`dss_to_flamo` that unpacks an
    :class:`~pyFDN.generate.fdn_matrix_gallery.FDNBuild` (as returned by
    :func:`pyFDN.fdn_build_gallery`) into its state-space arguments. The build's
    three filter hooks go straight through under the same names: ``post_delay``
    for the in-loop absorption, ``post_matrix`` for the feedback path, and
    ``post_output`` for the per-output EQ.

    Parameters
    ----------
    build : FDNBuild
        Complete FDN parameters (``A``, ``B``, ``C``, ``D``, ``delays``,
        ``fs``, optional ``post_delay`` and ``post_output``), e.g. from
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
    post_delay, post_matrix, post_output : array, FLAMO module, sequence, or None
        Extra modules for the three filter hooks, appended *after* whatever the
        build already carries in that position -- so ``post_delay=schroeder_core``
        on a build with absorption gives ``delay -> absorption -> schroeder``.
        See :func:`pyFDN.hook_module`.

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
        post_delay=_appended(build.post_delay, post_delay),
        post_matrix=_appended(build.post_matrix, post_matrix),
        post_output=_appended(build.post_output, post_output),
    )


def _appended(baked: Any, extra: Any) -> Any:
    """The build's own hook contents, then whatever the caller adds to it."""
    parts = [] if baked is None else [baked]
    if isinstance(extra, list | tuple):
        parts.extend(extra)
    elif extra is not None:
        parts.append(extra)
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else parts
