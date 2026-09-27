"""
Convert a delay state-space (DSS) system (A, B, C, D, delays) to a FLAMO model
for rendering.

Optionally place an allpass (or other) filter behind the delays in the loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.flamo import delay_module

if TYPE_CHECKING:
    from pyFDN.build import FDNBuild


def dss_to_flamo(
    delays: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
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
    Build a FLAMO model from a delay state-space (DSS) system (delays, A, B, C, D).

    Signal flow: input -> B -> [recursion: delay -> (post_delay); fB = A -> (post_matrix)]
    -> C -> (post_output) -> output, with direct path D summed in parallel.

    Parameters
    ----------
    delays : array-like, (N,)
        Delay lengths in samples (one per delay line).
    A : array-like, (N, N) or (N, N, L)
        Feedback matrix. A 3-D array is a polynomial (FIR) matrix in z^{-1}
        convention (e.g. paraunitary) and is placed as a FLAMO Filter module.
    B : array-like, (N, num_in)
        Input gain.
    C : array-like, (num_out, N)
        Output gain.
    D : array-like, (num_out, num_in)
        Direct gain.
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
        same hook :func:`pyFDN.process_dss` calls ``post_delay``. An
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
    core = fdn_core(
        delays,
        A,
        B,
        C,
        D,
        fs,
        nfft,
        device=device,
        dtype=dtype,
        post_delay=post_delay,
        post_matrix=post_matrix,
        post_output=post_output,
    )
    if shell:
        from pyFDN.auxiliary.flamo import wrap_fdn_shell

        return wrap_fdn_shell(core, nfft=nfft, dtype=dtype)
    return core


def fdn_core(
    delays: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    fs: float,
    nfft: int,
    *,
    device: Any = None,
    dtype: Any = None,
    alias_decay_db: float = 0.0,
    feedback: Any = None,
    trainable: Any = None,
    post_delay: Any = None,
    post_matrix: Any = None,
    post_output: Any = None,
) -> Any:
    """The FLAMO FDN core shared by :func:`dss_to_flamo` and training.

    ``feedback`` replaces the default fixed feedback module (a parametrized
    matrix, when training); ``trainable`` names which of ``input_gain``,
    ``output_gain`` and ``direct`` require gradients. Hooks are resolved by
    :func:`pyFDN.hook_module`, so a baked SOS bank becomes a frozen filter.
    """
    from pyFDN.auxiliary.flamo import (
        assemble_fdn_core,
        default_device,
        fir_matrix_module,
        gain_module,
        hook_module,
    )

    A = np.asarray(A, dtype=np.float64)
    delays_arr = np.asarray(delays, dtype=np.float64).ravel()
    if delays_arr.shape[0] != A.shape[0]:
        raise ValueError("delays must have length N (number of delay lines)")
    device = default_device(device)
    common = {"device": device, "dtype": dtype, "alias_decay_db": alias_decay_db}

    def gains(values: ArrayLike, name: str) -> Any:
        requires_grad = bool(trainable is not None and getattr(trainable, name))
        return gain_module(
            np.asarray(values), nfft, requires_grad=requires_grad, **common
        )

    if feedback is None:
        feedback = (
            fir_matrix_module(A, nfft, device=device, dtype=dtype)
            if A.ndim == 3
            else gain_module(A, nfft, **common)
        )
    hooks = {
        name: hook_module(value, nfft, name=name, **common)
        for name, value in (
            ("post_delay", post_delay),
            ("post_matrix", post_matrix),
            ("post_output", post_output),
        )
    }
    # One assembler for render and training, so the topology (and the leaf
    # names extract_build reads back) cannot drift apart.
    return assemble_fdn_core(
        input_gain=gains(B, "input_gain"),
        feedback=feedback,
        delays=delay_module(delays_arr / float(fs), nfft, fs=fs, **common),
        output_gain=gains(C, "output_gain"),
        direct=gains(D, "direct"),
        **hooks,
    )


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
    :class:`~pyFDN.FDNBuild` (as returned by
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
        build.delays,
        build.A,
        build.B,
        build.C,
        build.D,
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
