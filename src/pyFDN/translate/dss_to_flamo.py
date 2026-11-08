"""
Convert a delay state-space (DSS) system (A, B, C, D, delays) to a FLAMO model
for rendering.

Optionally place an allpass (or other) filter behind the delays in the loop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.flamo import delay_module


def dss_to_flamo(
    delays: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    fs: float,
    *,
    nfft: int = 2**16,
    alias_decay_db: float = 0.0,
    device: Any = None,
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
    A : array-like, (N, N)
        Feedback matrix. A filter (FIR) feedback matrix goes on the feedback
        path instead: ``A=np.eye(N), post_matrix=pyFDN.fir_matrix_module(fir,
        nfft)``.
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
    alias_decay_db : float
        Anti-time-aliasing decay, the accuracy of the rendered response in dB;
        see :func:`pyFDN.trainable_from_dss`. Leave at 0 for a decaying FDN; a
        lossless one needs it, e.g. :data:`pyFDN.LOSSLESS_ALIAS_DECAY_DB`. A
        module passed into a hook must have been built with the same value.
    device : torch device or None
        Device; default is cuda if available else cpu.
    dtype : torch.dtype or None
        Optional dtype for FLAMO delay/gain/filter modules (e.g., torch.float64).
        If None, wrapper defaults are used.
    post_delay : array, FLAMO module, sequence, or None
        In-loop filter applied to the delay output, inside the recursion -- the
        same hook :func:`pyFDN.dss_to_td` calls ``post_delay``. An
        ``(n_sections, 6, N)`` SOS bank, a FLAMO module of input/output size N
        (e.g. an allpass FDN's ``dss_to_flamo(...).get_core()``), or a sequence of
        both applied in order. See ``pyFDN.auxiliary.flamo.hook_module``.
    post_matrix : array, FLAMO module, sequence, or None
        Filter applied to the feedback path after ``A``.
    post_output : array, FLAMO module, sequence, or None
        Per-output filter applied to the wet signal after ``C``; an
        ``(n_sections, 6, num_out)`` SOS bank, or a module.

    Returns
    -------
    flamo.processor.system.Shell
        Use :func:`pyFDN.flamo_to_impz` for a NumPy impulse response,
        and ``.get_core()`` for the FDN without its FFT layers, e.g. to nest
        it in another FDN's hook.
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
        alias_decay_db=alias_decay_db,
        post_delay=post_delay,
        post_matrix=post_matrix,
        post_output=post_output,
    )
    from pyFDN.auxiliary.flamo import wrap_fdn_shell

    return wrap_fdn_shell(core, nfft=nfft, dtype=dtype, device=device)


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
    ``pyFDN.auxiliary.flamo.hook_module``, so a baked SOS bank becomes a frozen filter.
    """
    from pyFDN.auxiliary.flamo import (
        assemble_fdn_core,
        default_device,
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
        feedback = gain_module(A, nfft, **common)
    hooks = {
        name: hook_module(value, nfft, name=name, **common)
        for name, value in (
            ("post_delay", post_delay),
            ("post_matrix", post_matrix),
            ("post_output", post_output),
        )
    }
    # One assembler for render and training, so the topology (and the leaf
    # names flamo_to_dss reads back) cannot drift apart.
    return assemble_fdn_core(
        input_gain=gains(B, "input_gain"),
        feedback=feedback,
        delays=delay_module(delays_arr, nfft, fs=fs, **common),
        output_gain=gains(C, "output_gain"),
        direct=gains(D, "direct"),
        **hooks,
    )
