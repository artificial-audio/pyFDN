"""
Standard wrappers for FLAMO modules that accept numpy arrays and return FLAMO modules.

All functions require flamo to be installed. They take numpy arrays and common
options (nfft, device, etc.) and return configured FLAMO dsp modules with
values assigned.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

try:
    from flamo.processor import dsp

    _HAS_FLAMO = True
except ImportError:
    _HAS_FLAMO = False


def _get_device(device):
    if device is None and _HAS_FLAMO:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def flamo_time_response(
    model,
    fs: int = 48000,
    identity: bool = False,
) -> np.ndarray:
    """Return a FLAMO model's time response as a NumPy array.

    This is the NumPy-facing counterpart of FLAMO's
    ``model.get_time_response()``. It detaches the returned tensor from any
    autograd graph, transfers it to CPU memory, and preserves its dimensions
    and dtype during conversion.

    Parameters
    ----------
    model
        FLAMO model exposing ``get_time_response``.
    fs : int
        Sampling frequency passed to FLAMO.
    identity : bool
        Whether to request FLAMO's input-free identity response.

    Returns
    -------
    np.ndarray
        Time response with the same shape and numeric dtype as FLAMO's tensor.
    """
    response = model.get_time_response(fs=fs, identity=identity)
    if hasattr(response, "detach"):
        response = response.detach()
    if hasattr(response, "cpu"):
        response = response.cpu()
    return np.asarray(response)


def flamo_freq_response(
    model,
    fs: int = 48000,
    identity: bool = False,
) -> np.ndarray:
    """Return a FLAMO model's (complex) frequency response as a NumPy array.

    The NumPy-facing counterpart of FLAMO's ``model.get_freq_response()`` and the
    frequency-domain sibling of :func:`flamo_time_response`. It detaches the
    returned tensor from any autograd graph, transfers it to CPU memory, and
    preserves its shape and (complex) dtype. Take ``np.abs(...)`` for the
    magnitude response, ``np.angle(...)`` for the phase.

    ``get_freq_response`` evaluates over ``nfft`` DFT bins by temporarily swapping
    the model's input/output layers to FFT and restoring them before returning,
    so this is side-effect-free regardless of the model's current output layer.

    Parameters
    ----------
    model
        FLAMO model exposing ``get_freq_response`` (e.g. a ``Shell``).
    fs : int
        Sampling frequency passed to FLAMO.
    identity : bool
        Whether to request FLAMO's input-free identity response.

    Returns
    -------
    np.ndarray
        Complex frequency response with the same shape and numeric dtype as
        FLAMO's tensor.
    """
    response = model.get_freq_response(fs=fs, identity=identity)
    if hasattr(response, "detach"):
        response = response.detach()
    if hasattr(response, "cpu"):
        response = response.cpu()
    return np.asarray(response)


def flamo_process(
    model,
    signal: np.ndarray,
    *,
    fs: int | None = None,
    tail_seconds: float = 0.0,
    dtype=None,
) -> np.ndarray:
    """Run a 1-D signal through a FLAMO ``Shell`` model offline.

    Wraps the boilerplate of turning a NumPy signal into the
    ``(batch, time, channel)`` tensor FLAMO expects, running a no-grad
    forward pass, and converting the result back to NumPy.

    The model convolves in the frequency domain over a block of length
    ``nfft`` (read from the model's input layer), so the signal is
    truncated or zero-padded to ``nfft``. Because that is a *circular*
    convolution, a long reverb tail can wrap around onto the start of the
    block; pass ``tail_seconds`` to reserve that much trailing silence for
    the tail to decay into (requires ``fs``).

    Parameters
    ----------
    model
        FLAMO ``Shell`` whose input layer exposes ``nfft`` (e.g. the output
        of :func:`pyFDN.dss_to_flamo`).
    signal : np.ndarray
        1-D input signal.
    fs : int, optional
        Sampling rate, required only when ``tail_seconds > 0``.
    tail_seconds : float
        Trailing silence to reserve so the reverb tail does not wrap around.
    dtype : torch.dtype or None
        Tensor dtype for the forward pass; defaults to float32.

    Returns
    -------
    np.ndarray
        Squeezed model output on CPU.
    """
    if not _HAS_FLAMO:
        raise ImportError("flamo_process requires flamo (pip install flamo)")
    import torch

    nfft = int(model.get_inputLayer().nfft)
    sig = np.asarray(signal, dtype=np.float64).ravel()

    if tail_seconds:
        if fs is None:
            raise ValueError("fs is required when tail_seconds > 0")
        usable = max(0, nfft - int(round(tail_seconds * fs)))
    else:
        usable = nfft

    buf = np.zeros(nfft, dtype=np.float64)
    n = min(len(sig), usable)
    buf[:n] = sig[:n]

    torch_dtype = torch.float32 if dtype is None else dtype
    x = torch.as_tensor(buf, dtype=torch_dtype).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        wet = model(x)
    return np.asarray(wet.squeeze().detach().cpu())


def gain_module(
    values: np.ndarray,
    nfft: int,
    *,
    device=None,
    dtype=None,
    alias_decay_db: float = 0,
    requires_grad: bool = False,
):
    """
    Build a FLAMO Gain module from a numpy array.

    Parameters
    ----------
    values : np.ndarray
        Gain matrix, shape (n_output, n_input). Will be cast to float64.
    nfft : int
        FFT size for the FLAMO module.
    device : torch device or None
        Device for the module; default is cuda if available else cpu.
    dtype : torch.dtype or None
        Optional dtype for module parameters (e.g., torch.float64).
        If None, uses float32.
    alias_decay_db : float
        FLAMO alias decay in dB.
    requires_grad : bool
        Whether the gain parameters are trainable.

    Returns
    -------
    flamo.processor.dsp.Gain
        FLAMO Gain module with values assigned.
    """
    if not _HAS_FLAMO:
        raise ImportError("gain_module requires flamo (pip install flamo)")
    import torch

    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    n_out, n_in = values.shape
    dev = _get_device(device)

    torch_dtype = torch.float32 if dtype is None else dtype
    gain = dsp.Gain(
        size=(n_out, n_in),
        nfft=nfft,
        requires_grad=requires_grad,
        alias_decay_db=alias_decay_db,
        device=dev,
        dtype=torch_dtype,
    )
    gain.assign_value(torch.as_tensor(values, dtype=torch_dtype, device=dev))
    return gain


def delay_module(
    lengths_seconds: np.ndarray,
    nfft: int,
    *,
    Fs: float,
    device=None,
    dtype=None,
    isint: bool = True,
    alias_decay_db: float = 0,
    requires_grad: bool = False,
):
    """
    Build a FLAMO parallelDelay module from delay lengths in seconds.

    Values are assigned directly (no sample conversion); buffer size is derived from Fs.

    Parameters
    ----------
    lengths_seconds : np.ndarray
        1D array of delay lengths in seconds, one per channel.
    nfft : int
        FFT size for the FLAMO module.
    Fs : float
        Sampling rate in Hz (used for buffer size max_len = max(lengths_seconds) * Fs).
    device : torch device or None
        Device for the module; default is cuda if available else cpu.
    dtype : torch.dtype or None
        Optional dtype for module parameters (e.g., torch.float64).
        If None, uses float32 to preserve previous behavior.
    isint : bool
        Whether delays are integer-sample (True) or fractional.
    alias_decay_db : float
        FLAMO alias decay in dB.
    requires_grad : bool
        Whether the delay parameters are trainable.

    Returns
    -------
    flamo.processor.dsp.parallelDelay
        FLAMO parallelDelay module with lengths assigned (in seconds).
    """
    if not _HAS_FLAMO:
        raise ImportError("delay_module requires flamo (pip install flamo)")
    import torch

    lengths = np.asarray(lengths_seconds, dtype=np.float64).ravel()
    n = len(lengths)
    max_len = int(np.ceil(np.max(lengths) * Fs)) if n else 1
    max_len = max(1, max_len)
    dev = _get_device(device)

    torch_dtype = torch.float32 if dtype is None else dtype
    delays = dsp.parallelDelay(
        size=(n,),
        max_len=max_len,
        nfft=nfft,
        isint=isint,
        unit=1,
        fs=Fs,
        requires_grad=requires_grad,
        alias_decay_db=alias_decay_db,
        device=dev,
        dtype=torch_dtype,
    )
    delays.assign_value(torch.as_tensor(lengths, dtype=torch_dtype, device=dev))
    return delays


def fir_matrix_module(
    coeffs: np.ndarray,
    nfft: int,
    *,
    device=None,
    dtype=None,
    requires_grad: bool = False,
):
    """
    Build a FLAMO Filter module from a matrix FIR coefficient array.

    Parameters
    ----------
    coeffs : np.ndarray
        FIR matrix in z^{-1} convention, shape (n_output, n_input, n_taps)
        (e.g. a paraunitary feedback matrix).
    nfft : int
        FFT size for the FLAMO module.
    device : torch device or None
        Device for the module; default is cuda if available else cpu.
    dtype : torch.dtype or None
        Optional dtype for module parameters (e.g., torch.float64).
        If None, uses float32.
    requires_grad : bool
        Whether the filter parameters are trainable.

    Returns
    -------
    flamo.processor.dsp.Filter
        FLAMO Filter module with coefficients assigned.
    """
    if not _HAS_FLAMO:
        raise ImportError("fir_matrix_module requires flamo (pip install flamo)")
    import torch

    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.ndim != 3:
        raise ValueError("coeffs must have shape (n_output, n_input, n_taps)")
    n_out, n_in, n_taps = coeffs.shape

    dev = _get_device(device)
    torch_dtype = torch.float32 if dtype is None else dtype
    filt = dsp.Filter(
        size=(n_taps, n_out, n_in),
        nfft=nfft,
        requires_grad=requires_grad,
        device=dev,
        dtype=torch_dtype,
    )
    filt.assign_value(
        torch.as_tensor(coeffs.transpose(2, 0, 1), dtype=torch_dtype, device=dev)
    )
    return filt


def sos_filter_module(
    sos: np.ndarray,
    nfft: int,
    *,
    device=None,
    dtype=None,
    requires_grad: bool = False,
):
    """
    Build a FLAMO parallelSOSFilter from an SOS coefficient array.

    Parameters
    ----------
    sos : np.ndarray
        Shape (n_sections, 6, n_channels). Each section is [b0, b1, b2, a0, a1, a2] (e.g. from SDN wall_filters_sos).
    nfft : int
        FFT size for the FLAMO module.
    device : torch device or None
        Device for the module; default is cuda if available else cpu.
    dtype : torch.dtype or None
        Optional dtype for module parameters (e.g., torch.float64).
        If None, uses float32 to preserve previous behavior.
    requires_grad : bool
        Whether the filter parameters are trainable.

    Returns
    -------
    flamo.processor.dsp.parallelSOSFilter
        FLAMO parallelSOSFilter with coefficients assigned.
    """
    if not _HAS_FLAMO:
        raise ImportError("sos_filter_module requires flamo (pip install flamo)")
    import torch

    sos_pad = np.asarray(sos, dtype=np.float64)
    if sos_pad.ndim != 3 or sos_pad.shape[1] != 6:
        raise ValueError("sos must have shape (n_sections, 6, n_channels)")
    n_sections, _, N = sos_pad.shape
    if N == 0:
        raise ValueError("sos must have at least one channel")

    dev = _get_device(device)
    torch_dtype = torch.float32 if dtype is None else dtype
    filt = dsp.parallelSOSFilter(
        size=(N,),
        n_sections=n_sections,
        nfft=nfft,
        device=dev,
        dtype=torch_dtype,
    )
    filt.assign_value(torch.as_tensor(sos_pad, dtype=torch_dtype, device=dev))
    return filt


def _matrix_preimage(values: np.ndarray, matrix_type: str) -> np.ndarray:
    """Pre-image ``param`` so a flamo ``Matrix.map(param)`` realizes ``values``.

    * ``"random"`` -- identity map, so the pre-image is ``values`` itself.
    * ``"orthogonal"`` -- map is ``matrix_exp(skew_matrix(param))``, which spans
      SO(N). The pre-image is the real matrix logarithm (a skew-symmetric matrix
      that ``skew_matrix`` reproduces). If ``values`` has ``det < 0`` it is not
      in SO(N); the last column is sign-flipped to the nearest SO(N) matrix and a
      warning is emitted.
    """
    if matrix_type == "random":
        return values
    if matrix_type == "orthogonal":
        from scipy.linalg import logm

        a = np.asarray(values, dtype=np.float64)
        if np.linalg.det(a) < 0:
            warnings.warn(
                "orthogonal feedback matrix has det<0 (not in SO(N)); flipping "
                "the last column to the nearest SO(N) matrix for the trainable "
                "orthogonal parametrization",
                stacklevel=3,
            )
            a = a.copy()
            a[:, -1] *= -1.0
        return np.real(logm(a))
    raise ValueError(
        f"matrix_type must be 'orthogonal' or 'random', got {matrix_type!r}"
    )


def matrix_module(
    values: np.ndarray,
    nfft: int,
    *,
    matrix_type: str = "orthogonal",
    device: Any = None,
    dtype: Any = None,
    alias_decay_db: float = 0,
    requires_grad: bool = False,
):
    """
    Build a FLAMO ``Matrix`` initialized to ``values`` under a parametrization.

    Unlike :func:`gain_module` (a plain value container), this preserves the
    flamo ``map`` that constrains the trainable matrix: ``"orthogonal"`` keeps it
    on the SO(N) manifold during optimization, ``"random"`` is unconstrained.

    Parameters
    ----------
    values : np.ndarray
        Square ``(N, N)`` initial feedback matrix.
    nfft : int
        FFT size for the FLAMO module.
    matrix_type : str
        ``"orthogonal"`` or ``"random"``.
    device : torch device or None
        Device; default is cuda if available else cpu.
    dtype : torch.dtype or None
        Module dtype; defaults to float32.
    alias_decay_db : float
        FLAMO alias decay in dB.
    requires_grad : bool
        Whether the matrix is trainable.

    Returns
    -------
    flamo.processor.dsp.Matrix
        Matrix whose realized value (``map(param)``) equals ``values`` (within
        the parametrization; an SO(N) projection may apply for orthogonal).
    """
    if not _HAS_FLAMO:
        raise ImportError("matrix_module requires flamo (pip install flamo)")
    import torch

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix values must be square (N, N)")
    n = values.shape[0]
    dev = _get_device(device)
    torch_dtype = torch.float32 if dtype is None else dtype

    matrix = dsp.Matrix(
        size=(n, n),
        nfft=nfft,
        matrix_type=matrix_type,
        requires_grad=requires_grad,
        alias_decay_db=alias_decay_db,
        device=dev,
        dtype=torch_dtype,
    )
    preimage = _matrix_preimage(values, matrix_type)
    matrix.assign_value(torch.as_tensor(preimage, dtype=torch_dtype, device=dev))
    return matrix


def assemble_fdn_core(
    *,
    input_gain: Any,
    feedback: Any,
    delays: Any,
    output_gain: Any,
    direct: Any,
    loop_filter: Any = None,
    output_filter: Any = None,
    post_delay_module: Any = None,
) -> Any:
    """
    Wire pre-built FLAMO modules into an FDN core (no FFT/iFFT wrapping).

    Returns ``Parallel(brA=fdn_branch, brB=direct)``. Signal flow::

        input_gain -> [recursion: delay -> (loop_filter) -> (post_delay_module); fB = feedback]
                   -> output_gain -> (output_filter)   +   direct

    Parameters
    ----------
    input_gain, output_gain : FLAMO modules
        Input gain ``B`` (leaf ``input_gain``) and output gain ``C`` (leaf
        ``output_gain``).
    feedback : FLAMO module
        Feedback matrix (leaf ``mixing_matrix``).
    delays : FLAMO module
        Delay module (leaf ``delay``).
    direct : FLAMO module
        Direct path ``D`` (leaf ``direct_gain``). Pass a zero gain for no direct
        path.
    loop_filter : FLAMO module or None
        Optional in-loop filter after the delays (leaf ``filter``).
    output_filter : FLAMO module or None
        Optional per-output filter after the output gain (leaf ``output_filter``).
    post_delay_module : FLAMO module or None
        Optional module appended after the delay in the recursion.

    Returns
    -------
    core : flamo.processor.system.Parallel
    """
    if not _HAS_FLAMO:
        raise ImportError("assemble_fdn_core requires flamo (pip install flamo)")
    from collections import OrderedDict

    from flamo.processor import system

    from pyFDN.auxiliary import flamo_graph as layout

    forward = OrderedDict({layout.DELAY: delays})
    if loop_filter is not None:
        forward[layout.LOOP_FILTER] = loop_filter
    if post_delay_module is not None:
        forward["post_delay_module"] = post_delay_module

    feedback_branch = system.Series(OrderedDict({layout.MIXING_MATRIX: feedback}))
    feedback_loop = system.Recursion(fF=system.Series(forward), fB=feedback_branch)
    fdn_modules = OrderedDict(
        {
            layout.INPUT_GAIN: input_gain,
            "feedback_loop": feedback_loop,
            layout.OUTPUT_GAIN: output_gain,
        }
    )
    if output_filter is not None:
        fdn_modules[layout.OUTPUT_FILTER] = output_filter
    fdn_branch = system.Series(fdn_modules)

    direct_branch = system.Series(OrderedDict({layout.DIRECT: direct}))
    return system.Parallel(brA=fdn_branch, brB=direct_branch, sum_output=True)


def output_layer(output: str, nfft: int, dtype: Any = None) -> Any:
    """Build the FLAMO output layer for an output domain.

    ``"time"`` -> ``iFFT`` (time response); ``"magnitude"`` -> ``|.|`` of the
    frequency response.
    """
    if not _HAS_FLAMO:
        raise ImportError("output_layer requires flamo (pip install flamo)")
    import torch

    torch_dtype = torch.float32 if dtype is None else dtype
    if output == "time":
        return dsp.iFFT(nfft, dtype=torch_dtype)
    if output == "magnitude":
        return dsp.Transform(transform=torch.abs, dtype=torch_dtype)
    raise ValueError(f"output must be 'time' or 'magnitude', got {output!r}")


def wrap_fdn_shell(
    core: Any, *, nfft: int, dtype: Any = None, output: str = "time"
) -> Any:
    """
    Wrap an FDN core in a FLAMO ``Shell`` with an FFT input layer.

    Parameters
    ----------
    core : FLAMO module
        FDN core, e.g. from :func:`assemble_fdn_core`.
    nfft : int
        FFT size.
    dtype : torch.dtype or None
        Dtype for the FFT/iFFT layers; defaults to float32.
    output : str
        Output-domain layer: ``"time"`` (``iFFT`` time response) or
        ``"magnitude"`` (``|.|`` of the frequency response).

    Returns
    -------
    flamo.processor.system.Shell
    """
    if not _HAS_FLAMO:
        raise ImportError("wrap_fdn_shell requires flamo (pip install flamo)")
    import torch
    from flamo.processor import dsp, system

    torch_dtype = torch.float32 if dtype is None else dtype
    return system.Shell(
        core=core,
        input_layer=dsp.FFT(nfft, dtype=torch_dtype),
        output_layer=output_layer(output, nfft, torch_dtype),
    )
