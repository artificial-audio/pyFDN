"""
Standard wrappers for FLAMO modules that accept numpy arrays and return FLAMO modules.

All functions require flamo to be installed. They take numpy arrays and common
options (nfft, device, etc.) and return configured FLAMO dsp modules with
values assigned.
"""

from __future__ import annotations

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
