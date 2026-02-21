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


def gain_module(
    values: np.ndarray,
    nfft: int,
    *,
    device=None,
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

    gain = dsp.Gain(
        size=(n_out, n_in),
        nfft=nfft,
        requires_grad=requires_grad,
        alias_decay_db=alias_decay_db,
        device=dev,
    )
    gain.assign_value(torch.as_tensor(values, dtype=torch.float64, device=dev))
    return gain


def delay_module(
    lengths_seconds: np.ndarray,
    nfft: int,
    *,
    Fs: float,
    device=None,
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
    )
    delays.assign_value(torch.as_tensor(lengths, dtype=torch.float32, device=dev))
    return delays


def sos_filter_module(
    sos: np.ndarray,
    nfft: int,
    *,
    device=None,
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
    filt = dsp.parallelSOSFilter(
        size=(N,),
        n_sections=n_sections,
        nfft=nfft,
        device=dev,
    )
    filt.assign_value(torch.as_tensor(sos_pad, dtype=torch.float32, device=dev))
    return filt
