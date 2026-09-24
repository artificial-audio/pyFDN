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


def _get_device(device: Any) -> Any:
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
    fs: float,
    device=None,
    dtype=None,
    isint: bool = True,
    alias_decay_db: float = 0,
    requires_grad: bool = False,
):
    """
    Build a FLAMO parallelDelay module from delay lengths in seconds.

    Values are assigned directly (no sample conversion); buffer size is derived from fs.

    Parameters
    ----------
    lengths_seconds : np.ndarray
        1D array of delay lengths in seconds, one per channel.
    nfft : int
        FFT size for the FLAMO module.
    fs : float
        Sampling rate in Hz (used for buffer size max_len = max(lengths_seconds) * fs).
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
    max_len = int(np.ceil(np.max(lengths) * fs)) if n else 1
    max_len = max(1, max_len)
    dev = _get_device(device)

    torch_dtype = torch.float32 if dtype is None else dtype
    delays = dsp.parallelDelay(
        size=(n,),
        max_len=max_len,
        nfft=nfft,
        isint=isint,
        unit=1,
        fs=fs,
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
    alias_decay_db: float = 0,
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
    alias_decay_db : float
        FLAMO alias decay in dB.
    requires_grad : bool
        Whether the SOS coefficients are trainable. The sections are normalized
        to ``a0 = 1`` here rather than by flamo's ``normalize_a0`` map, whose
        in-place writes break autograd; ``a0`` is then held at 1 by masking its
        gradient, so the trained coefficients stay a valid SOS array.

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

    a0 = sos_pad[:, 3:4, :]
    if np.any(a0 == 0):
        raise ValueError("sos has a section with a0 = 0")
    sos_pad = sos_pad / a0

    dev = _get_device(device)
    torch_dtype = torch.float32 if dtype is None else dtype
    filt = dsp.parallelSOSFilter(
        size=(N,),
        n_sections=n_sections,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        device=dev,
        dtype=torch_dtype,
        # normalization already done above, in numpy: flamo's normalize_a0 map
        # writes in place, which autograd refuses to differentiate through.
        normalize_a0=False,
    )
    filt.assign_value(torch.as_tensor(sos_pad, dtype=torch_dtype, device=dev))
    if requires_grad:
        filt.param.requires_grad_(True)
        # a0 is redundant with the section's overall scale; with the normalizing
        # map gone, hold it at 1 by dropping its gradient.
        mask = torch.ones_like(filt.param)
        mask[:, 3, :] = 0.0
        filt.param.register_hook(lambda grad, mask=mask: grad * mask)
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


def hook_module(
    value: Any,
    nfft: int,
    *,
    name: str,
    device: Any = None,
    dtype: Any = None,
    alias_decay_db: float = 0.0,
) -> Any:
    """One FLAMO module for a filter hook, from an SOS bank, a module, or several.

    The three hooks of :func:`assemble_fdn_core` each take a single module, but
    what a caller has is often an ``(n_sections, 6, n_channels)`` SOS array, and
    occasionally more than one thing to put in the same position. This resolves
    all three cases:

    * ``None`` -- no module.
    * an array -- built with :func:`sos_filter_module`.
    * a FLAMO module -- passed through.
    * a sequence of the above -- composed into a ``Series`` in the given order,
      with leaves named ``{name}_0``, ``{name}_1``, ...

    Parameters
    ----------
    value : None, array, FLAMO module, or sequence
        What to put in the hook.
    nfft : int
        FFT size, matching the rest of the model.
    name : str
        The hook's name (``"post_delay"``, ``"post_matrix"``, ``"post_output"``),
        used to name the leaves of a composed ``Series``.
    """
    if value is None:
        return None
    if isinstance(value, list | tuple):
        if not value:
            return None
        parts = [
            hook_module(
                item,
                nfft,
                name=name,
                device=device,
                dtype=dtype,
                alias_decay_db=alias_decay_db,
            )
            for item in value
        ]
        if len(parts) == 1:
            return parts[0]
        if not _HAS_FLAMO:
            raise ImportError("hook_module requires flamo (pip install flamo)")
        from collections import OrderedDict

        from flamo.processor import system

        return system.Series(
            OrderedDict((f"{name}_{i}", part) for i, part in enumerate(parts))
        )
    if isinstance(value, np.ndarray) or not hasattr(value, "forward"):
        return sos_filter_module(
            np.asarray(value, dtype=np.float64),
            nfft,
            device=device,
            dtype=dtype,
            alias_decay_db=alias_decay_db,
        )
    return value


def assemble_fdn_core(
    *,
    input_gain: Any,
    feedback: Any,
    delays: Any,
    output_gain: Any,
    direct: Any = None,
    post_delay: Any = None,
    post_matrix: Any = None,
    post_output: Any = None,
) -> Any:
    """
    Wire pre-built FLAMO modules into an FDN core (no FFT/iFFT wrapping).

    Single source of truth for the FDN signal flow, shared by the render path
    (:func:`pyFDN.dss_to_flamo`) and the training builder
    (:func:`pyFDN.train.trainable_from_build`). All arguments are already-built
    FLAMO ``dsp``/``system`` modules; this only composes them, so leaf names and
    topology stay identical across both callers (and match the names
    :func:`pyFDN.extract_build` requires).

    Signal flow::

        input_gain -> [recursion: fF = delay -> (post_delay)
                                  fB = feedback -> (post_matrix)]
                   -> output_gain -> (post_output)

    with the direct path ``direct`` summed in parallel when provided.

    The three optional filter slots are the same three hooks, in the same three
    positions and under the same three names, that :func:`pyFDN.process_dss`
    takes in NumPy -- ``post_delay`` on the shared delay output (so it shapes
    both what leaves the network and what is fed back), ``post_matrix`` on the
    feedback path only, ``post_output`` on the wet signal only. An
    :class:`~pyFDN.FDNBuild` has a field of each name, holding the SOS bank a
    hook bakes down to; a hook holding something that does not bake -- a nested
    core, a time-varying matrix -- simply has no build field to go in.

    Parameters
    ----------
    input_gain, output_gain : FLAMO modules
        Input gain ``B`` (named ``input_gain``) and output gain ``C`` (named
        ``output_gain``).
    feedback : FLAMO module
        Feedback matrix placed on the recursion feedback branch (``fB``); a
        plain ``Gain``/``Filter`` (render) or a parametrized ``Matrix``
        (training).
    delays : FLAMO module
        Delay module on the recursion forward branch (named ``delay``).
    direct : FLAMO module or None
        Direct path ``D``. When ``None`` the core is the plain feedforward
        ``Series`` (no ``Parallel`` wrapper) -- this keeps ``core.feedback_loop``
        reachable for losses such as ``sparsity_loss``. When provided the core
        is ``Parallel(brA=fdn_branch, brB=direct)``.
    post_delay : FLAMO module or None
        In-loop filter after the delays (named ``post_delay``). Any module of
        input/output size N: a :class:`~pyFDN.AttenuationFilter` or a plain
        :func:`sos_filter_module` for attenuation, or a whole nested core such as
        a Schroeder allpass. Only an SOS filter here is extractable into an
        :class:`~pyFDN.FDNBuild`.
    post_matrix : FLAMO module or None
        Filter on the feedback path after the feedback matrix (named
        ``post_matrix``); the position a time-varying mixing stage occupies.
        Adding it names the feedback branch ``mixing_matrix`` rather than
        leaving the matrix bare on ``fB``, which is where
        :func:`pyFDN.extract_build` looks for it either way.
    post_output : FLAMO module or None
        Per-output filter after the output gain (named ``post_output``);
        typically an :class:`OutputEQ`.

    Returns
    -------
    core : flamo.processor.system.Series or Parallel
        The FDN core, ready for :func:`wrap_fdn_shell`.
    """
    if not _HAS_FLAMO:
        raise ImportError("assemble_fdn_core requires flamo (pip install flamo)")
    from collections import OrderedDict

    from flamo.processor import system

    if post_delay is not None:
        forward = system.Series(
            OrderedDict({"delay": delays, "post_delay": post_delay})
        )
    else:
        forward = delays

    if post_matrix is not None:
        back = system.Series(
            OrderedDict({"mixing_matrix": feedback, "post_matrix": post_matrix})
        )
    else:
        back = feedback

    feedback_loop = system.Recursion(fF=forward, fB=back)
    fdn_modules = OrderedDict(
        {
            "input_gain": input_gain,
            "feedback_loop": feedback_loop,
            "output_gain": output_gain,
        }
    )
    if post_output is not None:
        fdn_modules["post_output"] = post_output
    fdn_branch = system.Series(fdn_modules)

    if direct is not None:
        return system.Parallel(brA=fdn_branch, brB=direct, sum_output=True)
    return fdn_branch


def core_alias_decay_db(core: Any) -> float:
    """The anti-aliasing decay the FLAMO ``core`` was built with, in dB.

    FLAMO containers (``Series``/``Parallel``/``Recursion``) assert that every
    module agrees on ``alias_decay_db``, so the core is the single source of
    truth -- reading it back beats threading the value through by hand and
    risking a mismatch with the modules.
    """
    value = getattr(core, "alias_decay_db", None)
    if value is None:
        return 0.0
    return abs(float(value))


def wrap_fdn_shell(core: Any, *, nfft: int, dtype: Any = None) -> Any:
    r"""
    Wrap an FDN core in a FLAMO ``Shell`` that returns the impulse response.

    The shell is FFT in, impulse response out: the input layer is an ``FFT``
    and the output layer the ``iFFTAntiAlias`` that matches the core's own
    ``alias_decay_db``. A pyFDN model therefore means one thing wherever it is
    used -- rendered, analyzed or trained -- and the time domain is a property
    of how the model was built rather than something a caller sets afterwards.

    The core evaluates the system on a circle of radius :math:`\gamma < 1`, so
    its response carries a :math:`\gamma^n` envelope; the output layer removes
    it again. What comes out is the true impulse response, accurate to
    ``alias_decay_db`` (see :func:`pyFDN.trainable_from_build`). At
    ``alias_decay_db=0`` the layer is an ordinary inverse FFT.

    Parameters
    ----------
    core : FLAMO module
        FDN core, e.g. from :func:`assemble_fdn_core`. Its ``alias_decay_db``
        is read back off it, so the output layer cannot disagree with the
        modules it undoes.
    nfft : int
        FFT size.
    dtype : torch.dtype or None
        Dtype for the FFT/iFFT layers; defaults to float32.

    Returns
    -------
    flamo.processor.system.Shell

    See Also
    --------
    pyFDN.model_response : the shell's output as a :class:`~pyFDN.Response`,
        including the magnitude spectrum a frequency-domain view wants.
    """
    if not _HAS_FLAMO:
        raise ImportError("wrap_fdn_shell requires flamo (pip install flamo)")
    import torch
    from flamo.processor import dsp, system

    torch_dtype = torch.float32 if dtype is None else dtype
    return system.Shell(
        core=core,
        input_layer=dsp.FFT(nfft, dtype=torch_dtype),
        output_layer=dsp.iFFTAntiAlias(
            nfft, alias_decay_db=core_alias_decay_db(core), dtype=torch_dtype
        ),
    )
