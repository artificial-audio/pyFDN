"""Build a trainable flamo FDN model from a config.

:func:`build_fdn` turns a config (delays/N, decay, what is trainable) into a
trainable flamo ``Shell`` you can render, train, and extract.
:func:`trainable_from_build` does the same starting from an existing
:class:`~pyFDN.FDNBuild`.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from pyFDN.generate.fdn_matrix_gallery import FDNBuild

# Feedback-matrix parametrization: "orthogonal" keeps the matrix on SO(N) during
# training (the colorless choice); "random" trains it unconstrained.
MatrixParam = Literal["orthogonal", "random"]


@dataclass(frozen=True)
class Trainable:
    """Which FDN parameter groups are trained. Delays are always fixed."""

    feedback: bool = True
    input_gain: bool = True
    output_gain: bool = True
    direct: bool = False


def build_fdn(
    *,
    delays: np.ndarray | None = None,
    N: int | None = None,
    rt: float | tuple[float, float] | None = 2.0,
    matrix: MatrixParam = "orthogonal",
    feedback: np.ndarray | None = None,
    input_gain: np.ndarray | None = None,
    output_gain: np.ndarray | None = None,
    direct: float | np.ndarray = 0.0,
    trainable: Trainable | None = None,
    loop_filter: Any | None = None,
    fs: float = 48000.0,
    nfft: int = 2**14,
    output: str = "time",
    device: Any = None,
    dtype: Any = None,
    rng: np.random.Generator | int | None = None,
) -> Any:
    """Build a trainable flamo ``Shell`` from a config.

    Parameters
    ----------
    delays : np.ndarray, optional
        Explicit integer delay lengths in samples. If omitted, ``N`` coprime
        delays are sampled (:func:`pyFDN.sample_delay_lengths`).
    N : int, optional
        Number of delay lines when ``delays`` is omitted.
    rt : float, (rt_dc, rt_nyquist), or None
        Reverberation time in seconds. ``None`` builds a lossless FDN.
    matrix : {"orthogonal", "random"}
        Feedback-matrix parametrization.
    feedback : np.ndarray, optional
        Initial ``(N, N)`` feedback matrix; defaults to a random SO(N) matrix.
    input_gain, output_gain : np.ndarray, optional
        ``B`` (``(N, n_in)``) and ``C`` (``(n_out, N)``); default ones / sqrt(N).
    direct : float or np.ndarray
        Direct path ``D``; a scalar fills ``(n_out, n_in)``.
    trainable : Trainable, optional
        Trainable parameter groups (default :class:`Trainable`).
    loop_filter : flamo dsp module, optional
        A prebuilt flamo filter module to use as the in-loop attenuation filter
        (its own ``requires_grad`` governs trainability). Must match the build:
        ``loop_filter.param.shape[-1] == N`` and ``loop_filter.nfft == nfft``.
        When given it overrides ``rt`` / sampled absorption for the loop slot;
        pass ``rt=None`` to silence the override warning. See
        :func:`trainable_from_build`.
    fs, nfft, output, device, dtype : see :func:`trainable_from_build`.
    rng : np.random.Generator, int, or None
        Seed for the sampled delays / default feedback matrix.

    Returns
    -------
    flamo.processor.system.Shell
    """
    from pyFDN.generate.fdn_matrix_gallery import FDNBuild
    from pyFDN.generate.sample_delay_lengths import sample_delay_lengths

    local_rng = (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    )

    if delays is not None:
        delays_arr = np.asarray(delays, dtype=int).ravel()
        n = int(delays_arr.size)
    elif N is not None:
        n = int(N)
        delays_arr = sample_delay_lengths(n, coprime=True, rng=local_rng)
    else:
        raise ValueError("provide either delays= or N=")

    if feedback is not None:
        a = np.asarray(feedback, dtype=float)
        if a.shape != (n, n):
            raise ValueError(f"feedback must have shape ({n}, {n}), got {a.shape}")
    else:
        a = _random_so_n(n, local_rng)

    # Default IO is ones / sqrt(N); override with input_gain / output_gain.
    b = (
        np.ones((n, 1)) / np.sqrt(n)
        if input_gain is None
        else np.asarray(input_gain, float).reshape(n, -1)
    )
    c = (
        np.ones((1, n)) / np.sqrt(n)
        if output_gain is None
        else np.atleast_2d(np.asarray(output_gain, float))
    )
    n_out, n_in = c.shape[0], b.shape[1]
    d = _resolve_direct(direct, n_out, n_in)

    if loop_filter is not None and rt is not None:
        warnings.warn(
            f"loop_filter provided; rt={rt!r} is ignored for the in-loop filter "
            "(pass rt=None to silence)",
            stacklevel=2,
        )

    filters = None
    if rt is not None and loop_filter is None:
        from pyFDN.auxiliary.acoustics import first_order_absorption

        rt_dc, rt_ny = _rt_pair(rt)
        filters = first_order_absorption(rt_dc, rt_ny, delays_arr, float(fs))

    build = FDNBuild(
        A=a,
        B=b,
        C=c,
        D=d,
        delays=delays_arr,
        fs=float(fs),
        filters=filters,
        post_eq=None,
    )
    return trainable_from_build(
        build,
        trainable=trainable,
        loop_filter=loop_filter,
        matrix=matrix,
        nfft=nfft,
        output=output,
        device=device,
        dtype=dtype,
    )


def trainable_from_build(
    build: FDNBuild,
    *,
    trainable: Trainable | None = None,
    loop_filter: Any | None = None,
    matrix: MatrixParam = "orthogonal",
    nfft: int = 2**14,
    output: str = "time",
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Build a trainable flamo ``Shell`` initialized from an ``FDNBuild``.

    Parameters
    ----------
    build : FDNBuild
        Initial FDN (``A``/``B``/``C``/``D``/``delays``/``fs`` + optional
        ``filters``/``post_eq``).
    trainable : Trainable, optional
        Trainable parameter groups (default :class:`Trainable`).
    loop_filter : flamo dsp module, optional
        A prebuilt flamo filter module placed in the in-loop attenuation slot.
        Trainability comes from the module's own ``requires_grad`` (there is no
        ``Trainable`` flag for it). Must satisfy ``loop_filter.param.shape[-1]``
        ``== N`` (the number of delay lines) and ``loop_filter.nfft == nfft``.
        When given it overrides ``build.filters`` for the loop slot;
        :func:`pyFDN.extract_build` bakes a trained ``parallelSVF`` /
        ``parallelSOSFilter`` back to ``FDNBuild.filters`` (other types raise).
    matrix : {"orthogonal", "random"}
        Feedback-matrix parametrization.
    nfft : int
        FFT size.
    output : str
        ``"time"`` or ``"magnitude"`` output layer (``train_fdn`` sets this to
        match the mode).
    device, dtype : optional
        Torch device / dtype (default cpu-or-cuda / float32).

    Returns
    -------
    flamo.processor.system.Shell
    """
    from pyFDN.auxiliary.flamo import (
        assemble_fdn_core,
        gain_module,
        matrix_module,
        sos_filter_module,
        wrap_fdn_shell,
    )

    trainable = trainable or Trainable()
    fs = float(build.fs)
    b = np.asarray(build.B, dtype=np.float64)
    c = np.asarray(build.C, dtype=np.float64)
    d = (
        np.asarray(build.D, dtype=np.float64)
        if build.D is not None
        else np.zeros((c.shape[0], b.shape[1]))
    )

    input_gain = gain_module(
        b, nfft, device=device, dtype=dtype, requires_grad=trainable.input_gain
    )
    output_gain = gain_module(
        c, nfft, device=device, dtype=dtype, requires_grad=trainable.output_gain
    )
    feedback = matrix_module(
        build.A,
        nfft,
        matrix_type=matrix,
        device=device,
        dtype=dtype,
        requires_grad=trainable.feedback,
    )
    direct_gain = gain_module(
        d, nfft, device=device, dtype=dtype, requires_grad=trainable.direct
    )
    delays = _frozen_delays(
        np.asarray(build.delays, dtype=np.float64).ravel(), fs, nfft, device, dtype
    )

    # A prebuilt loop_filter takes precedence over the build's SOS bank.
    n = int(np.asarray(build.delays).ravel().size)
    if loop_filter is not None:
        loop_filter_module = _validated_loop_filter(loop_filter, n, nfft)
        if build.filters is not None:
            warnings.warn(
                "loop_filter provided; build.filters is ignored", stacklevel=2
            )
    elif build.filters is not None:
        loop_filter_module = sos_filter_module(
            np.asarray(build.filters, dtype=np.float64),
            nfft,
            device=device,
            dtype=dtype,
        )
    else:
        loop_filter_module = None
    output_filter = (
        sos_filter_module(
            np.asarray(build.post_eq, dtype=np.float64),
            nfft,
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        if build.post_eq is not None
        else None
    )

    core = assemble_fdn_core(
        input_gain=input_gain,
        feedback=feedback,
        delays=delays,
        output_gain=output_gain,
        direct=direct_gain,
        loop_filter=loop_filter_module,
        output_filter=output_filter,
    )
    return wrap_fdn_shell(core, nfft=nfft, dtype=dtype, output=output)


def build_set_decay(
    build: FDNBuild,
    rt: float | tuple[float, float],
    *,
    rt_crossover: float | None = None,
) -> FDNBuild:
    """Return a copy of ``build`` with homogeneous decay matching ``rt``.

    Sets per-delay first-order absorption (:func:`pyFDN.first_order_absorption`)
    for ``rt`` (a single value, or ``(rt_dc, rt_nyquist)``). Decay does not change
    colouration, so this is the natural way to add a tail to a colorless build.
    """
    from pyFDN.auxiliary.acoustics import first_order_absorption

    rt_dc, rt_ny = _rt_pair(rt)
    filters = first_order_absorption(
        rt_dc, rt_ny, np.asarray(build.delays), float(build.fs), rt_crossover
    )
    return dataclasses.replace(build, filters=filters)


def _validated_loop_filter(loop_filter: Any, n: int, nfft: int) -> Any:
    """Return ``loop_filter`` after checking its channel count and ``nfft`` match
    the FDN (``N`` delay lines, FFT size ``nfft``)."""
    param = getattr(loop_filter, "param", None)
    n_ch = int(param.shape[-1]) if param is not None else None
    if n_ch != n:
        raise ValueError(
            f"loop_filter operates on {n_ch} channels but the FDN has {n} "
            "delay lines"
        )
    lf_nfft = getattr(loop_filter, "nfft", None)
    if lf_nfft is None or int(lf_nfft) != int(nfft):
        raise ValueError(
            f"loop_filter nfft={lf_nfft} must match the build nfft={nfft}"
        )
    return loop_filter


def _rt_pair(rt: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(rt, tuple | list):
        return float(rt[0]), float(rt[1])
    return float(rt), float(rt)


def _resolve_direct(direct: float | np.ndarray, n_out: int, n_in: int) -> np.ndarray:
    arr = np.asarray(direct, dtype=float)
    if arr.ndim == 0:
        return np.full((n_out, n_in), float(arr))
    return arr.reshape(n_out, n_in)


def _random_so_n(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random orthogonal matrix projected into SO(N) (det = +1).

    Landing in SO(N) means the orthogonal parametrization's preimage
    (``logm``) round-trips without the det<0 projection warning.
    """
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q * signs
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1.0
    return q


def _frozen_delays(
    delay_samples: np.ndarray, fs: float, nfft: int, device: Any, dtype: Any
) -> Any:
    """Frozen integer parallelDelay from delay lengths in samples."""
    from pyFDN.auxiliary.flamo import delay_module

    return delay_module(
        np.asarray(delay_samples, dtype=np.float64) / float(fs),
        nfft,
        Fs=fs,
        device=device,
        dtype=dtype,
        isint=True,
        requires_grad=False,
    )
