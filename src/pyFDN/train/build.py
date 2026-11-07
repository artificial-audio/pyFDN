"""Build a trainable flamo FDN model from a config or from a DSS system.

:func:`trainable_fdn` turns a config (delays/N, decay, which gains train) into a
trainable flamo ``Shell`` you can render, train, and extract.
:func:`trainable_from_dss` does the same starting from an existing
:class:`~pyFDN.DSS`.

Both are conveniences over assembling flamo modules yourself with
:func:`pyFDN.assemble_fdn_core`. A DSS system holds baked values only, so a
trainable filter is a module -- :class:`~pyFDN.AttenuationFilter` or
:class:`~pyFDN.OutputEQ` -- initialized with the values to start from and
handed to whichever hook it belongs in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

# Feedback-matrix parametrization: "orthogonal" keeps the matrix on SO(N) during
# training (the colorless choice); "random" trains it unconstrained.
MatrixParam = Literal["orthogonal", "random"]

# Default anti-time-aliasing decay for a LOSSLESS FDN (``rt=None``), whose poles
# lie exactly on the unit circle -- so the FFT-domain evaluation of (I - A D(z))^-1
# is near-singular without it. The value is the accuracy of the resulting impulse
# response in dB (see ``trainable_from_dss``); 60 dB is about the ceiling in
# float32, where the reconstruction envelope amplifies round-off at the end of
# the buffer by the same factor. Use float64 to go higher.
LOSSLESS_ALIAS_DECAY_DB = 60.0


@dataclass(frozen=True)
class Trainable:
    """Which of the FDN's gain groups are trained. Delays are always fixed.

    These are plain arrays, so the flag is named here; a filter hook is a
    module (:class:`~pyFDN.AttenuationFilter`, :class:`~pyFDN.OutputEQ`) and
    carries its own ``requires_grad``.

    A baked SOS bank passed as a filter hook is always frozen:
    nothing keeps raw biquad coefficients inside the unit circle, so a fit can
    push the loop gain past 1. Training one is a module you build on purpose
    (:func:`pyFDN.sos_filter_module`), not a flag.
    """

    feedback: bool = True
    input_gain: bool = True
    output_gain: bool = True
    direct: bool = False


def trainable_fdn(
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
    train_rt: bool = False,
    fs: float = 48000.0,
    nfft: int = 2**16,
    alias_decay_db: float | None = None,
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
        Reverberation time in seconds, realized as an
        :class:`~pyFDN.AttenuationFilter` with ``design="first_order_shelf"``.
        ``None`` builds a lossless FDN.
        For any other design, build the module yourself and pass it to
        :func:`trainable_from_dss` as ``post_delay=``.
    matrix : {"orthogonal", "random"}
        Feedback-matrix parametrization.
    feedback : np.ndarray, optional
        Initial ``(N, N)`` feedback matrix; defaults to a random SO(N) matrix.
    input_gain, output_gain : np.ndarray, optional
        ``B`` (``(N, n_in)``) and ``C`` (``(n_out, N)``); default ones / sqrt(N).
    direct : float or np.ndarray
        Direct path ``D``; a scalar fills ``(n_out, n_in)``.
    trainable : Trainable, optional
        Which gain groups are trained (default :class:`~pyFDN.Trainable`).
    train_rt : bool
        Whether ``rt`` is a *parameter* rather than a design. Off by default,
        since the decay is usually designed from a measured reverberation time.
        What trains is the reverberation time itself, which keeps the loop
        contractive for every value it can take -- unlike raw filter
        coefficients, which nothing holds inside the unit circle.
    alias_decay_db : float or None
        Anti-time-aliasing decay, see :func:`trainable_from_dss`. ``None``
        (default) picks :data:`LOSSLESS_ALIAS_DECAY_DB` when ``rt is None``,
        else 0.
    fs, nfft, device, dtype : see :func:`trainable_from_dss`.
    rng : np.random.Generator, int, or None
        Seed for the sampled delays / default feedback matrix.

    Returns
    -------
    flamo.processor.system.Shell
    """
    from pyFDN.auxiliary.utils import as_generator
    from pyFDN.generate.sample_delay_lengths import sample_delay_lengths

    trainable = trainable or Trainable()

    local_rng = as_generator(rng)

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

    # Default IO is "normalized" (ones / sqrt(N)): puts the initial |H| near unity
    # so a colorless objective starts well-conditioned. Override with input_gain/
    # output_gain for other layouts.
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

    if alias_decay_db is None:
        alias_decay_db = LOSSLESS_ALIAS_DECAY_DB if rt is None else 0.0

    post_delay = None
    if rt is not None:
        from pyFDN.train.filters import AttenuationFilter

        rt_value, rt_nyquist = _rt_pair(rt)
        post_delay = AttenuationFilter(
            rt_value,
            delays_arr,
            float(fs),
            rt_nyquist=rt_nyquist,
            design="first_order_shelf",
            nfft=nfft,
            alias_decay_db=float(alias_decay_db),
            device=device,
            dtype=dtype,
            requires_grad=train_rt,
        )

    return trainable_from_dss(
        delays_arr,
        a,
        b,
        c,
        d,
        fs,
        trainable=trainable,
        matrix=matrix,
        post_delay=post_delay,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        device=device,
        dtype=dtype,
    )


def trainable_from_dss(
    delays: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    fs: float,
    *,
    trainable: Trainable | None = None,
    matrix: MatrixParam = "orthogonal",
    post_delay: Any = None,
    post_matrix: Any = None,
    post_output: Any = None,
    nfft: int = 2**16,
    alias_decay_db: float = 0.0,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Build a trainable flamo ``Shell`` initialized from a :class:`~pyFDN.DSS`.

    The gains and the feedback matrix come from the system. Each filter hook is
    either a baked SOS bank, wired in frozen, or a module -- which is how a
    *designed*, trainable filter gets in, since baked values do not remember
    the reverberation time or the EQ curve they were designed from::

        model = pyFDN.trainable_from_dss(
            *dss, fs,
            post_delay=pyFDN.AttenuationFilter(
                1.0, dss.delays, fs, rt_nyquist=1.0,
                design="first_order_shelf", nfft=nfft),
            post_output=pyFDN.OutputEQ(
                0.0, dss.C.shape[0], fs,
                design="first_order_shelf", nfft=nfft),
        )

    Baked filters from :func:`pyFDN.flamo_to_dss` go straight back in:
    ``trainable_from_dss(*dss, fs, **filters)``.

    Both modules default to ``requires_grad=True``.

    Parameters
    ----------
    delays, A, B, C, D : array-like
        Initial delay state-space system; a :class:`~pyFDN.DSS` unpacks
        straight in, as for every ``dss_*`` function.
    fs : float
        Sampling rate in Hz.
    trainable : Trainable, optional
        Which gain groups are trained (default :class:`~pyFDN.Trainable`). It
        says nothing about the filter hooks: each module below carries its own
        ``requires_grad``, and is wired in exactly as it was built.
    matrix : {"orthogonal", "random"}
        Feedback-matrix parametrization.
    post_delay : SOS bank or FLAMO module, optional
        In-loop filter. A baked ``(sections, 6, N)`` SOS bank is wired in
        frozen. A
        :class:`~pyFDN.AttenuationFilter` here makes the trained parameter the
        reverberation time itself, which keeps the loop contractive for every
        value it can take.
    post_matrix : SOS bank or FLAMO module, optional
        Filter on the feedback path.
    post_output : SOS bank or FLAMO module, optional
        Output EQ; typically an
        :class:`~pyFDN.OutputEQ`. It sits *outside* the recursion, which makes
        it the only part of an FDN that can shape the response's spectral
        envelope without touching the decay -- ``b`` and ``c`` are single
        numbers per delay line, with no frequency dependence at all.
    nfft : int
        FFT size.
    alias_decay_db : float
        **The accuracy of the rendered impulse response, in dB.** Applies a
        :math:`\\gamma^n` envelope to every module (evaluating the system on a
        circle of radius :math:`\\gamma < 1`); the shell's output layer
        removes it again, so the response is the true one and only the
        time-aliased wrap-around remains, suppressed by exactly
        ``alias_decay_db``. In float32 the reconstruction amplifies round-off by
        the same factor, so ~60 dB is the practical ceiling; use
        ``dtype=torch.float64`` beyond that.

        Leave at 0 for a decaying FDN, which damps itself within ``nfft``
        samples. A **lossless** FDN needs it: with its poles exactly on the unit
        circle the FFT-domain evaluation is near-singular and the response comes
        out wrong, not merely aliased. It does not affect the extracted system
        (it enters the frequency-domain evaluation, not the parameter ``map``,
        so :func:`pyFDN.flamo_to_dss` still returns the undamped ``A``/``B``/
        ``C``). A module you pass into a hook must have been built with the same
        value: it is a change of evaluation radius for the whole system, not a
        per-module gain.
    device, dtype : optional
        Torch device / dtype (default cpu-or-cuda / float32).

    Returns
    -------
    flamo.processor.system.Shell
    """
    from pyFDN.auxiliary.flamo import matrix_module, wrap_fdn_shell
    from pyFDN.translate.dss_to_flamo import fdn_core

    trainable = trainable or Trainable()
    # The alias envelope must be identical on every module -- it is a change of
    # evaluation radius for the whole system, not a per-module gain.
    alias = float(alias_decay_db)
    feedback = matrix_module(
        np.asarray(A, dtype=float),
        nfft,
        matrix_type=matrix,
        device=device,
        dtype=dtype,
        alias_decay_db=alias,
        requires_grad=trainable.feedback,
    )
    # A module given for a hook is wired in exactly as it was built -- what it
    # trains is its own business. A baked SOS bank is wired in frozen: see
    # Trainable for why raw coefficients are not trained by default.
    # The direct path is always wired (zero by default), so the core is a
    # Parallel and the same model serves any objective.
    core = fdn_core(
        np.asarray(delays),
        np.asarray(A, dtype=float),
        np.asarray(B, dtype=float),
        np.asarray(C, dtype=float),
        np.asarray(D, dtype=float),
        float(fs),
        nfft,
        device=device,
        dtype=dtype,
        alias_decay_db=alias,
        feedback=feedback,
        trainable=trainable,
        post_delay=post_delay,
        post_matrix=post_matrix,
        post_output=post_output,
    )
    return wrap_fdn_shell(core, nfft=nfft, dtype=dtype, device=device)


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
    from pyFDN.generate.orthogonal import random_orthogonal

    q = random_orthogonal(n, rng)
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1.0
    return q
