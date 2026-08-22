"""Build a trainable flamo FDN model from a config.

:func:`build_fdn` turns a config (delays/N, decay, what is trainable) into a
trainable flamo ``Shell`` you can render, train, and extract.
:func:`trainable_from_build` does the same starting from an existing
:class:`~pyFDN.FDNBuild`.

Both are conveniences over assembling flamo modules yourself with
:func:`pyFDN.assemble_fdn_core`; neither knows anything about filter *design*.
A trainable filter is a module -- :class:`~pyFDN.DecayFilter`,
:class:`~pyFDN.OutputEQ` -- built from an :class:`~pyFDN.eq.EQDesign` that
carries its own target, and handed to whichever hook it belongs in.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from pyFDN.generate.fdn_matrix_gallery import FDNBuild

# Feedback-matrix parametrization: "orthogonal" keeps the matrix on SO(N) during
# training (the colorless choice); "random" trains it unconstrained.
MatrixParam = Literal["orthogonal", "random"]

# Default anti-time-aliasing decay for a LOSSLESS FDN (``rt=None``), whose poles
# lie exactly on the unit circle -- so the FFT-domain evaluation of (I - A D(z))^-1
# is near-singular without it. The value is the accuracy of the resulting impulse
# response in dB (see ``trainable_from_build``); 60 dB is about the ceiling in
# float32, where the reconstruction envelope amplifies round-off at the end of
# the buffer by the same factor. Use float64 to go higher.
LOSSLESS_ALIAS_DECAY_DB = 60.0


@dataclass(frozen=True)
class Trainable:
    """Which FDN parameter groups are trained. Delays are always fixed.

    The two filter fields are named for the hooks they govern, the same three
    names :func:`pyFDN.process_fdn` and :func:`pyFDN.assemble_fdn_core` use.
    They govern whatever module sits in that hook, including one you built and
    passed in yourself -- so a :class:`~pyFDN.DecayFilter` handed to
    :func:`trainable_from_build` is trained only if ``post_delay=True`` here.

    ``post_delay`` trains the in-loop filter, i.e. the decay itself. It is off
    by default because the decay is usually *designed*, from a measured RT.
    What it trains depends on what is in the hook: a
    :class:`~pyFDN.DecayFilter`'s parameter is the reverberation time itself,
    which keeps the loop contractive for every value it can take, while a plain
    SOS bank taken from ``build.post_delay`` exposes raw biquad coefficients
    that nothing keeps inside the unit circle -- a fit that wants more energy
    raises the loop gain past 1 and the network diverges.

    ``post_output`` trains the output filter, which sits *outside* the recursion
    and is therefore the only part of an FDN that can shape the response's
    spectral envelope without touching the decay: the gains ``b`` and ``c`` are
    single numbers per delay line and have no frequency dependence at all.
    """

    feedback: bool = True
    input_gain: bool = True
    output_gain: bool = True
    direct: bool = False
    post_delay: bool = False
    post_output: bool = False


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
    fs: float = 48000.0,
    nfft: int = 2**14,
    output: str = "time",
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
        Reverberation time in seconds, realized as a :class:`~pyFDN.DecayFilter`
        on a :class:`~pyFDN.FirstOrderShelf` -- so ``Trainable(post_delay=True)``
        trains the reverberation time itself. ``None`` builds a lossless FDN.
        For any other design, build the module yourself and pass it to
        :func:`trainable_from_build` as ``post_delay=``.
    matrix : {"orthogonal", "random"}
        Feedback-matrix parametrization.
    feedback : np.ndarray, optional
        Initial ``(N, N)`` feedback matrix; defaults to a random SO(N) matrix.
    input_gain, output_gain : np.ndarray, optional
        ``B`` (``(N, n_in)``) and ``C`` (``(n_out, N)``); default ones / sqrt(N).
    direct : float or np.ndarray
        Direct path ``D``; a scalar fills ``(n_out, n_in)``.
    trainable : Trainable, optional
        Trainable parameter groups (default :class:`~pyFDN.Trainable`).
    alias_decay_db : float or None
        Anti-time-aliasing decay, see :func:`trainable_from_build`. ``None``
        (default) picks it from ``rt``: :data:`LOSSLESS_ALIAS_DECAY_DB` when
        ``rt is None``, else 0. A lossless FDN has every pole exactly on the
        unit circle, where the FFT-domain evaluation breaks down entirely; a
        decaying FDN damps itself within ``nfft`` samples and needs no nudge.
        Pass ``0.0`` to opt out.
    fs, nfft, output, device, dtype : see :func:`trainable_from_build`.
    rng : np.random.Generator, int, or None
        Seed for the sampled delays / default feedback matrix.

    Returns
    -------
    flamo.processor.system.Shell
    """
    from pyFDN.generate.fdn_matrix_gallery import FDNBuild
    from pyFDN.generate.sample_delay_lengths import sample_delay_lengths

    trainable = trainable or Trainable()

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
        from pyFDN.auxiliary.flamo import DecayFilter
        from pyFDN.eq.designs import FirstOrderShelf

        post_delay = DecayFilter(
            FirstOrderShelf(_rt_pair(rt)),
            delays_arr,
            float(fs),
            nfft=nfft,
            alias_decay_db=float(alias_decay_db),
            device=device,
            dtype=dtype,
            requires_grad=trainable.post_delay,
        )

    build = FDNBuild(A=a, B=b, C=c, D=d, delays=delays_arr, fs=float(fs))
    return trainable_from_build(
        build,
        trainable=trainable,
        matrix=matrix,
        post_delay=post_delay,
        nfft=nfft,
        output=output,
        alias_decay_db=alias_decay_db,
        device=device,
        dtype=dtype,
    )


def trainable_from_build(
    build: FDNBuild,
    *,
    trainable: Trainable | None = None,
    matrix: MatrixParam = "orthogonal",
    post_delay: Any = None,
    post_matrix: Any = None,
    post_output: Any = None,
    nfft: int = 2**14,
    output: str = "time",
    alias_decay_db: float = 0.0,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Build a trainable flamo ``Shell`` initialized from an ``FDNBuild``.

    The gains and the feedback matrix come from the build. The three filter
    hooks are the build's own baked SOS banks unless you hand in a module for
    that position -- which is how a *designed*, trainable filter gets in, since
    a baked build no longer remembers the reverberation time or the EQ curve it
    was designed from::

        model = pyFDN.trainable_from_build(
            build,
            trainable=pyFDN.Trainable(post_delay=True, post_output=True),
            post_delay=pyFDN.DecayFilter(pyFDN.FirstOrderShelf((1.0, 1.0)),
                                         build.delays, build.fs, nfft=nfft),
            post_output=pyFDN.OutputEQ(pyFDN.FirstOrderShelf(0.0),
                                       build.C.shape[0], build.fs, nfft=nfft),
        )

    Parameters
    ----------
    build : FDNBuild
        Initial FDN (``A``/``B``/``C``/``D``/``delays``/``fs`` + optional
        ``post_delay``/``post_output`` SOS banks).
    trainable : Trainable, optional
        Trainable parameter groups (default :class:`~pyFDN.Trainable`). It
        governs the modules you pass in as well as the ones built from the
        build, so a :class:`~pyFDN.DecayFilter` is frozen unless
        ``Trainable(post_delay=True)``.
    matrix : {"orthogonal", "random"}
        Feedback-matrix parametrization.
    post_delay : FLAMO module, optional
        In-loop filter, replacing ``build.post_delay``. A
        :class:`~pyFDN.DecayFilter` here makes the trained parameter the
        reverberation time itself, which keeps the loop contractive for every
        value it can take; ``build.post_delay`` trained directly is raw biquad
        coefficients, which nothing keeps inside the unit circle.
    post_matrix : FLAMO module, optional
        Filter on the feedback path, replacing ``build.post_matrix``. Always
        frozen: :class:`~pyFDN.Trainable` has no field for it, since nothing in
        pyFDN trains this hook yet.
    post_output : FLAMO module, optional
        Output EQ, replacing ``build.post_output``; typically an
        :class:`~pyFDN.OutputEQ`.
    nfft : int
        FFT size.
    output : str
        ``"time"`` (the impulse response -- the default, and what training
        needs) or ``"magnitude"`` (``|H|`` at the DFT bins, for inspection).
    alias_decay_db : float
        **The accuracy of the rendered impulse response, in dB.** Applies a
        :math:`\\gamma^n` envelope to every module, i.e. evaluates the system on
        a circle of radius :math:`\\gamma < 1` instead of the unit circle; the
        ``"time"`` output layer removes the envelope again, so the response is
        the true one and what remains is the time-aliased wrap-around,
        suppressed by exactly ``alias_decay_db``. Measured against
        :func:`pyFDN.build_to_impz` (which cannot alias), the residual error of
        a lossless FDN is -30 dB at 30, -60 dB at 60, -90 dB at 90 in float64.

        In float32 the reconstruction envelope amplifies round-off by the same
        factor, so ~60 dB is the practical ceiling (the last eighth of the
        buffer degrades to about -38 dB, and 90 dB is worse than useless); use
        ``dtype=torch.float64`` beyond that.

        Leave at 0 for a decaying FDN, which damps itself within ``nfft``
        samples. A **lossless** FDN needs it: with its poles exactly on the unit
        circle the FFT-domain evaluation is near-singular and the response comes
        out wrong, not merely aliased. It does not affect the extracted build:
        ``alias_decay_db`` enters the frequency-domain evaluation, not the
        parameter ``map``, so :func:`pyFDN.extract_build` still returns the
        undamped ``A``/``B``/``C``.

        A module you pass into a hook must have been built with the same value:
        it is a change of evaluation radius for the whole system, not a
        per-module gain.
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

    # The alias envelope must be identical on every module -- it is a change of
    # evaluation radius for the whole system, not a per-module gain.
    alias = float(alias_decay_db)

    input_gain = gain_module(
        b,
        nfft,
        device=device,
        dtype=dtype,
        alias_decay_db=alias,
        requires_grad=trainable.input_gain,
    )
    output_gain = gain_module(
        c,
        nfft,
        device=device,
        dtype=dtype,
        alias_decay_db=alias,
        requires_grad=trainable.output_gain,
    )
    feedback = matrix_module(
        build.A,
        nfft,
        matrix_type=matrix,
        device=device,
        dtype=dtype,
        alias_decay_db=alias,
        requires_grad=trainable.feedback,
    )
    # Direct path is ALWAYS wired (zero by default) so the same model serves any
    # objective; the core is therefore a Parallel.
    direct_gain = gain_module(
        d,
        nfft,
        device=device,
        dtype=dtype,
        alias_decay_db=alias,
        requires_grad=trainable.direct,
    )
    delays = _frozen_delays(
        np.asarray(build.delays, dtype=np.float64).ravel(),
        fs,
        nfft,
        device,
        dtype,
        alias_decay_db=alias,
    )

    def _hook(module: Any, baked: np.ndarray | None, requires_grad: bool) -> Any:
        """A hook's module: the one given, else the build's baked SOS, else none.

        ``Trainable`` is the single source of truth for what moves, so a module
        the caller built is re-flagged here rather than keeping whatever
        ``requires_grad`` it was constructed with. A composite module -- a
        nested core in the ``post_delay`` hook, say -- has no single ``param``
        to flag, so it is wired in as it was built.
        """
        if module is not None:
            leaf_param = getattr(module, "param", None)
            if leaf_param is not None:
                leaf_param.requires_grad_(requires_grad)
            return module
        if baked is None:
            return None
        return sos_filter_module(
            np.asarray(baked, dtype=np.float64),
            nfft,
            device=device,
            dtype=dtype,
            alias_decay_db=alias,
            requires_grad=requires_grad,
        )

    core = assemble_fdn_core(
        input_gain=input_gain,
        feedback=feedback,
        delays=delays,
        output_gain=output_gain,
        direct=direct_gain,
        post_delay=_hook(post_delay, build.post_delay, trainable.post_delay),
        post_matrix=_hook(post_matrix, build.post_matrix, False),
        post_output=_hook(post_output, build.post_output, trainable.post_output),
    )
    return wrap_fdn_shell(core, nfft=nfft, dtype=dtype, output=output)


def build_set_decay(
    build: FDNBuild,
    rt: float | tuple[float, float],
    *,
    rt_crossover: float | None = None,
) -> FDNBuild:
    """Return a copy of ``build`` with homogeneous decay matching ``rt``.

    Sets the ``post_delay`` hook to per-delay first-order absorption
    (:func:`pyFDN.first_order_absorption`) for ``rt`` (a single value, or
    ``(rt_dc, rt_nyquist)``). Decay does not change colouration, so this is the
    natural way to add a tail to a colorless build.
    """
    from pyFDN.eq.first_order import first_order_absorption

    rt_dc, rt_ny = _rt_pair(rt)
    filters = first_order_absorption(
        rt_dc, rt_ny, np.asarray(build.delays), float(build.fs), rt_crossover
    )
    return dataclasses.replace(build, post_delay=filters)


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
    delay_samples: np.ndarray,
    fs: float,
    nfft: int,
    device: Any,
    dtype: Any,
    alias_decay_db: float = 0.0,
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
        alias_decay_db=alias_decay_db,
        requires_grad=False,
    )
