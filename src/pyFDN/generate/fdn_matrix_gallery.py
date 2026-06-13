"""Gallery of feedback matrices, specialized systems, and complete FDN builds.

Translation of fdnMatrixGallery.m from fdnToolbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, NoReturn, overload

import numpy as np
from numpy.typing import ArrayLike

from .householder_matrix import householder_matrix
from .random_orthogonal import random_orthogonal


class FDNSystem(NamedTuple):
    """Full FDN system matrices returned by :func:`fdn_system_gallery`."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray


@dataclass(frozen=True)
class FDNBuild:
    """Complete FDN parameters returned by :func:`fdn_build_gallery`.

    ``filters`` is either ``None`` (lossless) or a per-delay first-order
    absorption SOS bank with shape ``(num_sections, 6, N)`` suitable for
    ``dss_to_flamo(..., sos_filter=...)``. ``post_eq`` is an optional per-output
    SOS bank with shape ``(num_sections, 6, num_outputs)`` suitable for the
    ``output_filter`` argument of :func:`pyFDN.dss_to_flamo`.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    delays: np.ndarray
    fs: float
    filters: np.ndarray | None = None
    post_eq: np.ndarray | None = None


def _circulant(v: np.ndarray, direction: int = 1) -> np.ndarray:
    """Build a circulant matrix from first-row vector v.

    ``direction=1``: each row shifts right; ``direction=-1``: shifts left.
    """
    v = np.asarray(v, dtype=float).ravel()
    N = len(v)
    rows = [np.roll(v, i * direction) for i in range(N)]
    return np.array(rows)


# Pure feedback-matrix types (return np.ndarray).
_MATRIX_TYPES = [
    "orthogonal",
    "Hadamard",
    "circulant",
    "Householder",
    "parallel",
    "permutation",
    "diagonallySimilarToOrthogonal",
    "tinyRotation",
    "Anderson",
]

# Full-system types (return FDNSystem).
_SYSTEM_TYPES = [
    "series",
    "nestedAllpass",
    "polettiAllpass",
    "homogeneousAllpass",
    "SchroederReverberator",
    "allpassInFDN",
]

# Filter (FIR paraunitary) matrix types (return (N, N, L) np.ndarray).
_FILTER_MATRIX_TYPES = [
    "RandomDense",
    "Velvet",
    "FromElementals",
]


def _build_rng(
    rng: np.random.Generator | int | None, default_seed: int
) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(default_seed if rng is None else rng)


def _random_orthogonal(N: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((N, N)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def _build_io_matrices(
    N: int,
    num_inputs: int,
    num_outputs: int,
    io_type: str,
    input_scale: float,
    output_scale: float,
    direct_gain: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if min(num_inputs, num_outputs) < 1:
        raise ValueError("num_inputs and num_outputs must be positive")

    if io_type == "ones":
        B = np.ones((N, num_inputs))
        C = np.ones((num_outputs, N))
    elif io_type == "normalized":
        B = np.ones((N, num_inputs)) / np.sqrt(N)
        C = np.ones((num_outputs, N)) / np.sqrt(N)
    elif io_type == "identity":
        B = np.eye(N, num_inputs)
        C = np.eye(num_outputs, N)
    elif io_type == "random":
        B = rng.standard_normal((N, num_inputs))
        C = rng.standard_normal((num_outputs, N))
    else:
        raise ValueError(
            "io_type must be one of 'ones', 'normalized', 'identity', or 'random'"
        )

    B = input_scale * B
    C = output_scale * C
    if direct_gain is None:
        D = rng.standard_normal((num_outputs, num_inputs))
    else:
        D = np.full((num_outputs, num_inputs), direct_gain, dtype=float)
    return B, C, D


def _build_post_eq(
    num_outputs: int,
    fs: float,
    db_dc: ArrayLike | None,
    db_nyquist: ArrayLike | None,
    crossover_frequency: float | None,
) -> np.ndarray | None:
    """Per-output first-order shelving post EQ from dB gains, or ``None``."""
    if db_dc is None and db_nyquist is None:
        if crossover_frequency is not None:
            raise ValueError(
                "post_eq_db_dc or post_eq_db_nyquist must be set to configure post EQ"
            )
        return None
    if db_dc is None:
        db_dc = db_nyquist
    if db_nyquist is None:
        db_nyquist = db_dc
    assert db_dc is not None and db_nyquist is not None

    def _per_output(value: ArrayLike, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size == 1:
            arr = np.full(num_outputs, arr.item())
        if arr.size != num_outputs:
            raise ValueError(f"{name} must be scalar or length num_outputs")
        return arr

    from ..auxiliary.acoustics import first_order_shelving_eq

    return first_order_shelving_eq(
        _per_output(db_dc, "post_eq_db_dc"),
        _per_output(db_nyquist, "post_eq_db_nyquist"),
        fs,
        crossover_frequency,
    )


def fdn_build_gallery(
    N: int | None = None,
    *,
    fs: float = 48_000.0,
    delays: np.ndarray | None = None,
    delay_range: tuple[int, int] = (400, 1200),
    sort_delays: bool = False,
    num_inputs: int = 1,
    num_outputs: int = 1,
    io_type: str = "normalized",
    input_scale: float = 1.0,
    output_scale: float = 1.0,
    direct_gain: float | None = 0.0,
    rt: float | None = 2.0,
    rt_nyquist: float | None = None,
    rt_crossover: float | None = None,
    post_eq_db_dc: ArrayLike | None = None,
    post_eq_db_nyquist: ArrayLike | None = None,
    post_eq_crossover: float | None = None,
    rng: np.random.Generator | int | None = None,
) -> FDNBuild:
    """Build a complete FDN from a delay range, a reverberation time, and an EQ.

    The feedback matrix ``A`` is a random orthogonal matrix. In-loop decay is
    realised as per-delay first-order shelving absorption filters matching
    ``rt`` at DC and ``rt_nyquist`` at Nyquist; pass ``rt=None`` for a
    lossless FDN (``filters=None``). An
    optional per-output first-order shelving post EQ is specified directly in
    decibels at DC and Nyquist.

    Delays and I/O matrices use a local :class:`numpy.random.Generator`; passing
    an integer or generator makes the build reproducible without mutating
    NumPy's global random state.

    Args:
        N: Number of delay lines. Inferred from ``delays`` when given.
        fs: Sample rate in Hz.
        delays: Optional explicit delay lengths in samples.
        delay_range: Half-open random delay range when ``delays`` is omitted.
        sort_delays: Sort randomly generated or supplied delays.
        num_inputs: Number of input channels.
        num_outputs: Number of output channels.
        io_type: I/O matrix style: ``ones``, ``normalized``, ``identity``, or
            ``random``.
        input_scale: Scalar applied to ``B``.
        output_scale: Scalar applied to ``C``.
        direct_gain: Constant direct gain, or ``None`` for random ``D``.
        rt: Reverberation time in seconds at DC, or ``None`` for a lossless
            FDN with no in-loop absorption filters.
        rt_nyquist: Reverberation time in seconds at Nyquist. Defaults to
            ``rt`` (frequency-flat decay).
        rt_crossover: Shelf crossover for the absorption filters in Hz.
        post_eq_db_dc: Post-EQ gain in dB at DC, scalar or length ``num_outputs``.
            Setting either post-EQ argument enables a per-output output filter.
        post_eq_db_nyquist: Post-EQ gain in dB at Nyquist, scalar or length
            ``num_outputs``. Defaults to ``post_eq_db_dc`` (flat gain).
        post_eq_crossover: Shelf crossover for the post EQ in Hz.
        rng: Local NumPy generator or integer seed.

    Returns:
        A complete :class:`FDNBuild`.
    """
    if fs <= 0:
        raise ValueError("fs must be positive")
    if rt is not None and rt <= 0:
        raise ValueError("rt must be positive")
    if rt_nyquist is not None and rt_nyquist <= 0:
        raise ValueError("rt_nyquist must be positive")

    if delays is not None:
        delays_array = np.asarray(delays, dtype=int).ravel()
        if N is None:
            N = delays_array.size
        elif delays_array.size != N:
            raise ValueError("delays must contain exactly N values")
        local_rng = _build_rng(rng, 0)
    else:
        if N is None:
            raise ValueError("N must be provided when delays is omitted")
        low, high = delay_range
        if low < 1 or high <= low:
            raise ValueError("delay_range must satisfy 1 <= low < high")
        local_rng = _build_rng(rng, 0)
        delays_array = local_rng.integers(low, high, size=N)

    if N < 1:
        raise ValueError("N must be positive")
    if sort_delays:
        delays_array = np.sort(delays_array)
    if np.any(delays_array < 1):
        raise ValueError("all delays must be positive")

    A = _random_orthogonal(N, local_rng)
    B, C, D = _build_io_matrices(
        N,
        num_inputs,
        num_outputs,
        io_type,
        input_scale,
        output_scale,
        direct_gain,
        local_rng,
    )

    filters: np.ndarray | None = None
    if rt is not None:
        from ..auxiliary.acoustics import first_order_absorption

        rt_ny = rt if rt_nyquist is None else rt_nyquist
        filters = first_order_absorption(
            rt, rt_ny, delays_array, float(fs), rt_crossover
        )

    post_eq = _build_post_eq(
        num_outputs, float(fs), post_eq_db_dc, post_eq_db_nyquist, post_eq_crossover
    )
    return FDNBuild(A, B, C, D, delays_array, float(fs), filters, post_eq)


@overload
def filter_matrix_gallery(
    N: int,
    matrix_type: str,
    *,
    num_stages: int = ...,
    sparsity: float = ...,
    stage_matrix_type: str = ...,
) -> np.ndarray: ...


@overload
def filter_matrix_gallery(
    N: int | None = ...,
    matrix_type: None = ...,
    *,
    num_stages: int = ...,
    sparsity: float = ...,
    stage_matrix_type: str = ...,
) -> list[str]: ...


@overload
def filter_matrix_gallery(
    N: None = ...,
    matrix_type: str = ...,
    *,
    num_stages: int = ...,
    sparsity: float = ...,
    stage_matrix_type: str = ...,
) -> NoReturn: ...


def filter_matrix_gallery(
    N: int | None = None,
    matrix_type: str | None = None,
    *,
    num_stages: int = 3,
    sparsity: float = 3.0,
    stage_matrix_type: str = "Hadamard",
) -> np.ndarray | list[str]:
    """Return an FIR (filter) feedback matrix of the requested type, or list all type names.

    All types are paraunitary (lossless): ``A^T(z^{-1}) A(z) = I``. Used as
    scattering feedback matrices in an FDN (Schlecht & Habets 2020).

    Args:
        N: Matrix size.  Ignored when ``matrix_type`` is ``None``.
        matrix_type: One of ``"RandomDense"`` (dense cascaded paraunitary
            matrix), ``"Velvet"`` (sparse velvet-noise feedback matrix), or
            ``"FromElementals"`` (cascade of degree-one lossless factors,
            polynomial degree ``N * num_stages``).  Pass ``None`` (or call
            with no arguments) to get the list of all type names.
        num_stages: Number of cascade stages (or degree factor for
            ``"FromElementals"``).
        sparsity: Sparsity of the ``"Velvet"`` type (ignored otherwise).
        stage_matrix_type: Stage matrix for ``"RandomDense"`` and ``"Velvet"``:
            ``"Hadamard"`` or ``"random"`` (random orthogonal; avoids the
            structural double poles at z = ±1 of Hadamard stages).

    Returns:
        Feedback matrix of shape ``(N, N, L)`` in z^{-1} convention, or a list
        of type-name strings.

    Example::

        filter_matrix_gallery()              # → list of type strings
        filter_matrix_gallery(4, "Velvet", num_stages=3, sparsity=3)
    """
    if matrix_type is None:
        return list(_FILTER_MATRIX_TYPES)

    if N is None:
        raise ValueError("N must be provided when matrix_type is specified")

    if matrix_type == "RandomDense":
        from .construct_cascaded_paraunitary_matrix import (
            construct_cascaded_paraunitary_matrix,
        )

        return construct_cascaded_paraunitary_matrix(
            N, num_stages, matrix_type=stage_matrix_type
        )[0]

    if matrix_type == "Velvet":
        from .construct_cascaded_paraunitary_matrix import (
            construct_cascaded_paraunitary_matrix,
        )

        return construct_cascaded_paraunitary_matrix(
            N, num_stages, sparsity=sparsity, matrix_type=stage_matrix_type
        )[0]

    if matrix_type == "FromElementals":
        from .construct_paraunitary_from_elementals import (
            construct_paraunitary_from_elementals,
        )

        return construct_paraunitary_from_elementals(N, N * num_stages)[0]

    raise ValueError(
        f"Unknown matrix_type {matrix_type!r}. Supported: {_FILTER_MATRIX_TYPES}"
    )


def fdn_matrix_gallery(
    N: int | None = None,
    matrix_type: str | None = None,
) -> np.ndarray | list[str]:
    """Return a feedback matrix of the requested type, or list all type names.

    Args:
        N: Matrix size.  Ignored when ``matrix_type`` is ``None``.
        matrix_type: One of the supported type strings.  Pass ``None`` (or call
                     with no arguments) to get the list of all type names.

    Returns:
        Feedback matrix of shape ``(N, N)``, or a list of type-name strings.

    Example::

        fdn_matrix_gallery()             # → list of type strings
        fdn_matrix_gallery(4, "orthogonal")
        fdn_matrix_gallery(8, "Hadamard")
    """
    if matrix_type is None:
        return list(_MATRIX_TYPES)

    if N is None:
        raise ValueError("N must be provided when matrix_type is specified")

    if matrix_type == "orthogonal":
        return random_orthogonal(N)

    if matrix_type == "Hadamard":
        from scipy.linalg import hadamard

        return hadamard(N) / np.sqrt(N)

    if matrix_type == "circulant":
        r_fft = np.fft.fft(np.random.randn(N))
        r_fft /= np.abs(r_fft)
        r = np.real(np.fft.ifft(r_fft))
        direction = np.random.choice([-1, 1])
        return _circulant(r, direction)

    if matrix_type == "Householder":
        return householder_matrix(np.random.rand(N))

    if matrix_type == "parallel":
        return np.eye(N)

    if matrix_type == "permutation":
        P = np.eye(N)
        return P[np.random.permutation(N)]

    if matrix_type == "diagonallySimilarToOrthogonal":
        D = np.diag(np.random.randn(N))
        return np.linalg.solve(D, random_orthogonal(N)) @ D

    if matrix_type == "tinyRotation":
        import torch

        from ..auxiliary.tiny_rotation_matrix import tiny_rotation_matrix

        return tiny_rotation_matrix(N, 0.01, dtype=torch.float64).numpy()

    if matrix_type == "Anderson":
        from .anderson_matrix import anderson_matrix

        return anderson_matrix(N)

    if matrix_type in _SYSTEM_TYPES:
        raise ValueError(
            f"'{matrix_type}' returns a full FDN system; use fdn_system_gallery() instead."
        )

    raise ValueError(f"Unknown matrix_type {matrix_type!r}. Supported: {_MATRIX_TYPES}")


def fdn_system_gallery(
    N: int | None = None,
    system_type: str | None = None,
) -> FDNSystem | list[str]:
    """Return a full FDN system (A, B, C, D) of the requested type, or list all type names.

    Args:
        N: System order.  Ignored when ``system_type`` is ``None``.
        system_type: One of the supported type strings.  Pass ``None`` (or call
                     with no arguments) to get the list of all type names.

    Returns:
        :class:`FDNSystem` named tuple ``(A, B, C, D)``, or a list of type-name strings.

    Example::

        fdn_system_gallery()                         # → list of type strings
        fdn_system_gallery(8, "allpassInFDN")
    """
    if system_type is None:
        return list(_SYSTEM_TYPES)

    if N is None:
        raise ValueError("N must be provided when system_type is specified")

    if system_type == "series":
        from ..auxiliary.allpass import series_allpass

        g = np.random.rand(N) * 0.6 + 0.2
        A, B, C, D = series_allpass(g)
        return FDNSystem(A, B, C, D)

    if system_type == "nestedAllpass":
        from ..auxiliary.allpass import nested_allpass

        g = np.random.rand(N) * 0.6 + 0.2
        A, B, C, D = nested_allpass(g)
        return FDNSystem(A, B, C, D)

    if system_type == "polettiAllpass":
        from ..auxiliary.allpass import poletti_allpass

        A, B, C, D = poletti_allpass(0.7, random_orthogonal(N))
        return FDNSystem(A, B, C, D)

    if system_type == "homogeneousAllpass":
        from .allpass_FDN.homogeneous_allpass_fdn import homogeneous_allpass_fdn
        from .allpass_FDN.rand_admissible_homogeneous_allpass import (
            rand_admissible_homogeneous_allpass,
        )

        G = np.diag(np.random.uniform(0.9, 0.99, N))
        X = rand_admissible_homogeneous_allpass(G, (0.8, 0.99))
        A, B, C, D, _ = homogeneous_allpass_fdn(G, X)
        return FDNSystem(A, B, C, D)

    if system_type == "SchroederReverberator":
        from .schroeder_reverberator import schroeder_reverberator

        N_c = N // 2
        N_a = N - N_c
        comb_gain = np.random.uniform(0.5, 0.9, N_c) * np.random.choice([-1, 1], N_c)
        allpass_gain = np.random.uniform(0.2, 0.7, N_a)
        b = np.random.randn(N_c, 1) / np.sqrt(N_c)
        c = np.random.randn(1, N_c) / np.sqrt(N_c)
        d = 0.0
        A, B, C, D = schroeder_reverberator(allpass_gain, comb_gain, b, c, d)
        return FDNSystem(A, B, C, D)

    if system_type == "allpassInFDN":
        from .allpass_in_fdn import allpass_in_fdn

        g = np.random.uniform(-0.8, 0.8, N // 2)
        A_inner = random_orthogonal(N // 2)
        b_inner = np.random.randn(N // 2, 1) / np.sqrt(N // 2)
        c_inner = np.random.randn(1, N // 2) / np.sqrt(N // 2)
        A, B, C, D = allpass_in_fdn(g, A_inner, b_inner, c_inner, 0.0)
        return FDNSystem(A, B, C, D)

    raise ValueError(f"Unknown system_type {system_type!r}. Supported: {_SYSTEM_TYPES}")
