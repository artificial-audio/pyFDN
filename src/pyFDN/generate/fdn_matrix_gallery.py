"""Gallery of feedback matrices and specialized FDN systems.

Translation of fdnMatrixGallery.m from fdnToolbox.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.utils import as_generator

from .orthogonal import householder_matrix, random_orthogonal


def _circulant(v: np.ndarray, direction: int = 1) -> np.ndarray:
    """Build a circulant matrix from first-row vector v.

    ``direction=1``: each row shifts right; ``direction=-1``: shifts left.
    """
    v = np.asarray(v, dtype=float).ravel()
    N = len(v)
    rows = [np.roll(v, i * direction) for i in range(N)]
    return np.array(rows)


FeedbackMatrixType = Literal[
    "orthogonal",
    "hadamard",
    "kronecker",
    "householder",
    "circulant",
    "permutation",
    "diagonally_similar_to_orthogonal",
    "tiny_rotation",
    "anderson",
    "parallel",
]
FEEDBACK_MATRIX_TYPES = get_args(FeedbackMatrixType)

SYSTEM_TYPES = (
    "series_allpass",
    "nested_allpass",
    "poletti_allpass",
    "homogeneous_allpass",
    "schroeder_reverberator",
    "allpass_in_fdn",
)
"""The ``system_type`` names :func:`fdn_system_gallery` accepts."""

FILTER_MATRIX_TYPES = ("random_dense", "velvet", "from_elementals")
"""The ``matrix_type`` names :func:`filter_matrix_gallery` accepts (FIR, paraunitary)."""


def filter_matrix_gallery(
    N: int,
    matrix_type: str,
    *,
    num_stages: int = 3,
    sparsity: float = 3.0,
    stage_matrix_type: str = "hadamard",
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Return an FIR (filter) feedback matrix of the requested type.

    All types are paraunitary (lossless): ``A^T(z^{-1}) A(z) = I``. Used as
    scattering feedback matrices in an FDN (Schlecht & Habets 2020).

    Args:
        N: Matrix size.
        matrix_type: One of :data:`FILTER_MATRIX_TYPES`: ``"random_dense"`` (dense cascaded paraunitary
            matrix), ``"velvet"`` (sparse velvet-noise feedback matrix), or
            ``"from_elementals"`` (cascade of degree-one lossless factors,
            polynomial degree ``N * num_stages``).
        num_stages: Number of cascade stages (or degree factor for
            ``"from_elementals"``).
        sparsity: Sparsity of the ``"velvet"`` type (ignored otherwise).
        stage_matrix_type: Stage matrix for ``"random_dense"`` and ``"velvet"``:
            ``"hadamard"`` or ``"random"`` (random orthogonal; avoids the
            structural double poles at z = ±1 of Hadamard stages).
        rng: Generator or integer seed, as for every random function.

    Returns:
        Feedback matrix of shape ``(N, N, L)`` in z^{-1} convention.

    Example::

        filter_matrix_gallery(4, "velvet", num_stages=3, sparsity=3)
    """
    if matrix_type == "random_dense":
        from .paraunitary import (
            cascaded_paraunitary_matrix,
        )

        return cascaded_paraunitary_matrix(
            N, num_stages, matrix_type=stage_matrix_type, rng=rng
        )[0]

    if matrix_type == "velvet":
        from .paraunitary import (
            cascaded_paraunitary_matrix,
        )

        return cascaded_paraunitary_matrix(
            N, num_stages, sparsity=sparsity, matrix_type=stage_matrix_type, rng=rng
        )[0]

    if matrix_type == "from_elementals":
        from .paraunitary import (
            paraunitary_from_elementals,
        )

        return paraunitary_from_elementals(N, N * num_stages, rng=rng)[0]

    raise ValueError(
        f"Unknown matrix_type {matrix_type!r}. Supported: {FILTER_MATRIX_TYPES}"
    )


def fdn_matrix_gallery(
    N: int,
    matrix_type: str,
    *,
    angles: ArrayLike | None = None,
    kernel_type: str | Sequence[str] = "rotation",
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Return a feedback matrix of the requested type.

    Args:
        N: Matrix size.
        matrix_type: One of :data:`FEEDBACK_MATRIX_TYPES`.
        angles: Kernel angles for the ``"kronecker"`` type, innermost first
            (see :func:`pyFDN.kronecker_matrix`).  Defaults to ``pi / 4`` on
            every kernel, the equal-mixing setting.  Ignored otherwise.
        kernel_type: Kernel family for the ``"kronecker"`` type,
            ``"rotation"`` or ``"reflection"``, either once for all kernels or
            one per kernel.  Ignored otherwise.
        rng: Generator or integer seed for the random types.

    Returns:
        Feedback matrix of shape ``(N, N)``.

    Example::

        fdn_matrix_gallery(4, "orthogonal")
        fdn_matrix_gallery(8, "hadamard")
        fdn_matrix_gallery(8, "kronecker", angles=[0.0, np.pi / 4, np.pi / 8])
    """
    rng = as_generator(rng)
    if matrix_type == "orthogonal":
        return random_orthogonal(N, rng=rng)

    if matrix_type == "hadamard":
        from scipy.linalg import hadamard

        return hadamard(N) / np.sqrt(N)

    if matrix_type == "kronecker":
        from .kronecker_matrix import kronecker_angles, kronecker_matrix

        theta = kronecker_angles(N) if angles is None else angles
        return kronecker_matrix(theta, kernel_type)

    if matrix_type == "circulant":
        r_fft = np.fft.fft(rng.standard_normal(N))
        r_fft /= np.abs(r_fft)
        r = np.real(np.fft.ifft(r_fft))
        direction = int(rng.choice([-1, 1]))
        return _circulant(r, direction)

    if matrix_type == "householder":
        return householder_matrix(rng.random(N))

    if matrix_type == "parallel":
        return np.eye(N)

    if matrix_type == "permutation":
        P = np.eye(N)
        return P[rng.permutation(N)]

    if matrix_type == "diagonally_similar_to_orthogonal":
        D = np.diag(rng.standard_normal(N))
        return np.linalg.solve(D, random_orthogonal(N, rng=rng)) @ D

    if matrix_type == "tiny_rotation":
        from .orthogonal import tiny_rotation_matrix

        return tiny_rotation_matrix(N, 0.01, rng=rng)

    if matrix_type == "anderson":
        from .orthogonal import anderson_matrix

        return anderson_matrix(N, rng=rng)

    raise ValueError(
        f"Unknown matrix_type {matrix_type!r}. Supported: {FEEDBACK_MATRIX_TYPES}"
    )


def fdn_system_gallery(
    N: int,
    system_type: str,
    *,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a full FDN system ``(A, B, C, D)`` of the requested type.

    Args:
        N: System order.
        system_type: One of :data:`SYSTEM_TYPES`.
        rng: Generator or integer seed for the random gains.

    Returns:
        The system matrices ``(A, B, C, D)``, as the structure builders in
        :mod:`pyFDN.generate.structures` return them.

    Example::

        A, B, C, D = fdn_system_gallery(8, "allpass_in_fdn")
    """
    rng = as_generator(rng)
    if system_type == "series_allpass":
        from .structures import series_allpass

        g = rng.random(N) * 0.6 + 0.2
        return series_allpass(g)

    if system_type == "nested_allpass":
        from .structures import nested_allpass

        g = rng.random(N) * 0.6 + 0.2
        return nested_allpass(g)

    if system_type == "poletti_allpass":
        from .structures import poletti_allpass

        return poletti_allpass(0.7, random_orthogonal(N, rng=rng))

    if system_type == "homogeneous_allpass":
        from .allpass_fdn.homogeneous_allpass_fdn import homogeneous_allpass_fdn
        from .allpass_fdn.random_admissible_homogeneous_allpass import (
            random_admissible_homogeneous_allpass,
        )

        G = np.diag(rng.uniform(0.9, 0.99, N))
        X = random_admissible_homogeneous_allpass(G, (0.8, 0.99), rng=rng)
        return homogeneous_allpass_fdn(G, X)[:4]

    if system_type == "schroeder_reverberator":
        from .structures import schroeder_reverberator

        N_c = N // 2
        N_a = N - N_c
        comb_gain = rng.uniform(0.5, 0.9, N_c) * rng.choice([-1, 1], N_c)
        allpass_gain = rng.uniform(0.2, 0.7, N_a)
        B = rng.standard_normal((N_c, 1)) / np.sqrt(N_c)
        C = rng.standard_normal((1, N_c)) / np.sqrt(N_c)
        return schroeder_reverberator(allpass_gain, comb_gain, B, C, 0.0)

    if system_type == "allpass_in_fdn":
        from .structures import allpass_in_fdn

        g = rng.uniform(-0.8, 0.8, N // 2)
        A_inner = random_orthogonal(N // 2, rng=rng)
        b_inner = rng.standard_normal((N // 2, 1)) / np.sqrt(N // 2)
        c_inner = rng.standard_normal((1, N // 2)) / np.sqrt(N // 2)
        return allpass_in_fdn(g, A_inner, b_inner, c_inner, 0.0)

    raise ValueError(f"Unknown system_type {system_type!r}. Supported: {SYSTEM_TYPES}")
