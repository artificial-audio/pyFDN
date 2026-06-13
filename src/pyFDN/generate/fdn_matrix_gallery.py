"""Gallery of feedback matrices and FDN systems.

Translation of fdnMatrixGallery.m from fdnToolbox.
"""

# TODO: Menzel matrix

from __future__ import annotations

from typing import NamedTuple, NoReturn, overload

import numpy as np

from .householder_matrix import householder_matrix
from .random_orthogonal import random_orthogonal


class FDNSystem(NamedTuple):
    """Full FDN system matrices returned by :func:`fdn_system_gallery`."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray


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
