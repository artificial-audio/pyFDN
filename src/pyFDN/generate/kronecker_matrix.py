"""Parametric orthogonal feedback matrices built from Kronecker products.

Implementation of the Kronecker Feedback Matrix of Coppola (2026),
``Fast parametric matrices for lossless feedback delay networks`` (DAFx26).

A matrix of size ``N = 2**M`` is the Kronecker product of ``M`` independent
2x2 orthogonal kernels, each parameterised by a single angle::

    Psi_M = K_M kron K_{M-1} kron ... kron K_1

with ``K_i`` either a rotation :func:`rotation_kernel` or a reflection
:func:`reflection_kernel`.  The product of orthogonal matrices under ``kron``
is orthogonal, so ``Psi_M`` is lossless for *every* angle -- no
re-orthogonalisation is ever needed while the angles move.

Each kernel index addresses one bit of the delay-line index, which is what
makes the parameterisation structurally meaningful:

* ``theta_M`` (outermost) mixes the two contiguous halves of the network.  At
  ``theta_M = 0`` the halves are independent, which is the stereo
  cross-coupling control.
* ``theta_1`` (innermost) mixes adjacent pairs, i.e. it couples the
  even- and odd-indexed delay lines.  At ``theta_1 = 0`` the network splits
  into an even and an odd subnetwork.
* Intermediate kernels are free for time-varying modulation, see
  :class:`pyFDN.td.TimeVaryingKroneckerMatrix`.

Setting every kernel to ``reflection_kernel(pi / 4)`` recovers the normalised
Hadamard matrix, so the family contains the standard FDN mixing matrix as one
point and morphs continuously away from it.

Because the matrix is never formed, :func:`kronecker_transform` applies it in
``O(N log2 N)`` operations instead of ``O(N**2)`` -- the same cost as the fast
Walsh-Hadamard transform, but with ``M`` free parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike

KernelType = Literal["rotation", "reflection"]
KERNEL_TYPES = get_args(KernelType)


def rotation_kernel(theta: float) -> np.ndarray:
    """2x2 rotation by ``theta``: ``[[cos, -sin], [sin, cos]]``.

    ``rotation_kernel(0)`` is the identity, so a zero angle decouples the two
    branches that the kernel addresses.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def reflection_kernel(theta: float) -> np.ndarray:
    """2x2 reflection at half-angle ``theta``: ``[[cos, sin], [sin, -cos]]``.

    Reflects across the line at ``theta / 2`` from the horizontal axis; the
    half-angle convention keeps the parameterisation consistent with
    :func:`rotation_kernel`.  ``reflection_kernel(pi / 4)`` is the order-2
    Hadamard matrix ``[[1, 1], [1, -1]] / sqrt(2)``.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [s, -c]])


_KERNELS = {"rotation": rotation_kernel, "reflection": reflection_kernel}


def _normalise_kernel_types(
    kernel_type: str | Sequence[str], M: int
) -> tuple[str, ...]:
    """Broadcast ``kernel_type`` to one entry per kernel and validate it."""
    types = (kernel_type,) * M if isinstance(kernel_type, str) else tuple(kernel_type)
    if len(types) != M:
        raise ValueError(
            f"kernel_type has {len(types)} entries but there are {M} kernels"
        )
    for name in types:
        if name not in _KERNELS:
            raise ValueError(
                f"Unknown kernel_type {name!r}. Supported: {list(KERNEL_TYPES)}"
            )
    return types


def _angles_1d(angles: ArrayLike) -> np.ndarray:
    """Validate and return the angle vector ``[theta_1, ..., theta_M]``."""
    a = np.asarray(angles, dtype=float).reshape(-1)
    if a.size == 0:
        raise ValueError("angles must contain at least one angle")
    return a


def kronecker_angles(
    N: int, angle: float = np.pi / 4, **overrides: float
) -> np.ndarray:
    """Angle vector ``[theta_1, ..., theta_M]`` for a network of size ``N``.

    A convenience for the common case of "all kernels at one value, except a
    named few".  ``theta_1`` is the innermost (even/odd) kernel and
    ``theta_M`` the outermost (contiguous halves).

    Parameters
    ----------
    N : int
        Network size; must be a power of two.  ``M = log2(N)`` angles are
        returned.
    angle : float
        Default value for every kernel (default ``pi / 4``, the
        equal-mixing setting that yields Hadamard-like magnitudes).
    **overrides : float
        Per-kernel overrides given as ``theta1=..., theta2=...``, using the
        paper's one-based indexing.  Negative indices are not supported;
        use ``theta{M}`` for the outermost kernel.

    Returns
    -------
    np.ndarray
        Angles of shape ``(M,)``, ordered innermost to outermost.

    Example::

        # 32 channels, fully decoupled stereo halves (theta_M = 0)
        kronecker_angles(32, theta5=0.0)
    """
    M = num_kernels(N)
    a = np.full(M, float(angle))
    for key, value in overrides.items():
        if not key.startswith("theta"):
            raise TypeError(f"Unexpected keyword argument {key!r}")
        index = int(key.removeprefix("theta"))
        if not 1 <= index <= M:
            raise ValueError(f"{key!r} is out of range for N = {N} (M = {M})")
        a[index - 1] = float(value)
    return a


def num_kernels(N: int) -> int:
    """Number of kernels ``M = log2(N)``, raising unless ``N`` is a power of two."""
    N = int(N)
    if N < 2 or N & (N - 1):
        raise ValueError(f"N must be a power of two and at least 2, got {N}")
    return int(N).bit_length() - 1


def kronecker_matrix(
    angles: ArrayLike,
    kernel_type: str | Sequence[str] = "rotation",
) -> np.ndarray:
    """Build the explicit Kronecker feedback matrix ``Psi_M``.

    Formed by the recursion ``Psi_0 = 1``, ``Psi_m = K_m kron Psi_{m-1}``, so
    ``angles[0]`` is the innermost kernel ``theta_1`` and ``angles[-1]`` the
    outermost ``theta_M``.  The result is orthogonal for any angles.

    Use this for analysis and plotting; for rendering prefer
    :func:`kronecker_transform`, which never forms the matrix and costs
    ``O(N log2 N)`` per input vector.

    Parameters
    ----------
    angles : array-like
        ``M`` kernel angles in radians, innermost first.
    kernel_type : str or sequence of str
        ``"rotation"`` or ``"reflection"``, either once for all kernels or one
        per kernel (innermost first).

    Returns
    -------
    np.ndarray
        Orthogonal matrix of shape ``(2**M, 2**M)``.

    Example::

        >>> import numpy as np
        >>> A = kronecker_matrix([np.pi / 4] * 3, "reflection")
        >>> np.allclose(A @ A.T, np.eye(8))
        True
    """
    a = _angles_1d(angles)
    types = _normalise_kernel_types(kernel_type, a.size)

    matrix = np.ones((1, 1))
    for theta, name in zip(a, types, strict=True):
        matrix = np.kron(_KERNELS[name](theta), matrix)
    return matrix


def kronecker_transform(
    x: ArrayLike,
    angles: ArrayLike,
    kernel_type: str | Sequence[str] = "rotation",
) -> np.ndarray:
    """Apply the Kronecker feedback matrix to ``x`` in ``O(N log2 N)`` per row.

    Equivalent to ``x @ kronecker_matrix(angles, kernel_type).T`` but computed
    with the in-place divide-and-conquer butterfly of the paper's Algorithm 2:
    ``M = log2(N)`` levels, each touching every sample once.  Level ``i`` mixes
    the channel pairs that differ in bit ``i - 1`` of their index, which is
    exactly the pairing kernel ``K_i`` addresses.

    Angles may vary per sample, which is what makes an audio-rate modulated
    feedback matrix cheap: only the modulated 2x2 kernel changes, never a
    rebuilt ``N x N`` matrix.

    Parameters
    ----------
    x : array-like
        Signal of shape ``(N,)`` or ``(num_samples, N)``; the last axis is the
        channel axis.
    angles : array-like
        ``M`` angles of shape ``(M,)`` for a fixed matrix, or
        ``(num_samples, M)`` for one angle set per sample.
    kernel_type : str or sequence of str
        ``"rotation"`` or ``"reflection"``, either once for all kernels or one
        per kernel (innermost first).

    Returns
    -------
    np.ndarray
        Mixed signal, same shape as ``x`` (2-D input stays 2-D).

    Example::

        >>> import numpy as np
        >>> angles = np.full(3, np.pi / 4)
        >>> x = np.random.randn(16, 8)
        >>> fast = kronecker_transform(x, angles)
        >>> np.allclose(fast, x @ kronecker_matrix(angles).T)
        True
    """
    signal = np.asarray(x, dtype=float)
    vector_input = signal.ndim == 1
    if vector_input:
        signal = signal[np.newaxis, :]
    if signal.ndim != 2:
        raise ValueError("x must be 1-D or 2-D (num_samples, channels)")

    theta = np.asarray(angles, dtype=float)
    if theta.ndim == 1:
        theta = theta[np.newaxis, :]
    if theta.ndim != 2:
        raise ValueError("angles must be 1-D (M,) or 2-D (num_samples, M)")

    num_samples, N = signal.shape
    M = theta.shape[1]
    if N != 1 << M:
        raise ValueError(
            f"x has {N} channels but {M} angles describe a {1 << M} matrix"
        )
    if theta.shape[0] not in (1, num_samples):
        raise ValueError(
            f"angles has {theta.shape[0]} rows, expected 1 or {num_samples}"
        )
    types = _normalise_kernel_types(kernel_type, M)

    out = np.ascontiguousarray(signal, dtype=float)
    if out is signal:
        out = out.copy()

    for level, name in enumerate(types):
        stride = 1 << level  # pairs differ in bit `level` of the channel index
        blocks = N // (2 * stride)
        c = np.cos(theta[:, level])[:, np.newaxis, np.newaxis]
        s = np.sin(theta[:, level])[:, np.newaxis, np.newaxis]

        pairs = out.reshape(num_samples, blocks, 2, stride)
        upper = pairs[:, :, 0, :]
        lower = pairs[:, :, 1, :]
        if name == "rotation":
            mixed_upper = c * upper - s * lower
            mixed_lower = s * upper + c * lower
        else:  # reflection
            mixed_upper = c * upper + s * lower
            mixed_lower = s * upper - c * lower
        pairs[:, :, 0, :] = mixed_upper
        pairs[:, :, 1, :] = mixed_lower

    return out[0] if vector_input else out
