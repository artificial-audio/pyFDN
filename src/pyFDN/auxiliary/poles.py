"""Pole utilities: conjugate pairing, residues, and the boundaries poles lie within."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.linalg import svd
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import freqz, group_delay
from scipy.spatial import cKDTree

from pyFDN.auxiliary.utils import ensure_3d


def _conjugate_pair_indices(
    poles: np.ndarray, tol_real: float, tol_pair: float
) -> np.ndarray:
    """Sorted matching, with bounded local assignments for ambiguous clusters."""
    if not np.all(np.isfinite(poles)):
        raise ValueError("Expected finite poles")
    mate = np.full(poles.size, -1, dtype=np.intp)
    real = np.abs(poles.imag) < tol_real / 2
    mate[real] = np.flatnonzero(real)
    positive = np.flatnonzero(~real & (poles.imag > 0))
    negative = np.flatnonzero(~real & (poles.imag < 0))
    upper = poles[positive]
    lower = poles[negative].conj()
    ip = np.lexsort((upper.imag, upper.real))
    im = np.lexsort((lower.imag, lower.real))
    if positive.size == negative.size and np.all(
        np.abs(upper[ip] - lower[im]) < tol_pair
    ):
        resolved = np.array_equal(upper[ip], lower[im]) or upper.size < 2
        if not resolved:
            # Disjoint tolerance balls make the sorted matching unambiguous.
            points = np.column_stack((upper.real, upper.imag))
            distances, _ = cKDTree(points).query(points, k=2)
            resolved = bool(np.all(distances[:, 1] >= 2 * tol_pair))
        if resolved:
            mate[positive[ip]], mate[negative[im]] = negative[im], positive[ip]
            return mate
    indices = np.concatenate((positive, negative))
    if positive.size == 0 or negative.size == 0:
        return mate
    values = np.concatenate((upper, lower))
    coordinates = np.column_stack((values.real, values.imag))
    tree = cKDTree(coordinates)
    visited = np.zeros(indices.size, dtype=bool)
    for seed in range(indices.size):
        if visited[seed]:
            continue
        component = [seed]
        visited[seed] = True
        cursor = 0
        while cursor < len(component):
            neighbors = tree.query_ball_point(coordinates[component[cursor]], tol_pair)
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    component.append(neighbor)
            # Bound both local assignment memory and pathological cluster work.
            if len(component) > 512:
                raise ValueError(
                    "Ambiguous conjugate cluster exceeds the 512-pole local assignment limit"
                )
            cursor += 1
        local = np.asarray(component)
        left = local[local < positive.size]
        right = local[local >= positive.size]
        if left.size == 0 or right.size == 0:
            continue
        cost = np.abs(values[left, None] - values[right])
        allowed = cost < tol_pair
        # Prefer the maximum number of legal pairs before minimizing distances.
        penalty = (max(left.size, right.size) + 1) * tol_pair
        rows, cols = linear_sum_assignment(np.where(allowed, cost, penalty))
        valid = allowed[rows, cols]
        a, b = indices[left[rows[valid]]], indices[right[cols[valid]]]
        mate[a], mate[b] = b, a
    return mate


def reduce_conjugate_pairs(
    poles: np.ndarray | ArrayLike,
    *,
    tol_real: float = 1e-10,
    tol_pair: float = 1e-8,
    verbose: bool = False,
    strict: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group real poles and conjugate pairs without a global pairwise cost matrix.

    Sorted upper/lower-half-plane matching handles complete, well-resolved sets.
    Otherwise a spatial tree identifies tolerance-connected components, with
    optimal assignments restricted to local components of at most 512 poles.
    Larger ambiguous components raise rather than allocating quadratic storage.
    Storage is linear in the pole count plus one bounded local cost matrix.
    A real pole satisfies ``2*abs(imag(pole)) < tol_real``; pair distances must
    be below ``tol_pair``. Output order follows the original input order.

    Unpaired poles are reported via ``non_paired`` AND a :class:`UserWarning`
    so callers cannot silently lose poles to imprecise pairing. Set
    ``strict=True`` to raise :class:`ValueError` instead of warning.

    Returns
    -------
    poles_out : np.ndarray
        One representative per real pole and per conjugate pair (imag >= 0).
    is_conjugate : np.ndarray
        Boolean, same length as poles_out: False for real, True for conjugate pair or unpaired.
    non_paired : np.ndarray
        Poles that could not be paired.
    """
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    n_poles = poles.size
    mate = _conjugate_pair_indices(poles, tol_real, tol_pair)
    indices = np.arange(n_poles)
    select = (mate >= 0) & (indices <= mate)
    is_conjugate = mate != indices
    if verbose:
        print(
            f"Poles reduction: {np.sum(select & ~is_conjugate)} real, "
            f"{np.sum(select & is_conjugate)} conjugate pairs, {np.sum(mate < 0)} unpaired"
        )

    non_paired = poles[mate < 0]

    if non_paired.size > 0:
        msg = (
            f"reduce_conjugate_pairs: {non_paired.size} unpaired pole(s) "
            f"will be dropped (tol_pair={tol_pair:.1e}, tol_real={tol_real:.1e}). "
            f"This often indicates imprecise pole estimates upstream."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)

    # Mirror the poles to the upper half of the complex plane
    poles = np.real(poles) + 1j * np.abs(np.imag(poles))
    return poles[select], is_conjugate[select], non_paired


def residue_at_pole(
    P: np.ndarray, dP: np.ndarray, B: np.ndarray, C: np.ndarray
) -> tuple[complex, np.ndarray, np.ndarray, np.ndarray]:
    """Residue terms of ``C P(z)^{-1} B`` at a pole ``z_k`` where ``P`` is singular.

    With ``r`` and ``l`` the right and left null vectors of ``P(z_k)``, the
    residue is ``(C r)(l^H B) / (l^H P'(z_k) r)``. ``P`` is scaled before the
    SVD (null vectors are scale-invariant) so poles far from the unit circle,
    where ``z^m`` terms blow up, stay well conditioned.

    Returns
    -------
    denominator : complex
        ``l^H P'(z_k) r``; zero for a multiple pole.
    numerator : ndarray ``(n_out, n_in)``
        ``(C r)(l^H B)``.
    right, left : ndarray ``(N,)``
        The null vectors.
    """
    scale = np.max(np.abs(P))
    if scale > 0 and np.isfinite(scale):
        P = P / scale
    u, singular_values, vh = np.linalg.svd(P)
    if P.shape[0] > 1 and singular_values[-2] * scale <= (
        np.finfo(float).eps * P.shape[0] * max(scale, float(np.max(np.abs(dP))))
    ):
        raise ValueError(
            "Multiple null vectors cannot be represented by simple residues"
        )
    # These vectors are retained for every mode; views would keep full SVD matrices.
    right = vh.conj().T[:, -1].copy()
    left = u[:, -1].copy()
    denominator = complex(np.vdot(left, dP @ right))
    numerator = np.outer(C @ right, left.conj() @ B)
    return denominator, numerator, right, left


def residues_from_terms(
    numerators: np.ndarray, denominators: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Residues ``numerator / denominator``; reject unsupported multiple poles.

    Returns ``(residues, undriven)`` where ``undriven = 1 / denominator``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        undriven = 1.0 / denominators
        residues = numerators / denominators[:, None, None]
    if np.any(~np.isfinite(undriven)) or np.any(~np.isfinite(residues)):
        raise ValueError(
            "Multiple or nonfinite poles cannot be represented by simple residues"
        )
    return residues, undriven


def assemble_residues(
    terms: list[tuple[complex, np.ndarray, np.ndarray, np.ndarray]],
    n: int,
    n_out: int,
    n_in: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Assemble shared SVD residue terms for static and FLAMO loops."""
    denominators = np.array([t[0] for t in terms], dtype=np.complex128)
    numerators = np.array([t[1] for t in terms], dtype=np.complex128).reshape(
        len(terms), n_out, n_in
    )
    right = np.array([t[2] for t in terms], dtype=np.complex128).reshape(-1, n).T
    left = np.array([t[3] for t in terms], dtype=np.complex128).reshape(-1, n).T
    residues, undriven = residues_from_terms(numerators, denominators)
    return residues, undriven, {"right": right, "left": left}


def is_bounding_curve(
    x_points: ArrayLike,
    y_points: ArrayLike,
    x_curve: ArrayLike,
    y_curve: ArrayLike,
    bound_type: str,
) -> bool:
    """
    Check if all value points are bounded by the curve.
    Parameters
    ----------
    x_points
        x-coordinates of data points (1D array)
    y_points
        y-coordinates of data points (1D array)
    x_curve
        x-coordinates of curve points (1D array)
    y_curve
        y-coordinates of curve points (1D array)
    bound_type
        'upper' or 'lower'
    Returns
    -------
    bool
        Whether every data point lies on the bounded side of the curve.
    """
    # Spline interpolation with extrapolation
    interp = interp1d(x_curve, y_curve, kind="cubic", fill_value="extrapolate")
    y_curve_interp = interp(x_points)

    if bound_type == "upper":
        is_bounded = y_curve_interp >= y_points
    elif bound_type == "lower":
        is_bounded = y_curve_interp <= y_points
    else:
        raise ValueError("bound_type must be 'upper' or 'lower'")

    return bool(np.all(is_bounded))


def pole_boundaries(
    delays: ArrayLike,
    A: ArrayLike,
    fs: float,
    *,
    post_delay: ArrayLike,
    nfft: int = 2**12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper and lower bounds of the pole magnitudes of an FDN loop.

    Combines the singular values of the feedback matrix with the magnitude
    and group delay of the per-line absorption filters.

    Parameters
    ----------
    delays
        Delays in samples, shape ``(N,)``.
    A
        Feedback matrix ``(N, N)``, or the FIR matrix ``(N, N, L)`` in the
        z^{-1} convention that sits on the feedback path.
    fs
        Sampling frequency in Hz.
    post_delay
        Per-line absorption, the ``post_delay`` filter hook as an
        SOS bank of shape ``(sections, 6, N)``.
    nfft
        Number of frequency bins between 0 and Nyquist.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        min_curve: Lower bound of the pole magnitude, shape ``(nfft,)``.
        max_curve: Upper bound of the pole magnitude, shape ``(nfft,)``.
        f: Frequency points in Hz, shape ``(nfft,)``.
    """
    delays_arr = np.asarray(delays, dtype=float).ravel()
    sos = np.asarray(post_delay, dtype=float)
    if sos.ndim != 3 or sos.shape[1] != 6 or sos.shape[2] != delays_arr.size:
        raise ValueError("post_delay must be an SOS bank of shape (sections, 6, N)")
    # The FFT bins of a 2 * nfft transform: 0 up to (not including) Nyquist.
    w = np.pi * np.arange(nfft) / nfft

    spectrum = np.fft.fft(ensure_3d(A), n=nfft * 2, axis=2)[:, :, :nfft]
    singular = np.array([svd(spectrum[:, :, k], compute_uv=False) for k in range(nfft)])
    matrix_min = np.min(singular, axis=1) ** (1 / np.min(delays_arr))
    matrix_max = np.max(singular, axis=1) ** (1 / np.max(delays_arr))

    magnitude = np.ones((nfft, delays_arr.size))
    group = np.zeros((nfft, delays_arr.size))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for line in range(delays_arr.size):
            for section in sos[:, :, line]:
                _, h = freqz(section[:3], section[3:], w)
                _, gd = group_delay((section[:3], section[3:]), w)
                magnitude[:, line] *= np.abs(h)
                group[:, line] += gd

    per_line = magnitude ** (1.0 / (delays_arr + group))
    min_curve = np.min(per_line, axis=1) * matrix_min
    max_curve = np.max(per_line, axis=1) * matrix_max
    return min_curve, max_curve, w / np.pi * fs / 2
