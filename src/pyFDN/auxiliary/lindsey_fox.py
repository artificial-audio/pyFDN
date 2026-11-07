"""Experimental Lindsey–Fox grid search for a bounded annulus of simple roots.

Independent implementation of the FFT search / Laguerre polish / coefficient
verification described by Sitton et al., IEEE Signal Processing Magazine 20(6),
2003, doi:10.1109/MSP.2003.1253552, with Laguerre polishing as in the authors'
2006 three-stage description. This is not a port of Rice's MATLAB code.
"""

from __future__ import annotations

from time import monotonic

import numpy as np
from scipy.spatial import cKDTree

from pyFDN.auxiliary._fft_polynomial import FFTPolynomial


def _unfactor(roots: np.ndarray) -> np.ndarray:
    """Balanced product with interleaved angles to limit intermediate growth."""
    if roots.size == 0:
        return np.ones(1, dtype=complex)
    if roots.size == 1:
        return np.array([-roots[0], 1.0])
    left, right = _unfactor(roots[::2]), _unfactor(roots[1::2])
    if left.size * right.size <= 4096:
        return np.convolve(left, right)
    length = left.size + right.size - 1
    nfft = 1 << (length - 1).bit_length()
    return np.fft.ifft(np.fft.fft(left, nfft) * np.fft.fft(right, nfft))[:length]


def _unique_roots(roots: np.ndarray) -> np.ndarray:
    """Preserve angular-order greedy uniqueness using linear tree storage."""
    roots = roots[np.argsort(np.angle(roots))]
    if roots.size < 2:
        return roots
    coordinates = np.column_stack((roots.real, roots.imag))
    tree = cKDTree(coordinates)
    distances, _ = tree.query(coordinates, k=2)
    if np.all(distances[:, 1] > 1e-8):
        return roots
    blocked = np.zeros(roots.size, dtype=bool)
    selected = []
    for index in range(roots.size):
        if not blocked[index]:
            selected.append(index)
            blocked[tree.query_ball_point(coordinates[index], 1e-8)] = True
    return roots[selected]


def _polish(roots: np.ndarray, evaluator: FFTPolynomial, degree: int) -> np.ndarray:
    """Independent Laguerre iterations, freezing converged candidates."""
    active = np.ones(roots.size, dtype=bool)
    for _ in range(80):
        evaluator.check_time()
        indices = np.flatnonzero(active)
        if indices.size == 0:
            break
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            p, dp, ddp, _, _ = evaluator.evaluate(roots[indices])
            g = dp / p
            h = g * g - ddp / p
            radical = np.sqrt((degree - 1) * (degree * h - g * g))
            plus, minus = g + radical, g - radical
            denominator = np.where(np.abs(plus) > np.abs(minus), plus, minus)
            step = degree / denominator
        step = np.where(p == 0, 0, step)
        roots[indices] -= step
        active[indices] = np.isfinite(roots[indices]) & (
            np.abs(step) >= 1e-13 * np.maximum(1.0, abs(roots[indices]))
        )
    return roots[np.isfinite(roots)]


def _complete(roots: np.ndarray, evaluator: FFTPolynomial, degree: int) -> np.ndarray:
    """Seed a nearby missing partner from curvature; acceptance remains separate."""
    for _ in range(3):
        if not 0 < degree - roots.size <= degree // 8:
            break
        p, dp, ddp, _, _ = evaluator.evaluate(roots)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Remove the uniform-circle background from P''/(2P').
            step = -1 / (ddp / (2 * dp) - (degree - 1) / (2 * roots))
            seeds = roots + step
        selected = (
            np.isfinite(seeds) & (abs(step) < abs(roots) / degree) & (abs(step) > 1e-8)
        )
        seeds, step = seeds[selected], step[selected]
        tree = cKDTree(np.column_stack((roots.real, roots.imag)))
        distance, _ = tree.query(np.column_stack((seeds.real, seeds.imag)))
        seeds = seeds[distance > np.maximum(1e-8, 0.2 * abs(step))]
        if not seeds.size:
            break
        seeds = _polish(_unique_roots(seeds), evaluator, degree)
        combined = _unique_roots(np.concatenate((roots, seeds)))
        if combined.size <= roots.size:
            break
        roots = combined
    return roots


def lindsey_fox_roots(
    coefficients: np.ndarray,
    radius_min: float,
    radius_max: float,
    *,
    max_attempts: int = 10,
    max_seconds: float = 3.0,
    max_fft_size: int = 2**21,
) -> np.ndarray:
    """Find all simple roots in an annulus; coefficients are ascending powers.

    Uses FFT evaluations on concentric circles, 3×3 modulus minima, vectorized
    Laguerre polishing against the original polynomial, and reconstruction.
    Large near-complete sets also try curvature-based seeds for missing partners.
    This bounded heuristic supplements the grid; every acceptance check remains.
    Up to ten progressively finer searches are attempted, bounded by a time
    budget and FFT size. The time budget is checked between numerical operations,
    so one FFT or polishing step may overrun it. A failed completeness or
    reconstruction check raises ``RuntimeError``; no companion-matrix fallback
    hides a failure or contaminates a timing comparison.

    shortcut: no quotient deflation or multiple-root handling; use eig/roots/EAI
    for those cases, or extend this experimental backend if FDN trials need it.
    """
    coefficients = np.asarray(coefficients, dtype=complex)
    if (
        coefficients.ndim != 1
        or coefficients.size < 2
        or not np.all(np.isfinite(coefficients))
        or coefficients[0] == 0
        or coefficients[-1] == 0
    ):
        raise ValueError("Expected finite polynomial coefficients with nonzero ends")
    if not 0 < radius_min <= radius_max < np.inf:
        raise ValueError("Expected 0 < radius_min <= radius_max < infinity")
    if (
        max_attempts < 1
        or not np.isfinite(max_seconds)
        or max_seconds <= 0
        or max_fft_size < 1
    ):
        raise ValueError("Search limits must be positive and max_seconds finite")
    deadline = monotonic() + max_seconds
    coefficients = coefficients / np.max(np.abs(coefficients))
    degree = coefficients.size - 1
    if degree == 1:
        return np.array([-coefficients[0] / coefficients[1]])
    powers = np.arange(degree + 1)

    for attempt in range(max_attempts):
        nfft = 1 << int(np.ceil(np.log2(8 * degree)))
        nfft *= 2**attempt
        if nfft > max_fft_size:
            break
        spacing = 1.0 / (degree * 2**attempt)
        low = np.log(radius_min) - 2 * spacing
        high = np.log(radius_max) + 2 * spacing
        radial_count = max(5, int((high - low) / spacing) + 2)
        # Stream even very broad annuli; the budget can stop the scan safely.
        radii = (
            np.exp(low + (high - low) * k / (radial_count - 1))
            for k in range(radial_count)
        )
        angles = np.exp(2j * np.pi * np.arange(nfft) / nfft)
        candidates = []
        previous = None
        center = None
        center_radius = 0.0
        for radius in radii:
            if monotonic() >= deadline:
                raise RuntimeError("Lindsey-Fox search exceeded its time budget")
            with np.errstate(over="ignore", invalid="ignore"):
                values = np.abs(np.fft.ifft(coefficients * radius**powers, n=nfft))
            if not np.all(np.isfinite(values)):
                raise RuntimeError("Lindsey-Fox encountered nonfinite circle samples")
            if previous is not None and center is not None:
                minimum = (center <= np.roll(center, 1)) & (
                    center < np.roll(center, -1)
                )
                for neighbor in (previous, values):
                    for shift in (-1, 0, 1):
                        minimum &= center < np.roll(neighbor, shift)
                candidates.extend(center_radius * angles[minimum])
            previous, center = center, values
            center_radius = radius
        roots = np.asarray(candidates, dtype=complex)
        evaluator = FFTPolynomial(coefficients, nfft, deadline)
        roots = _unique_roots(_polish(roots, evaluator, degree))
        if degree >= 2048:
            roots = _complete(roots, evaluator, degree)
        if roots.size != degree:
            continue
        p, _, _, allowance, scale = evaluator.evaluate(roots, derivatives=False)
        error = np.abs(p) + allowance
        evaluator.check_time()
        reconstructed = coefficients[-1] * _unfactor(roots)
        if (
            np.all(np.isfinite(error))
            and np.all(np.isfinite(scale))
            and np.all(error <= 1e-10 * scale)
            and np.max(np.abs(reconstructed - coefficients)) <= 1e-7
        ):
            return roots
    raise RuntimeError(
        "Lindsey-Fox search failed completeness/reconstruction verification; "
        "try mode='eig', 'roots', or 'eai' (clustered/multiple roots are unsupported)"
    )
