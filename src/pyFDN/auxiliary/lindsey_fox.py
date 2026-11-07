"""Experimental Lindsey–Fox grid search for a bounded annulus of simple roots.

Independent implementation of the FFT search / Laguerre polish / coefficient
verification described by Sitton et al., IEEE Signal Processing Magazine 20(6),
2003, doi:10.1109/MSP.2003.1253552, with Laguerre polishing as in the authors'
2006 three-stage description. This is not a port of Rice's MATLAB code.
"""

from __future__ import annotations

import numpy as np


def _unfactor(roots: np.ndarray) -> np.ndarray:
    """Balanced product with interleaved angles to limit intermediate growth."""
    if roots.size == 0:
        return np.ones(1, dtype=complex)
    if roots.size == 1:
        return np.array([-roots[0], 1.0])
    return np.convolve(_unfactor(roots[::2]), _unfactor(roots[1::2]))


def lindsey_fox_roots(
    coefficients: np.ndarray, radius_min: float, radius_max: float
) -> np.ndarray:
    """Find all simple roots in an annulus; coefficients are ascending powers.

    Uses FFT evaluations on concentric circles, 3×3 modulus minima, vectorized
    Laguerre polishing against the original polynomial, and reconstruction.
    Four progressively finer searches are attempted. A failed completeness or
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
    coefficients = coefficients / np.max(np.abs(coefficients))
    degree = coefficients.size - 1
    if degree == 1:
        return np.array([-coefficients[0] / coefficients[1]])
    powers = np.arange(degree + 1)
    derivative = coefficients[1:] * powers[1:]
    second = derivative[1:] * powers[1:-1]
    polyval = np.polynomial.polynomial.polyval

    for attempt in range(4):
        nfft = 1 << int(np.ceil(np.log2(8 * degree)))
        nfft *= 2**attempt
        spacing = 1.0 / (degree * 2**attempt)
        low = np.log(radius_min) - 2 * spacing
        high = np.log(radius_max) + 2 * spacing
        radii = np.exp(np.linspace(low, high, max(5, int((high - low) / spacing) + 2)))
        angles = np.exp(2j * np.pi * np.arange(nfft) / nfft)
        candidates = []
        previous = None
        center = None
        for index, radius in enumerate(radii):
            values = np.abs(np.fft.ifft(coefficients * radius**powers, n=nfft))
            if previous is not None and center is not None:
                minimum = (center <= np.roll(center, 1)) & (
                    center < np.roll(center, -1)
                )
                for neighbor in (previous, values):
                    for shift in (-1, 0, 1):
                        minimum &= center < np.roll(neighbor, shift)
                candidates.extend(radii[index - 1] * angles[minimum])
            previous, center = center, values
        roots = np.asarray(candidates, dtype=complex)
        for _ in range(80):
            if roots.size == 0:
                break
            p = polyval(roots, coefficients)
            dp = polyval(roots, derivative)
            ddp = polyval(roots, second)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                g = dp / p
                h = g * g - ddp / p
                radical = np.sqrt((degree - 1) * (degree * h - g * g))
                plus, minus = g + radical, g - radical
                denominator = np.where(np.abs(plus) > np.abs(minus), plus, minus)
                step = degree / denominator
            step = np.where(p == 0, 0, step)
            roots -= step
            if np.all(np.abs(step) < 1e-13 * np.maximum(1.0, np.abs(roots))):
                break
        roots = roots[np.isfinite(roots)]
        # Greedy uniqueness avoids allocating a candidate-by-candidate matrix.
        unique: list[complex] = []
        for root in roots[np.argsort(np.angle(roots))]:
            if not unique or np.min(np.abs(np.asarray(unique) - root)) > 1e-8:
                unique.append(root)
        roots = np.asarray(unique, dtype=complex)
        if roots.size != degree:
            continue
        error = np.abs(polyval(roots, coefficients))
        scale = polyval(np.abs(roots), np.abs(coefficients))
        reconstructed = coefficients[-1] * _unfactor(roots)
        if (
            np.all(error <= 1e-10 * scale)
            and np.max(np.abs(reconstructed - coefficients)) <= 1e-7
        ):
            return roots
    raise RuntimeError(
        "Lindsey-Fox search failed completeness/reconstruction verification; "
        "try mode='eig', 'roots', or 'eai' (clustered/multiple roots are unsupported)"
    )
