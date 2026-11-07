"""Internal validation of modal data against the original all-port transfer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

ModalResult = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]


def refine_dss_poles(
    poles: np.ndarray, delays: np.ndarray, a: np.ndarray
) -> np.ndarray:
    """Independent SVD null-vector Newton steps on diag(z**m) - A."""
    refined = np.asarray(poles, dtype=complex).copy()
    for index, z in enumerate(refined):
        for _ in range(5):
            p = np.diag(z**delays) - a
            dp = np.diag(delays * z ** (delays - 1))
            scale = np.max(np.abs(p))
            if scale > 0:
                p, dp = p / scale, dp / scale
            u, s, vh = np.linalg.svd(p)
            denominator = np.vdot(u[:, -1], dp @ vh.conj().T[:, -1])
            if abs(denominator) < 1e-30:
                break
            dz = -s[-1] / denominator
            z += dz
            if abs(dz) < 1e-14:
                break
        refined[index] = z
    return refined


def validate_modal_response(
    poles: np.ndarray,
    residues: np.ndarray,
    pairs: np.ndarray,
    probe: Callable[[complex], np.ndarray],
    order: int,
    tolerance: float,
) -> dict[str, float | int]:
    """Check the wet transfer at deterministic off-grid points outside all poles.

    `residues` and `probe` must include every internal input/output port. This
    sampled check is an acceptance criterion, not a uniform error bound.
    The contour corresponds to an exponential window of at least 4096 samples.
    """
    if (
        poles.size == 0
        or not np.all(np.isfinite(poles))
        or not np.all(np.isfinite(residues))
    ):
        raise RuntimeError(
            "Modal response contains empty or nonfinite pole/residue data"
        )
    full_poles = np.concatenate((poles, poles[pairs].conj()))
    full_residues = np.concatenate((residues, residues[pairs].conj()))
    radius = max(1.0, float(np.max(np.abs(full_poles)))) * np.exp(
        1 / max(4096, 8 * order)
    )
    count = max(256, 2 * order + 1)
    points = radius * np.exp(1j * np.pi * (np.arange(count) + 0.37) / count)
    error_sq = reference_sq = error_peak = reference_peak = 0.0
    for start in range(0, count, 64):
        z = points[start : start + 64]
        reference = np.array([probe(complex(point)) for point in z])
        modal = (
            (1 / (z[:, None] - full_poles)) @ full_residues.reshape(full_poles.size, -1)
        ).reshape(reference.shape)
        error = modal - reference
        if not np.all(np.isfinite(error)):
            raise RuntimeError("Modal response validation produced nonfinite values")
        error_sq += float(np.sum(np.abs(error) ** 2))
        reference_sq += float(np.sum(np.abs(reference) ** 2))
        error_peak = max(error_peak, float(np.max(np.abs(error))))
        reference_peak = max(reference_peak, float(np.max(np.abs(reference))))
    relative = float(np.sqrt(error_sq / max(reference_sq, np.finfo(float).tiny)))
    peak = error_peak / max(reference_peak, np.finfo(float).tiny)
    if max(relative, peak) > tolerance:
        raise RuntimeError(
            f"All-port modal response validation failed: relative L2={relative:.3g}, normalized peak={peak:.3g}, tolerance={tolerance:.3g}"
        )
    return {
        "relativeL2": relative,
        "normalizedPeak": peak,
        "points": count,
        "radius": float(radius),
    }
