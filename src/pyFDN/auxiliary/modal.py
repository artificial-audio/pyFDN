"""Internal validation of modal data against the original all-port transfer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

ModalResult = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]


def modal_result_metadata(
    pairs: np.ndarray, order: int, solver: str, allow_approximate: bool = False
) -> dict[str, Any]:
    """Apply the same completeness policy to every modal backend."""
    count = int(np.sum(1 + pairs))
    if count > order or count == 0 or (count != order and not allow_approximate):
        policy = (
            "incomplete results require allow_approximate=True and response validation"
            if solver == "eai"
            else "a complete pole set is required"
        )
        raise RuntimeError(f"{solver} returned {count}/{order} poles; {policy}")
    return {
        "solver": solver,
        "requestedSolver": solver,
        "expectedPoles": order,
        "returnedPoles": count,
        "complete": count == order,
        "approximate": count != order,
        "omittedPoleCount": order - count,
        "responseValidation": None,
    }


@dataclass
class RankOneResidues:
    """Internal residues R[k] = weights[k] * outer(output_vectors[k], input_rows[k])."""

    output_vectors: np.ndarray
    input_rows: np.ndarray
    weights: np.ndarray


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
    residues: np.ndarray | RankOneResidues,
    pairs: np.ndarray,
    probe: Callable[[complex], np.ndarray],
    order: int,
    tolerance: float,
    *,
    static_matrix: np.ndarray | None = None,
    delays: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Check the wet transfer at deterministic off-grid points outside all poles.

    `residues` and `probe` must include every internal input/output port. This
    sampled check is an acceptance criterion, not a uniform error bound.
    The contour corresponds to an exponential window of at least 4096 samples.
    For supported static DSS matrices, a guarded residual bound can certify
    both norms on this same grid. ``errorBound=True`` identifies upper bounds
    rather than directly measured errors; ``referenceProbes`` counts actual
    dense reference evaluations. Inconclusive bounds use the direct check.
    """
    arrays = (
        (residues.output_vectors, residues.input_rows, residues.weights)
        if isinstance(residues, RankOneResidues)
        else (residues,)
    )
    if (
        poles.size == 0
        or not np.all(np.isfinite(poles))
        or not all(np.all(np.isfinite(array)) for array in arrays)
    ):
        raise RuntimeError(
            "Modal response contains empty or nonfinite pole/residue data"
        )
    full_poles = np.concatenate((poles, poles[pairs].conj()))
    if isinstance(residues, RankOneResidues):
        outputs = np.concatenate(
            (residues.output_vectors, residues.output_vectors[pairs].conj())
        )
        inputs = np.concatenate(
            (residues.input_rows, residues.input_rows[pairs].conj())
        )
        weights = np.concatenate((residues.weights, residues.weights[pairs].conj()))
    else:
        full_residues = np.concatenate((residues, residues[pairs].conj()))
    radius = max(1.0, float(np.max(np.abs(full_poles)))) * np.exp(
        1 / max(4096, 8 * order)
    )
    count = max(256, 2 * order + 1)
    points = radius * np.exp(1j * np.pi * (np.arange(count) + 0.37) / count)
    reference_probes = 0

    def measured_probe(z: complex) -> np.ndarray:
        nonlocal reference_probes
        reference_probes += 1
        return probe(z)

    if (
        isinstance(residues, RankOneResidues)
        and static_matrix is not None
        and delays is not None
    ):
        from pyFDN.auxiliary._modal_bound import dss_response_bound

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            bound = dss_response_bound(
                full_poles,
                outputs,
                inputs,
                weights,
                points,
                static_matrix,
                delays,
                tolerance,
                measured_probe,
            )
        if bound is not None:
            return {
                "relativeL2": bound[0],
                "normalizedPeak": bound[1],
                "points": count,
                "radius": float(radius),
                "errorBound": True,
                "referenceProbes": reference_probes,
            }
    error_sq = reference_sq = error_peak = reference_peak = 0.0
    for start in range(0, count, 64):
        z = points[start : start + 64]
        reference = np.array([measured_probe(complex(point)) for point in z])
        if isinstance(residues, RankOneResidues):
            # One all-port matrix at a time; storage is O(poles * ports).
            modal = np.array(
                [
                    outputs.T @ ((weights / (point - full_poles))[:, None] * inputs)
                    for point in z
                ]
            )
        else:
            modal = (
                (1 / (z[:, None] - full_poles))
                @ full_residues.reshape(full_poles.size, -1)
            ).reshape(reference.shape)
        if modal.shape != reference.shape:
            raise RuntimeError(
                "Modal response shape does not match the all-port transfer"
            )
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
        "errorBound": False,
        "referenceProbes": reference_probes,
    }
