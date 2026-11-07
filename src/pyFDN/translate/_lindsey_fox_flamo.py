"""Validated Lindsey-Fox path for canonical real FLAMO FDNs."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import logsumexp

from pyFDN.auxiliary.flamo import to_numpy
from pyFDN.auxiliary.lindsey_fox import lindsey_fox_roots
from pyFDN.auxiliary.modal import ModalResult
from pyFDN.auxiliary.poles import reduce_conjugate_pairs
from pyFDN.translate.flamo_to_pr import (
    _decomposition_to_public_dict,
    _dss_to_res_flamo,
    _FDNLoopFlamo,
    _iter_leaf_modules,
    _refine_pole_via_svd,
    _validate_flamo_result,
    flamo_to_pr,
)


def _model_polynomial(decomposition: Any, degree: int) -> np.ndarray:
    """Clear parallel SOS denominators and recover coefficients once by FFT."""
    rec = decomposition.recursion_module
    denominators: list[np.ndarray] = []
    for branch in (rec.feedforward, rec.feedback):
        for leaf in _iter_leaf_modules(branch):
            if getattr(leaf, "n_sections", None) is not None:
                sos = to_numpy(leaf.map(leaf.param)).reshape(leaf.n_sections, 6, -1)
                denominators.extend(sos[:, 3:, :].transpose(0, 2, 1).reshape(-1, 3))
    nfft = 1 << (2 * degree + 1).bit_length()
    samples = np.empty(nfft // 2 + 1, dtype=complex)
    with torch.no_grad():
        for k in range(samples.size):
            w = np.exp(-2j * np.pi * k / nfft)
            value = torch.tensor(
                w, dtype=torch.complex128, device=next(rec.parameters()).device
            )
            sample = complex(
                torch.linalg.det(rec.probe_recursion_w(value)).cpu().item()
            )
            for denominator in denominators:
                sample *= np.polynomial.polynomial.polyval(w, denominator)
            samples[k] = sample
    if not np.all(np.isfinite(samples)):
        raise RuntimeError("Nonfinite determinant samples")
    coefficients = np.fft.irfft(samples, n=nfft)
    scale = np.max(np.abs(coefficients))
    if np.max(np.abs(coefficients[degree + 1 :])) > 1e-10 * scale:
        raise RuntimeError("Determinant is not a polynomial of the specified order")
    return coefficients[: degree + 1]


def _upper_bound(coefficients: np.ndarray) -> float:
    """Positive root of the absolute-coefficient Cauchy majorant, in log space."""
    degree = coefficients.size - 1
    with np.errstate(divide="ignore"):
        logs = np.log(np.abs(coefficients[:-1])) - np.log(abs(coefficients[-1]))
    powers = degree - np.arange(degree)
    return float(
        np.exp(brentq(lambda x: float(logsumexp(logs - powers * x)), -700, 700))
    )


def lindsey_fox_flamo(
    decomposition: Any,
    order: int,
    *,
    max_seconds: float,
    response_tolerance: float,
    allow_approximate: bool,
    eai_options: dict[str, Any],
) -> ModalResult:
    loop = _FDNLoopFlamo(decomposition.recursion_module)

    try:
        coefficients = _model_polynomial(decomposition, order)
        if coefficients[0] == 0 or coefficients[-1] == 0:
            raise RuntimeError("Scalar polynomial has a zero end coefficient")
        lower = 1 / _upper_bound(coefficients)
        upper = _upper_bound(coefficients[::-1])
        poles = lindsey_fox_roots(
            coefficients[::-1], lower, upper, max_seconds=max_seconds
        )
        poles, pairs, unpaired = reduce_conjugate_pairs(poles, strict=True)
        poles = np.array([_refine_pole_via_svd(complex(z), loop) for z in poles])
        residues, direct, undriven, vectors = _dss_to_res_flamo(
            poles, loop, decomposition
        )
        result: ModalResult = (
            residues,
            poles,
            direct,
            pairs,
            {
                "solver": "lindsey_fox",
                "nonPairedPoles": unpaired,
                "undrivenResidues": undriven,
                "eigenvectors": vectors,
                "decomposition": _decomposition_to_public_dict(decomposition),
                "loop": loop,
            },
        )
        return _validate_flamo_result(
            result,
            decomposition,
            order,
            response_tolerance,
            allow_approximate,
            "lindsey_fox",
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        reason = str(error)
    warnings.warn(
        f"Lindsey-Fox falling back to EAI: {reason}", RuntimeWarning, stacklevel=3
    )
    result = flamo_to_pr(
        decomposition=decomposition,
        num_poles=order,
        allow_approximate=allow_approximate,
        response_tolerance=response_tolerance,
        **eai_options,
    )
    return _validate_flamo_result(
        result,
        decomposition,
        order,
        response_tolerance,
        allow_approximate,
        "lindsey_fox",
        reason,
    )
