"""Modal decomposition for delay state-space (DSS) FDNs.

H(z) = C·(diag(z^m_i) − A)^{-1}·B + D = Σ_k residue_k/(1 − p_k z^{-1}) + D

The static algorithms are NumPy-only. EAI (including fallback) lazily imports torch + FLAMO, so
callers using only ``"eig"``/``"roots"`` never pay that import cost.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.math import general_char_poly, general_char_poly_fft
from pyFDN.auxiliary.modal import ModalResult
from pyFDN.auxiliary.poles import (
    reduce_conjugate_pairs,
    residue_at_pole,
    residues_from_terms,
)
from pyFDN.translate.dss_to_ss import dss_to_ss


def _validated_lindsey_fox(
    delays: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    bounds: np.ndarray,
    max_seconds: float,
    response_tolerance: float,
    eai_options: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    from pyFDN.auxiliary.lindsey_fox import lindsey_fox_roots
    from pyFDN.auxiliary.modal import refine_dss_poles, validate_modal_response

    order = int(delays.sum())
    identity = np.eye(delays.size)

    def validate(
        result: ModalResult, solver: str, reason: str | None = None
    ) -> ModalResult:
        residues, poles, direct, pairs, meta = result
        if int(np.sum(1 + pairs)) != order:
            raise RuntimeError(
                f"{solver} returned an incomplete decomposition; expected {order} poles"
            )
        # Validate all internal ports, even when B/C hide missing modes.
        internal, _, _, _ = _dss_to_res(
            poles, delays, a, identity, identity, np.zeros_like(a)
        )
        expected_residues = np.einsum("oi,kij,jb->kob", c, internal, b)
        if not np.allclose(
            residues,
            expected_residues,
            rtol=response_tolerance,
            atol=response_tolerance
            * max(float(np.max(np.abs(expected_residues))), np.finfo(float).tiny),
        ):
            raise RuntimeError(
                "Modal residues disagree with the all-port decomposition"
            )
        check = validate_modal_response(
            poles,
            internal,
            pairs,
            lambda z: np.linalg.solve(np.diag(z**delays) - a, identity),
            order,
            response_tolerance,
        )
        meta.update(
            solver=solver,
            requestedSolver="lindsey_fox",
            complete=True,
            expectedPoles=order,
            returnedPoles=order,
            approximate=False,
            responseValidation=check,
        )
        if reason is not None:
            meta["fallbackReason"] = reason
        return residues, poles, direct, pairs, meta

    try:
        p = general_char_poly_fft(delays, a)
        poles = lindsey_fox_roots(
            p[::-1], float(bounds.min()), float(bounds.max()), max_seconds=max_seconds
        )
        poles = refine_dss_poles(poles, delays, a)
        poles, pairs, unpaired = reduce_conjugate_pairs(poles, strict=True)
        residues, direct, undriven, vectors = _dss_to_res(poles, delays, a, b, c, d)
        result: ModalResult = (
            residues,
            poles,
            direct,
            pairs,
            {
                "nonPairedPoles": unpaired,
                "undrivenResidues": undriven,
                "eigenvectors": vectors,
            },
        )
        return validate(result, "lindsey_fox")
    except (RuntimeError, np.linalg.LinAlgError, ValueError) as error:
        reason = str(error)
    warnings.warn(
        f"Lindsey-Fox falling back to EAI: {reason}", RuntimeWarning, stacklevel=3
    )
    result = dss_to_pr(delays, a, b, c, d, mode="eai", **eai_options)
    return validate(result, "eai", reason)


def _dss_to_res(
    poles: np.ndarray,
    delays: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Residues at ``poles`` of ``P(z) = diag(z^m) - A`` (SVD null vectors)."""
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    delays = np.asarray(delays, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.complex128)
    B = np.asarray(B, dtype=np.complex128)
    C = np.asarray(C, dtype=np.complex128)
    D = np.asarray(D, dtype=np.complex128)

    terms = [
        residue_at_pole(
            np.diag(np.power(pole, delays)) - A,
            np.diag(delays * np.power(pole, delays - 1.0)),
            B,
            C,
        )
        for pole in poles
    ]
    n = delays.size
    denominators = np.array([t[0] for t in terms], dtype=np.complex128)
    numerators = np.array([t[1] for t in terms], dtype=np.complex128).reshape(
        poles.size, C.shape[0], B.shape[1]
    )
    right = np.array([t[2] for t in terms], dtype=np.complex128).reshape(-1, n).T
    left = np.array([t[3] for t in terms], dtype=np.complex128).reshape(-1, n).T
    residues, undriven = residues_from_terms(numerators, denominators)
    return residues, D, undriven, {"right": right, "left": left}


def dss_to_pr(
    delays: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    *,
    mode: str = "eig",
    lf_max_seconds: float = 3.0,
    response_tolerance: float = 1e-8,
    # ─── EAI-only knobs (ignored for mode in {"eig","roots"}) ─────────────
    deflation_type: str = "fullDeflation",
    quality_threshold: float = 1e-10,
    maximum_iterations: int = 50,
    refinement_tol: float = 1e-12,
    reject_unstable_poles: bool = False,
    svd_refine: bool = True,
    symmetrize: bool = True,
    verbose: bool = False,
    # ─── FLAMO model-construction (only used for mode="eai") ──────────────
    fs: float = 1.0,
    nfft: int = 2**16,
    dtype: Any = None,
    device: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Modal decomposition of a delay state-space (DSS) system.

    ``H(z) = C·(diag(z^m_i) − A)^{-1}·B + D = Σ_k residue_k / (1 − p_k z^{-1}) + D``

    No ``build_to_pr`` counterpart exists: pass ``build.delays`` and
    ``build.A``/``B``/``C``/``D`` directly, plus ``fs=build.fs`` for
    ``mode="eai"``.

    Parameters
    ----------
    delays : array-like of int, shape ``(N,)``
        Per-line delays in samples.
    A : array-like ``(N, N)``
        Static feedback matrix.
    B : array-like ``(N, n_in)``
    C : array-like ``(n_out, N)``
    D : array-like ``(n_out, n_in)``
    mode : ``{"eig", "roots", "eai", "lindsey_fox"}``, default ``"eig"``
        Pole-extraction algorithm.

        - ``"eig"``   — eigenvalues of the minimal state-space realization.
        - ``"roots"`` — roots of the generalized characteristic polynomial.
        - ``"lindsey_fox"`` — experimental FFT grid search and Laguerre
          polishing of the same polynomial (real, nonsingular static A and
          positive integer delays). Refines against the original matrix and
          checks the all-port response. Falls back to EAI on failed completion
          or validation; raises if the fallback also fails validation. Does not change the
          default or guarantee a speedup. See :mod:`pyFDN.auxiliary.lindsey_fox`.
        - ``"eai"``   — Ehrlich-Aberth iteration in w = 1/z via FLAMO
          (delegates to :func:`pyFDN.translate.flamo_to_pr.flamo_to_pr`).
    lf_max_seconds : float, default 3.0
        LF root-search budget checked between numerical operations. Coefficient
        construction, mandatory matrix refinement, validation, and EAI fallback
        are outside this budget. The EAI-specific options configure fallback.
    response_tolerance : float, default 1e-8
        LF and its fallback must pass relative L2 and normalized peak checks
        against the original all-port transfer at deterministic off-grid points.
        This sampled check is not a uniform response error bound.

    Returns
    -------
    residues : ndarray ``(n_poles_reduced, n_out, n_in)``, complex
    poles : ndarray ``(n_poles_reduced,)``, complex
    direct : ndarray ``(n_out, n_in)``, complex
    is_conjugate : ndarray of bool ``(n_poles_reduced,)``
    meta : dict
        Lindsey-Fox reports ``solver``, ``complete``, ``responseValidation``, and
        ``fallbackReason`` when EAI was needed.
    """
    delays_arr = np.asarray(delays, dtype=int).ravel()
    if delays_arr.ndim != 1 or delays_arr.size == 0:
        raise ValueError("delays must be a non-empty 1-D array")

    a_mat = np.asarray(A, dtype=np.complex128)
    b_mat = np.asarray(B, dtype=np.complex128)
    c_mat = np.asarray(C, dtype=np.complex128)
    d_mat = np.asarray(D, dtype=np.complex128)

    n = delays_arr.size
    if a_mat.shape != (n, n):
        raise ValueError(f"A must have shape ({n},{n}), got {a_mat.shape}")

    mode_l = str(mode).lower()

    if mode_l == "eai":
        # Lazy: torch + FLAMO are imported only when EAI is actually requested,
        # keeping the eig/roots path numpy-only. Float64 by default for
        # machine-precision EAI (FLAMO's own default is float32).
        import torch

        from pyFDN.translate.dss_to_flamo import dss_to_flamo
        from pyFDN.translate.flamo_to_pr import flamo_to_pr

        if dtype is None:
            dtype = torch.float64
        model = dss_to_flamo(
            A=np.asarray(A, dtype=np.float64),
            B=np.asarray(B, dtype=np.float64),
            C=np.asarray(C, dtype=np.float64),
            D=np.asarray(D, dtype=np.float64),
            delays=delays_arr,
            fs=float(fs),
            nfft=int(nfft),
            device=device,
            shell=False,
            dtype=dtype,
        )
        return flamo_to_pr(
            model,
            deflation_type=deflation_type,
            reject_unstable_poles=reject_unstable_poles,
            quality_threshold=quality_threshold,
            maximum_iterations=maximum_iterations,
            refinement_tol=refinement_tol,
            svd_refine=svd_refine,
            symmetrize=symmetrize,
            verbose=verbose,
        )

    # eig / roots: pure-numpy pole finding, shared SVD residue extractor.
    a_poly = np.real_if_close(a_mat, tol=1000)
    if np.iscomplexobj(a_poly) and np.max(np.abs(np.imag(a_poly))) > 1e-12:
        raise ValueError(
            "roots mode currently requires a real-valued static feedback matrix."
        )
    a_poly = np.asarray(np.real(a_poly), dtype=float)

    if mode_l == "eig":
        aa, _, _, _ = dss_to_ss(delays_arr, a_mat)
        poles = np.linalg.eigvals(aa)
    elif mode_l == "lindsey_fox":
        raw_delays = np.asarray(delays)
        if np.any(delays_arr <= 0) or not np.array_equal(raw_delays, delays_arr):
            raise ValueError("lindsey_fox requires positive integer delays")
        singular_values = np.linalg.svd(a_poly, compute_uv=False)
        if singular_values[-1] <= np.finfo(float).eps * singular_values[0]:
            raise ValueError("lindsey_fox requires nonsingular feedback A")
        # At a pole, max_i |z|^m_i >= sigma_min(A) and
        # min_i |z|^m_i <= sigma_max(A). Both delay extremes cover |z| > 1.
        bounds = singular_values[[0, -1], None] ** (
            1.0 / delays_arr[[np.argmin(delays_arr), np.argmax(delays_arr)]]
        )
        if (
            not np.isfinite(lf_max_seconds)
            or lf_max_seconds <= 0
            or not np.isfinite(response_tolerance)
            or response_tolerance <= 0
        ):
            raise ValueError(
                "lf_max_seconds and response_tolerance must be positive and finite"
            )
        for matrix in (b_mat, c_mat, d_mat):
            if not np.all(np.isfinite(matrix)) or np.any(matrix.imag != 0):
                raise ValueError("lindsey_fox requires finite real B, C, and D")
        return _validated_lindsey_fox(
            delays_arr,
            a_poly,
            b_mat.real,
            c_mat.real,
            d_mat.real,
            bounds,
            lf_max_seconds,
            response_tolerance,
            {
                "deflation_type": deflation_type,
                "quality_threshold": quality_threshold,
                "maximum_iterations": maximum_iterations,
                "refinement_tol": refinement_tol,
                "reject_unstable_poles": reject_unstable_poles,
                "svd_refine": svd_refine,
                "symmetrize": symmetrize,
                "verbose": verbose,
                "fs": fs,
                "nfft": nfft,
                "dtype": dtype,
                "device": device,
            },
        )
    elif mode_l == "roots":
        # GCP p has p[k] = coef of z^{-k}; polynomial in w = z^{-1} is sum_k p[k] w^k
        p = general_char_poly(delays_arr, a_poly)
        w_roots = np.roots(p[::-1])  # np.roots expects high-to-low coefficient order
        poles = np.array(
            [1.0 / w for w in w_roots if np.abs(w) > 1e-14], dtype=np.complex128
        )
    else:
        raise ValueError("mode must be 'eig', 'roots', 'eai', or 'lindsey_fox'")

    poles, is_conjugate_pair, non_paired = reduce_conjugate_pairs(poles)
    residues, direct, undriven, eigenvectors = _dss_to_res(
        poles, delays_arr, a_mat, b_mat, c_mat, d_mat
    )

    meta_data: dict[str, Any] = {
        "nonPairedPoles": non_paired,
        "undrivenResidues": undriven,
        "eigenvectors": eigenvectors,
    }
    return residues, poles, direct, is_conjugate_pair, meta_data
