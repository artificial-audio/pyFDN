"""Modal decomposition for delay state-space (DSS) FDNs.

H(z) = C·(diag(z^m_i) − A)^{-1}·B + D = Σ_k residue_k/(1 − p_k z^{-1}) + D

This module is numpy-only. ``mode="eai"`` lazily imports torch + FLAMO, so
callers using only ``"eig"``/``"roots"`` never pay that import cost.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.math import general_char_poly
from pyFDN.auxiliary.poles import (
    reduce_conjugate_pairs,
    residue_at_pole,
    residues_from_terms,
)
from pyFDN.translate.dss_to_ss import dss_to_ss


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
    mode : ``{"eig", "roots", "eai"}``, default ``"eig"``
        Pole-extraction algorithm.

        - ``"eig"``   — eigenvalues of the minimal state-space realization.
        - ``"roots"`` — roots of the generalized characteristic polynomial.
        - ``"eai"``   — Ehrlich-Aberth iteration in w = 1/z via FLAMO
          (delegates to :func:`pyFDN.translate.flamo_to_pr.flamo_to_pr`).

    Returns
    -------
    residues : ndarray ``(n_poles_reduced, n_out, n_in)``, complex
    poles : ndarray ``(n_poles_reduced,)``, complex
    direct : ndarray ``(n_out, n_in)``, complex
    is_conjugate : ndarray of bool ``(n_poles_reduced,)``
    meta : dict
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
    elif mode_l == "roots":
        # GCP p has p[k] = coef of z^{-k}; polynomial in w = z^{-1} is sum_k p[k] w^k
        p = general_char_poly(delays_arr, a_poly)
        w_roots = np.roots(p[::-1])  # np.roots expects high-to-low coefficient order
        poles = np.array(
            [1.0 / w for w in w_roots if np.abs(w) > 1e-14], dtype=np.complex128
        )
    else:
        raise ValueError("mode must be 'eig', 'roots', or 'eai'")

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
