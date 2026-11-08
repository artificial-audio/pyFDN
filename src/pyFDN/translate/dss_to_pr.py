"""Modal decomposition for delay state-space (DSS) FDNs.

H(z) = C·(diag(z^m_i) − A)^{-1}·B + D = Σ_k residue_k/(z − p_k) + D

All three algorithms here are NumPy-only; the Ehrlich-Aberth iteration,
which needs torch and FLAMO, is :func:`pyFDN.flamo_to_pr`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.math import general_char_poly
from pyFDN.auxiliary.modal import modal_result_metadata
from pyFDN.auxiliary.poles import (
    assemble_residues,
    reduce_conjugate_pairs,
    residue_at_pole,
)
from pyFDN.translate.dss_to_ss import dss_to_ss
from pyFDN.translate.pr_to_impz import PoleResidue


def _validated_lindsey_fox(
    delays: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    bounds: np.ndarray,
    max_seconds: float,
    response_tolerance: float,
) -> tuple[PoleResidue, dict[str, Any]]:
    from pyFDN.auxiliary.lindsey_fox import lindsey_fox_roots
    from pyFDN.auxiliary.modal import (
        RankOneResidues,
        refine_dss_poles,
        validate_modal_response,
    )

    order = int(delays.sum())
    identity = np.eye(delays.size)

    p = general_char_poly(delays, a)
    poles = lindsey_fox_roots(
        p[::-1], float(bounds.min()), float(bounds.max()), max_seconds=max_seconds
    )
    poles, pairs, unpaired = reduce_conjugate_pairs(poles, strict=True)
    # Real coefficients make the omitted conjugate refinements redundant.
    poles = refine_dss_poles(poles, delays, a)
    residues, direct, undriven, vectors = _dss_to_res(poles, delays, a, b, c, d)
    meta = {
        **modal_result_metadata(pairs, order, "lindsey_fox"),
        "nonPairedPoles": unpaired,
        "undrivenResidues": undriven,
        "eigenvectors": vectors,
    }
    # Validate all internal ports, even when B/C hide missing modes.
    output_vectors = vectors["right"].T
    input_rows = vectors["left"].conj().T
    internal = RankOneResidues(output_vectors, input_rows, meta["undrivenResidues"])
    expected_residues = np.einsum(
        "ko,ki,k->koi", output_vectors @ c.T, input_rows @ b, internal.weights
    )
    if not np.allclose(
        residues,
        expected_residues,
        rtol=response_tolerance,
        atol=response_tolerance
        * max(float(np.max(np.abs(expected_residues))), np.finfo(float).tiny),
    ):
        raise RuntimeError("Modal residues disagree with the all-port decomposition")
    check = validate_modal_response(
        poles,
        internal,
        pairs,
        lambda z: np.linalg.solve(np.diag(z**delays) - a, identity),
        order,
        response_tolerance,
        static_matrix=a,
        delays=delays,
    )
    meta["responseValidation"] = check
    return PoleResidue(residues, poles, direct, pairs), meta


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
    residues, undriven, vectors = assemble_residues(terms, n, C.shape[0], B.shape[1])
    return residues, D, undriven, vectors


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
) -> tuple[PoleResidue, dict[str, Any]]:
    """Modal decomposition of a delay state-space (DSS) system.

    ``H(z) = C·(diag(z^m_i) − A)^{-1}·B + D = Σ_k residue_k / (z − p_k) + D``

    A :class:`~pyFDN.DSS` unpacks straight in, and the result straight into
    :func:`pyFDN.pr_to_impz`::

        pr, info = pyFDN.dss_to_pr(*dss)
        ir = pyFDN.pr_to_impz(*pr, ir_len)

    The Ehrlich-Aberth iteration works on a FLAMO model, so it is
    :func:`pyFDN.flamo_to_pr`'s: ``flamo_to_pr(dss_to_flamo(*dss, fs))``.

    Parameters
    ----------
    delays : array-like of int, shape ``(N,)``
        Per-line delays in samples.
    A : array-like ``(N, N)``
        Static feedback matrix.
    B : array-like ``(N, n_in)``
    C : array-like ``(n_out, N)``
    D : array-like ``(n_out, n_in)``
    mode : ``{"eig", "roots", "lindsey_fox"}``, default ``"eig"``
        Pole-extraction algorithm.

        - ``"eig"``   — eigenvalues of the minimal state-space realization.
        - ``"roots"`` — roots of the generalized characteristic polynomial.
        - ``"lindsey_fox"`` — experimental FFT grid search and Laguerre
          polishing of the same polynomial (real, nonsingular static A and
          positive integer delays). Refines against the original matrix and
          checks the all-port response. Failed completion or validation raises;
          no other solver is called. Does not change the
          default or guarantee a speedup. See :mod:`pyFDN.auxiliary.lindsey_fox`.
    lf_max_seconds : float, default 3.0
        LF root-search budget checked between numerical operations. Coefficient
        construction, mandatory matrix refinement, and validation are outside
        this budget.
    response_tolerance : float, default 1e-8
        LF must pass relative L2 and normalized peak checks
        against the original all-port transfer at deterministic off-grid points.
        Supported static loops can certify upper bounds on those same errors;
        otherwise the transfer is evaluated directly. Metadata ``errorBound``
        distinguishes bounds from measured errors. The sampled acceptance
        criterion is not a uniform response error guarantee. LF always
        requires a complete result.

    Returns
    -------
    pr : PoleResidue
        ``residues`` ``(n_poles, n_out, n_in)``, ``poles`` ``(n_poles,)``,
        ``direct`` ``(n_out, n_in)`` and ``is_conjugate_pole_pair``
        ``(n_poles,)``, each conjugate pair stored once.
    info : dict
        All modes report ``solver``, ``complete``, and expected/returned pole
        counts. ``responseValidation`` is None when no response check was run
        (complete eig/roots results). LF always validates its response;
        LF never switches to another solver.
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

    if mode == "lindsey_fox":
        if not np.all(np.isfinite(a_mat)) or np.any(a_mat.imag != 0):
            raise ValueError("lindsey_fox requires finite real A")
        if b_mat.ndim != 2 or b_mat.shape[0] != n or b_mat.shape[1] == 0:
            raise ValueError(f"B must have shape ({n}, n_in) with n_in > 0")
        if c_mat.ndim != 2 or c_mat.shape[1] != n or c_mat.shape[0] == 0:
            raise ValueError(f"C must have shape (n_out, {n}) with n_out > 0")
        if d_mat.shape != (c_mat.shape[0], b_mat.shape[1]):
            raise ValueError("D must have shape (n_out, n_in) matching C and B")
        for matrix in (b_mat, c_mat, d_mat):
            if not np.all(np.isfinite(matrix)) or np.any(matrix.imag != 0):
                raise ValueError("lindsey_fox requires finite real B, C, and D")

    # eig / roots: pure-numpy pole finding, shared SVD residue extractor.
    a_poly = np.real_if_close(a_mat, tol=1000)
    if np.iscomplexobj(a_poly) and np.max(np.abs(np.imag(a_poly))) > 1e-12:
        raise ValueError(
            "roots mode currently requires a real-valued static feedback matrix."
        )
    a_poly = np.asarray(np.real(a_poly), dtype=float)

    if mode == "eig":
        # the poles are the eigenvalues of the expanded A; no inputs or outputs
        aa = dss_to_ss(
            delays_arr, a_mat, np.zeros((n, 0)), np.zeros((0, n)), np.zeros((0, 0))
        )[0]
        poles = np.linalg.eigvals(aa)
    elif mode == "lindsey_fox":
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
        return _validated_lindsey_fox(
            delays_arr,
            a_poly,
            b_mat.real,
            c_mat.real,
            d_mat.real,
            bounds,
            lf_max_seconds,
            response_tolerance,
        )
    elif mode == "roots":
        # The w coefficients are the descending z coefficients, including zero poles.
        p = general_char_poly(delays_arr, a_poly)
        poles = np.roots(p)

    else:
        raise ValueError(
            f"mode must be 'eig', 'roots' or 'lindsey_fox', got {mode!r}; the "
            "Ehrlich-Aberth iteration is flamo_to_pr(dss_to_flamo(*dss, fs))"
        )

    poles, is_conjugate_pair, non_paired = reduce_conjugate_pairs(poles)
    residues, direct, undriven, eigenvectors = _dss_to_res(
        poles, delays_arr, a_mat, b_mat, c_mat, d_mat
    )

    meta_data: dict[str, Any] = {
        **modal_result_metadata(is_conjugate_pair, int(delays_arr.sum()), mode),
        "nonPairedPoles": non_paired,
        "undrivenResidues": undriven,
        "eigenvectors": eigenvectors,
    }
    return PoleResidue(residues, poles, direct, is_conjugate_pair), meta_data
