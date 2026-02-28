"""
Two-channel delay feedback: analytic vs FLAMO.

System: 2x2 FDN with delays m1, m2 and feedback matrix A.
Normal convention: P(z) = I - A @ diag(z^{-m1}, z^{-m2}) (delays on columns).
FLAMO uses the same convention (normal P(z) and dP/dz). We compare
det(P), (d/dz) log det P, and EAI poles (EAI in w-plane).

Requires: pyFDN and FLAMO (e.g. local flamo-probe).
"""

from __future__ import annotations

import numpy as np

import pyFDN


# -----------------------------------------------------------------------------
# Analytic formulas (2x2: delays m1, m2, feedback matrix A)
# -----------------------------------------------------------------------------


def analytic_P(z: complex, A: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Characteristic 2x2: P(z) = I - A @ diag(z^{-m1}, z^{-m2})."""
    m1, m2 = int(m[0]), int(m[1])
    z1 = z ** (-m1)
    z2 = z ** (-m2)
    P = np.array(
        [
            [1.0 - A[0, 0] * z1, -A[0, 1] * z2],
            [-A[1, 0] * z1, 1.0 - A[1, 1] * z2],
        ],
        dtype=np.complex128,
    )
    return P


def analytic_dP_dz(z: complex, A: np.ndarray, m: np.ndarray) -> np.ndarray:
    """dP/dz for P(z) = I - A @ diag(z^{-m1}, z^{-m2})."""
    m1, m2 = int(m[0]), int(m[1])
    z1 = z ** (-m1)
    z2 = z ** (-m2)
    # d(z^{-k})/dz = -k z^{-k-1}
    dP = np.array(
        [
            [A[0, 0] * m1 * (z ** (-m1 - 1)), A[0, 1] * m2 * (z ** (-m2 - 1))],
            [A[1, 0] * m1 * (z ** (-m1 - 1)), A[1, 1] * m2 * (z ** (-m2 - 1))],
        ],
        dtype=np.complex128,
    )
    return dP


def analytic_log_det_derivative(z: complex, A: np.ndarray, m: np.ndarray) -> complex:
    """(d/dz) log det P(z) = (d/dz)(det P) / det P. Same as trace(P^{-1} dP/dz)."""
    P = analytic_P(z, A, m)
    dP = analytic_dP_dz(z, A, m)
    detP = P[0, 0] * P[1, 1] - P[0, 1] * P[1, 0]
    if np.abs(detP) < 1e-20:
        return np.inf + 0.0j
    # (d/dz)(det P) = dP_11*P_22 + P_11*dP_22 - dP_12*P_21 - P_12*dP_21
    d_det = dP[0, 0] * P[1, 1] + P[0, 0] * dP[1, 1] - dP[0, 1] * P[1, 0] - P[0, 1] * dP[1, 0]
    # return d_det / detP
    return np.trace(np.linalg.solve(P, dP))


def gcp_poly_at_z(z: complex, p: np.ndarray) -> complex:
    """Evaluate GCP Q(w)=sum_k p[k]w^k at w=z^{-1}. Returns Q(1/z) = det(P(z))."""
    w = 1.0 / z
    return np.polyval(p[::-1], w)  # polyval expects high-to-low: p[L]*w^L + ... + p[0]


def gcp_log_derivative_at_z(z: complex, p: np.ndarray) -> complex:
    """(d/dz) log Q(1/z) = (1/Q)*dQ/dz. With w=1/z: dQ/dz = Q'(w)*(-1/z^2), so = -(1/z^2)*Q'(w)/Q(w)."""
    w = 1.0 / z
    q = np.polyval(p[::-1], w)
    if np.abs(q) < 1e-20:
        return np.inf + 0.0j
    # Q'(w) = polyder of p; polyval(polyder(coeffs), w)
    coeffs = p[::-1]  # high to low: coeffs[0]*w^L + ... + coeffs[L]
    deriv = np.polyder(coeffs)
    qp = np.polyval(deriv, w)
    return -(1.0 / (z * z)) * (qp / q)


def true_poles_two_channel(A: np.ndarray, m: np.ndarray) -> np.ndarray:
    """True poles from det(P(z)) = 0 using library GCP. p[k] = coef of z^{-k}; roots in w=z^{-1} then z=1/w."""
    p = pyFDN.general_char_poly(m, A)
    # Polynomial in w = z^{-1}: sum_k p[k] w^k = 0. np.roots expects high-to-low.
    w_roots = np.roots(p[::-1])
    z_poles = []
    for w in w_roots:
        if np.abs(w) < 1e-14:
            continue
        z_poles.append(1.0 / w)
    return np.array(z_poles, dtype=np.complex128)


def _deflation_w_from_z(z_i: complex, poles_z: np.ndarray, i: int) -> complex:
    """Deflation in w-plane: sum_{j!=i} 1/(w_i - w_j), with w = 1/z."""
    w_i = 1.0 / z_i
    defl = 0.0 + 0.0j
    for j in range(len(poles_z)):
        if j == i:
            continue
        w_j = 1.0 / poles_z[j]
        diff = w_i - w_j
        if np.abs(diff) < 1e-14:
            continue
        defl += 1.0 / diff
    return defl


def eai_refine_analytic_two_channel(
    poles_init: np.ndarray,
    A: np.ndarray,
    m: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-10,
    verbose: bool = False,
    sequential: bool = True,
    sort_every_step: bool = False,
    max_step: float | None = 0.2,
    use_z_step: bool = False,
) -> np.ndarray:
    """
    Ehrlich-Aberth for 2x2. By default uses the same step as GCP EAI: (d/dz) log det P
    is converted to w-plane (Q'/Q = -z^2 (d/dz) log det P), step in w, then z_new = 1/w_new.
    So analytic and GCP EAI coincide. If use_z_step=True, uses z_new = z - 1/((d/dz)log det P - defl_z)
    with defl_z = sum 1/(z_i - z_j) (can differ from GCP and may not converge as well).
    """
    poles = np.asarray(poles_init, dtype=np.complex128).ravel().copy()
    n = len(poles)
    order = np.argsort(np.angle(poles))
    poles = poles[order]
    for step in range(max_iter):
        if sort_every_step:
            order = np.argsort(np.angle(poles))
            poles = poles[order]
        poles_old = poles.copy()
        for i in range(n):
            z_i = poles[i]
            inv_newton_z = analytic_log_det_derivative(z_i, A, m)
            if not np.isfinite(inv_newton_z):
                continue
            if use_z_step:
                # Original z-plane step (can diverge): z_new = z - 1/(inv_newton - defl_z)
                deflation_z = 0.0 + 0.0j
                for j in range(n):
                    if j != i:
                        diff = z_i - poles[j]
                        if np.abs(diff) < 1e-14:
                            continue
                        deflation_z += 1.0 / diff
                denom = inv_newton_z - deflation_z
                if not np.isfinite(denom) or np.abs(denom) < 1e-20:
                    continue
                step_val = 1.0 / denom
                if max_step is not None and np.abs(step_val) > max_step:
                    step_val = step_val * (max_step / np.abs(step_val))
                poles[i] = z_i - step_val
            else:
                # Same step as GCP: (Q'/Q)(w) = -z^2 (d/dz) log det P, defl_w = sum 1/(w_i - w_j)
                qp_over_q = -(z_i * z_i) * inv_newton_z
                defl_w = _deflation_w_from_z(z_i, poles, i)
                D_w = qp_over_q - defl_w
                if not np.isfinite(D_w) or np.abs(D_w) < 1e-20:
                    continue
                w_i = 1.0 / z_i
                w_new = w_i - 1.0 / D_w
                if np.abs(w_new) < 1e-14:
                    continue
                poles[i] = 1.0 / w_new
        err = np.max(np.abs(poles - poles_old))
        if verbose:
            print(f"  step {step + 1}: max |Δz| = {err:.3e}")
        if err < tol:
            break
    return poles


def eai_refine_gcp(
    poles_init_z: np.ndarray,
    p: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-10,
    verbose: bool = False,
    sequential: bool = True,
    sort_every_step: bool = False,
) -> np.ndarray:
    """
    Ehrlich-Aberth in the w-plane (w = z^{-1}) using the GCP polynomial Q(w).
    inv_newton = Q'(w)/Q(w) = (d/dw) log Q(w). Deflation: sum_{j!=i} 1/(w_i - w_j).
    Returns poles in z (z = 1/w).
    """
    # Work in w = 1/z; initial w on unit circle (same angles as z)
    w_poles = 1.0 / np.asarray(poles_init_z, dtype=np.complex128).ravel()
    n = len(w_poles)
    order = np.argsort(np.angle(w_poles))
    w_poles = w_poles[order]
    coeffs = p[::-1]  # Q(w) = coeffs[0]*w^L + ... + coeffs[L]
    coeffs_der = np.polyder(coeffs)

    for step in range(max_iter):
        if sort_every_step:
            order = np.argsort(np.angle(w_poles))
            w_poles = w_poles[order]
        w_old = w_poles.copy()
        for i in range(n):
            w_i = w_poles[i]
            q = np.polyval(coeffs, w_i)
            qp = np.polyval(coeffs_der, w_i)
            if np.abs(q) < 1e-20:
                continue
            inv_newton = qp / q  # (d/dw) log Q(w)
            deflation = 0.0 + 0.0j
            for j in range(n):
                if j != i:
                    diff = w_i - w_poles[j]
                    if np.abs(diff) < 1e-14:
                        continue
                    deflation += 1.0 / diff
            denom = inv_newton - deflation
            if not np.isfinite(denom) or np.abs(denom) < 1e-20:
                continue
            w_poles[i] = w_i - 1.0 / denom
        err = np.max(np.abs(w_poles - w_old))
        if verbose:
            print(f"  step {step + 1}: max |Δw| = {err:.3e}")
        if err < tol:
            break
    z_poles = 1.0 / w_poles
    return z_poles


def sort_by_angle(p: np.ndarray) -> np.ndarray:
    return p[np.argsort(np.angle(p))]


def max_error_nearest_matching(poles_a: np.ndarray, poles_b: np.ndarray) -> float:
    """Max distance when each pole in poles_a is matched to nearest in poles_b (same size)."""
    a = np.asarray(poles_a, dtype=np.complex128).ravel()
    b = np.asarray(poles_b, dtype=np.complex128).ravel()
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]
    # For each a[i], min_j |a[i] - b[j]|
    dists = np.min(np.abs(a[:, None] - b[None, :]), axis=1)
    return float(np.max(dists))


def main() -> None:
    # Two-channel: delays m1, m2 and 2x2 feedback A (real, stable)
    m = np.array([3, 5], dtype=int)
    # Simple orthogonal-like 2x2: eigenvalues < 1 in magnitude
    a, b = 0.5, 0.4
    A = np.array([[a, b], [-b, a]], dtype=np.float64)
    n_poles = int(np.sum(m))

    print("Two-channel delay feedback: P(z) = I - A @ diag(z^{-m1}, z^{-m2})")
    print(f"  m = {m},  A = [[{A[0,0]}, {A[0,1]}], [{A[1,0]}, {A[1,1]}]]")
    print(f"  number of poles = {n_poles}")
    print()

    # -------------------------------------------------------------------------
    # 1) True poles (polynomial roots) and verify analytic formula
    # -------------------------------------------------------------------------
    p_gcp = pyFDN.general_char_poly(m, A)
    poles_true = true_poles_two_channel(A, m)
    poles_true = sort_by_angle(poles_true)
    poles_true = poles_true[np.abs(poles_true) < 1e6]
    poles_true = sort_by_angle(poles_true)
    n_true = len(poles_true)
    if n_true != n_poles:
        print(f"   [Note] Got {n_true} finite poles from polynomial (degree {n_poles})")
    print("1) True poles (from det(P)=0):")
    for k, z in enumerate(poles_true[: min(10, n_true)]):
        print(f"   z_{k} = {z:.6f}  (|z| = {np.abs(z):.6f})")
    if n_true > 10:
        print("   ...")

    # 1b) Verify: at true poles det(P) and GCP are ~0; at sample z, (d/dz)log det P matches GCP
    max_residual = max(np.abs(gcp_poly_at_z(z, p_gcp)) for z in poles_true)
    print(f"   max |GCP(z)| at true poles = {max_residual:.2e}  (should be ~0)")
    z_sample = 0.7 + 0.2j
    d_ana = analytic_log_det_derivative(z_sample, A, m)
    d_gcp = gcp_log_derivative_at_z(z_sample, p_gcp)
    print(f"   At z = {z_sample}: (d/dz)log det P: analytic = {d_ana:.6g}, GCP = {d_gcp:.6g}, |Δ| = {abs(d_ana - d_gcp):.2e}")
    print()

    # -------------------------------------------------------------------------
    # 2) Build FLAMO model (2x2)
    # -------------------------------------------------------------------------
    delays = m
    B = np.eye(2, dtype=np.float64)
    C = np.eye(2, dtype=np.float64)
    D = np.zeros((2, 2), dtype=np.float64)

    model = pyFDN.dss_to_flamo(
        A=A, B=B, C=C, D=D, m=delays, Fs=1.0, nfft=256, shell=False
    )
    core = model.get_core() if callable(getattr(model, "get_core", None)) else model
    recursion = list(core.branchA)[1]
    decomposition = pyFDN.flamo_extract_pr_decomposition(
        model, delays, recursion_module=recursion
    )
    p_probe = decomposition["P"]

    # -------------------------------------------------------------------------
    # 3) Compare det(P), (d/dz) log det P at sample points
    # -------------------------------------------------------------------------
    # FLAMO and analytic both use normal P(z) = I - A*D (delays on columns).
    test_z = [0.5 + 0.3j, 1.2 - 0.4j, 0.8 * np.exp(0.5j)]
    print("2) Compare det(P) and (d/dz) log det P at sample z (FLAMO vs analytic):")
    for z in test_z:
        P_ana = analytic_P(z, A, m)
        P_flamo = np.asarray(p_probe.at(z)).reshape(2, 2)
        det_ana = np.linalg.det(P_ana)
        det_flamo = np.linalg.det(P_flamo)
        d_ana = analytic_log_det_derivative(z, A, m)
        d_flamo = p_probe.log_det_derivative(z)
        err_det = abs(det_ana - det_flamo)
        err_d = abs(d_ana - d_flamo)
        print(f"   z = {z}")
        print(f"      |det(P)_ana - det(P)_flamo| = {err_det:.2e},  |(d/dz)ana - (d/dz)flamo| = {err_d:.2e}")
    print()

    # -------------------------------------------------------------------------
    # 4) EAI with analytic (d/dz) log det P -> compare to true poles
    # -------------------------------------------------------------------------
    # Initial poles: same count as true poles, on unit circle
    pole_angles = np.linspace(0.0, 2.0 * np.pi, n_true, endpoint=False)
    poles_init = np.exp(1j * pole_angles)
    poles_init = sort_by_angle(poles_init)

    # One-step diagnostic (first pole only): analytic vs GCP must give same z_new
    z0 = poles_init[0]
    inv_z = analytic_log_det_derivative(z0, A, m)
    qp_q = -(z0 * z0) * inv_z
    defl_w0 = _deflation_w_from_z(z0, poles_init, 0)
    D_w = qp_q - defl_w0
    w0 = 1.0 / z0
    w_new_ana = w0 - 1.0 / D_w
    z_new_ana = 1.0 / w_new_ana if np.abs(w_new_ana) > 1e-14 else z0
    coeffs = p_gcp[::-1]
    q0 = np.polyval(coeffs, w0)
    qp0 = np.polyval(np.polyder(coeffs), w0)
    D_w_gcp = (qp0 / q0) - defl_w0
    w_new_gcp = w0 - 1.0 / D_w_gcp
    z_new_gcp = 1.0 / w_new_gcp if np.abs(w_new_gcp) > 1e-14 else z0
    print(f"3a) One step (first pole): |z_new_analytic - z_new_GCP| = {abs(z_new_ana - z_new_gcp):.3e}")
    print()

    # Analytic EAI with same step as GCP (use_z_step=False): (d/dz)log det P -> w-plane step -> z_new
    poles_analytic_eai = eai_refine_analytic_two_channel(
        poles_init, A, m, max_iter=200, tol=1e-12, verbose=False, use_z_step=False
    )
    poles_analytic_eai_sorted = sort_by_angle(poles_analytic_eai)
    err_eai_analytic = max_error_nearest_matching(poles_analytic_eai, poles_true)
    print("3) EAI with analytic (d/dz) log det P, step = GCP-equivalent (w-plane then z=1/w):")
    print(f"   max distance (nearest-pole match) to true = {err_eai_analytic:.3e}")
    print()

    # -------------------------------------------------------------------------
    # 4) EAI with GCP (w-plane, Q(w), Newton + deflation)
    # -------------------------------------------------------------------------
    poles_gcp_eai = eai_refine_gcp(
        poles_init, p_gcp, max_iter=200, tol=1e-12, verbose=False
    )
    poles_gcp_eai_sorted = sort_by_angle(poles_gcp_eai)
    err_gcp_eai = max_error_nearest_matching(poles_gcp_eai, poles_true)
    err_analytic_vs_gcp = max_error_nearest_matching(poles_analytic_eai, poles_gcp_eai)
    print("4) EAI with GCP (w-plane, Q(w), Newton + deflation):")
    print(f"   max distance (nearest-pole match) to true = {err_gcp_eai:.3e}")
    print(f"   max distance analytic EAI vs GCP EAI     = {err_analytic_vs_gcp:.3e}")
    print()

    n_compare = min(n_true, len(poles_analytic_eai_sorted), len(poles_gcp_eai_sorted))
    print("{:<22} {:<22} {:<22}".format("Analytic EAI", "GCP EAI", "True poles"))
    for i in range(min(n_compare, 10)):
        print(
            "{:<22.6g} {:<22.6g} {:<22.6g}".format(
                poles_analytic_eai_sorted[i],
                poles_gcp_eai_sorted[i],
                poles_true[i],
            )
        )
    if n_compare > 10:
        print("   ...")
    print()

    # -------------------------------------------------------------------------
    # 5) FLAMO flamo_to_pr with refinement_tol (run to convergence)
    # -------------------------------------------------------------------------
    residues_flamo, poles_flamo, direct_flamo, is_pair, meta = pyFDN.flamo_to_pr(
        delays=delays,
        decomposition=decomposition,
        feedback_delay_units=0,
        absorption_delay_units=0,
        maximum_iterations=200,
        reject_unstable_poles=False,
        deflation_type="fullDeflation",
        quality_threshold=1e-15,
        refinement_tol=1e-12,
        verbose=False,
    )
    refined_flamo = np.asarray(meta["refinedPoles"], dtype=np.complex128).ravel()
    refined_flamo_sorted = sort_by_angle(refined_flamo)
    n_flamo = len(refined_flamo_sorted)

    err_flamo_true = max_error_nearest_matching(refined_flamo, poles_true)
    err_flamo_analytic = max_error_nearest_matching(refined_flamo, poles_analytic_eai)
    err_flamo_gcp = max_error_nearest_matching(refined_flamo, poles_gcp_eai)
    print("5) FLAMO refined (refinement_tol=1e-12):")
    print(f"   FLAMO iterations = {meta.get('iterations', 'N/A')}")
    print(f"   max dist (nearest) refined_FLAMO vs true   = {err_flamo_true:.3e}")
    print(f"   max dist (nearest) refined_FLAMO vs analytic EAI = {err_flamo_analytic:.3e}")
    print(f"   max dist (nearest) refined_FLAMO vs GCP EAI      = {err_flamo_gcp:.3e}")
    print()

    n_compare_all = min(n_true, len(poles_analytic_eai_sorted), len(poles_gcp_eai_sorted), n_flamo)
    print("{:<18} {:<18} {:<18} {:<18}".format("FLAMO refined", "Analytic EAI", "GCP EAI", "True poles"))
    for i in range(min(n_compare_all, 10)):
        print(
            "{:<18.6g} {:<18.6g} {:<18.6g} {:<18.6g}".format(
                refined_flamo_sorted[i],
                poles_analytic_eai_sorted[i],
                poles_gcp_eai_sorted[i],
                poles_true[i],
            )
        )
    if n_compare_all > 10:
        print("   ...")
    print()

    # assert err_flamo_analytic < 1e-5, "FLAMO refined should match analytic EAI"
    # assert err_flamo_gcp < 1e-5, "FLAMO refined should match GCP EAI"
    # assert err_flamo_true < 1e-5, "FLAMO refined should match true poles"
    # print("Two-channel comparison passed (analytic, GCP, FLAMO, true poles).")


if __name__ == "__main__":
    main()
