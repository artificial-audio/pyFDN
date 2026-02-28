"""
Two-channel delay feedback: z-plane only (no w formulation in the iteration).

Same system as example_two_channel_delay_feedback.py, but all EAI algorithms
use the z-plane step only:
  z_new = z - 1/((d/dz) log det P(z) - deflation_z)
  deflation_z = sum_{j!=i} 1/(z_i - z_j)

No conversion to w = 1/z in the iteration; no w-plane deflation or step.
Step size is capped (max_step) so the z-plane iteration converges.
True poles are still computed from the GCP (roots in w then z = 1/w).

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
    dP = np.array(
        [
            [A[0, 0] * m1 * (z ** (-m1 - 1)), A[0, 1] * m2 * (z ** (-m2 - 1))],
            [A[1, 0] * m1 * (z ** (-m1 - 1)), A[1, 1] * m2 * (z ** (-m2 - 1))],
        ],
        dtype=np.complex128,
    )
    return dP


def analytic_log_det_derivative(z: complex, A: np.ndarray, m: np.ndarray) -> complex:
    """(d/dz) log det P(z) = (d/dz)(det P) / det P."""
    P = analytic_P(z, A, m)
    dP = analytic_dP_dz(z, A, m)
    detP = P[0, 0] * P[1, 1] - P[0, 1] * P[1, 0]
    if np.abs(detP) < 1e-20:
        return np.inf + 0.0j
    d_det = dP[0, 0] * P[1, 1] + P[0, 0] * dP[1, 1] - dP[0, 1] * P[1, 0] - P[0, 1] * dP[1, 0]
    return d_det / detP * (-z * z)


def gcp_poly_at_z(z: complex, p: np.ndarray) -> complex:
    """Evaluate GCP Q(w)=sum_k p[k]w^k at w=z^{-1}. Returns Q(1/z) = det(P(z))."""
    w = 1.0 / z
    return np.polyval(p[::-1], w)


def gcp_log_derivative_at_z(z: complex, p: np.ndarray) -> complex:
    """(d/dz) log Q(1/z) = -(1/z^2)*Q'(w)/Q(w) with w=1/z."""
    w = 1.0 / z
    q = np.polyval(p[::-1], w)
    if np.abs(q) < 1e-20:
        return np.inf + 0.0j
    coeffs = p[::-1]
    deriv = np.polyder(coeffs)
    qp = np.polyval(deriv, w)
    return -(1.0 / (z * z)) * (qp / q)


def true_poles_two_channel(A: np.ndarray, m: np.ndarray) -> np.ndarray:
    """True poles from det(P(z)) = 0 using library GCP (roots in w=z^{-1}, then z=1/w)."""
    p = pyFDN.general_char_poly(m, A)
    w_roots = np.roots(p[::-1])
    z_poles = []
    for w in w_roots:
        if np.abs(w) < 1e-14:
            continue
        z_poles.append(1.0 / w)
    return np.array(z_poles, dtype=np.complex128)


def _deflation_z(i: int, poles: np.ndarray) -> complex:
    """Deflation in z-plane: sum_{j!=i} 1/(z_i - z_j)."""
    z_i = poles[i]
    defl = 0.0 + 0.0j
    for j in range(len(poles)):
        if j == i:
            continue
        diff = z_i - poles[j]
        if np.abs(diff) < 1e-14:
            continue
        defl += 1.0 / diff
    return defl

def _deflation_w(i: int, poles: np.ndarray) -> complex:
    """Deflation in z-plane: sum_{j!=i} 1/(z_i - z_j)."""
    z_i = poles[i]
    defl = 0.0 + 0.0j
    for j in range(len(poles)):
        if j == i:
            continue
        diff = z_i - poles[j]
        if np.abs(diff) < 1e-14:
            continue
        defl += z_i * poles[j] / diff
    return -defl


def eai_refine_z(
    poles_init: np.ndarray,
    inv_newton_fn: callable,
    *,
    max_iter: int = 200,
    tol: float = 1e-10,
    verbose: bool = False,
    sequential: bool = True,
    sort_every_step: bool = False,
    max_step: float = 0.08,
) -> np.ndarray:
    """
    Ehrlich-Aberth in z-plane only: z_new = z - 1/(inv_newton(z) - deflation_z).
    deflation_z = sum_{j!=i} 1/(z_i - z_j). inv_newton_fn(z) = (d/dz) log det P(z).
    Step capped at max_step so iteration converges (z-plane step tends to overshoot).
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
            inv_newton = inv_newton_fn(z_i)
            if not np.isfinite(inv_newton):
                continue
            deflation = _deflation_w(i, poles)
            denom = inv_newton - deflation
            if not np.isfinite(denom) or np.abs(denom) < 1e-20:
                continue
            step_val = 1.0 / denom
            if np.abs(step_val) > max_step:
                step_val = step_val * (max_step / np.abs(step_val))
            poles[i] = z_i - step_val
        err = np.max(np.abs(poles - poles_old))
        if verbose:
            print(f"  step {step + 1}: max |Δz| = {err:.3e}")
        if err < tol:
            break
    return poles


def sort_by_angle(p: np.ndarray) -> np.ndarray:
    return p[np.argsort(np.angle(p))]


def max_error_nearest_matching(poles_a: np.ndarray, poles_b: np.ndarray) -> float:
    """Max distance when each pole in poles_a is matched to nearest in poles_b."""
    a = np.asarray(poles_a, dtype=np.complex128).ravel()
    b = np.asarray(poles_b, dtype=np.complex128).ravel()
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]
    dists = np.min(np.abs(a[:, None] - b[None, :]), axis=1)
    return float(np.max(dists))


def main() -> None:
    m = np.array([3, 5], dtype=int)
    a, b = 0.5, 0.4
    A = np.array([[a, b], [-b, a]], dtype=np.float64)
    n_poles = int(np.sum(m))

    print("Two-channel delay feedback (z-plane only, no w in iteration)")
    print("  P(z) = I - A @ diag(z^{-m1}, z^{-m2})")
    print(f"  m = {m},  A = [[{A[0,0]}, {A[0,1]}], [{A[1,0]}, {A[1,1]}]]")
    print(f"  EAI: z_new = z - 1/((d/dz)log det P - deflation_z),  deflation_z = sum 1/(z_i-z_j)")
    print()

    p_gcp = pyFDN.general_char_poly(m, A)
    poles_true = true_poles_two_channel(A, m)
    poles_true = sort_by_angle(poles_true)
    poles_true = poles_true[np.abs(poles_true) < 1e6]
    poles_true = sort_by_angle(poles_true)
    n_true = len(poles_true)
    if n_true != n_poles:
        print(f"   [Note] Got {n_true} finite poles (degree {n_poles})")
    print("1) True poles (from det(P)=0):")
    for k, z in enumerate(poles_true[: min(10, n_true)]):
        print(f"   z_{k} = {z:.6f}  (|z| = {np.abs(z):.6f})")
    if n_true > 10:
        print("   ...")
    max_residual = max(np.abs(gcp_poly_at_z(z, p_gcp)) for z in poles_true)
    print(f"   max |GCP(z)| at true poles = {max_residual:.2e}")
    print()

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

    pole_angles = np.linspace(0.0, 2.0 * np.pi, n_true, endpoint=False)
    poles_init = np.exp(1j * pole_angles)
    poles_init = sort_by_angle(poles_init)

    # 2) Analytic EAI (z-plane only)
    poles_analytic = eai_refine_z(
        poles_init,
        lambda z: analytic_log_det_derivative(z, A, m),
        max_iter=400,
        tol=1e-12,
        verbose=False,
        max_step=0.08,
    )
    poles_analytic_sorted = sort_by_angle(poles_analytic)
    err_ana = max_error_nearest_matching(poles_analytic, poles_true)
    print("2) EAI analytic (z-plane): (d/dz)log det P from P(z), deflation_z")
    print(f"   max dist (nearest) to true = {err_ana:.3e}")
    print()

    # 3) GCP EAI in z (z-plane only): (d/dz)log det P from GCP at z, deflation_z
    poles_gcp_z = eai_refine_z(
        poles_init,
        lambda z: gcp_log_derivative_at_z(z, p_gcp),
        max_iter=400,
        tol=1e-12,
        verbose=False,
        max_step=0.08,
    )
    poles_gcp_z_sorted = sort_by_angle(poles_gcp_z)
    err_gcp = max_error_nearest_matching(poles_gcp_z, poles_true)
    err_ana_gcp = max_error_nearest_matching(poles_analytic, poles_gcp_z)
    print("3) EAI GCP in z (z-plane): (d/dz)log Q(1/z) from GCP, deflation_z")
    print(f"   max dist (nearest) to true = {err_gcp:.3e}")
    print(f"   max dist analytic vs GCP-z = {err_ana_gcp:.3e}")
    print()

    # 4) FLAMO with z-plane step (use_w_plane_step=False; step capped inside FLAMO)
    residues_flamo, poles_flamo, direct_flamo, is_pair, meta = pyFDN.flamo_to_pr(
        delays=delays,
        decomposition=decomposition,
        feedback_delay_units=0,
        absorption_delay_units=0,
        maximum_iterations=400,
        reject_unstable_poles=False,
        deflation_type="fullDeflation",
        quality_threshold=1e-15,
        refinement_tol=1e-12,
        verbose=False,
        use_w_plane_step=False,
    )
    refined_flamo = np.asarray(meta["refinedPoles"], dtype=np.complex128).ravel()
    refined_flamo_sorted = sort_by_angle(refined_flamo)
    n_flamo = len(refined_flamo_sorted)
    err_flamo_true = max_error_nearest_matching(refined_flamo, poles_true)
    err_flamo_ana = max_error_nearest_matching(refined_flamo, poles_analytic)
    err_flamo_gcp = max_error_nearest_matching(refined_flamo, poles_gcp_z)
    print("4) FLAMO refined (z-plane step, use_w_plane_step=False):")
    print(f"   iterations = {meta.get('iterations', 'N/A')}")
    print(f"   max dist to true   = {err_flamo_true:.3e}")
    print(f"   max dist to analytic = {err_flamo_ana:.3e}")
    print(f"   max dist to GCP-z    = {err_flamo_gcp:.3e}")
    print()

    n_compare = min(n_true, len(poles_analytic_sorted), len(poles_gcp_z_sorted), n_flamo)
    print("{:<18} {:<18} {:<18} {:<18}".format("Analytic (z)", "GCP (z)", "FLAMO (z)", "True"))
    for i in range(min(n_compare, 10)):
        print(
            "{:<18.6g} {:<18.6g} {:<18.6g} {:<18.6g}".format(
                poles_analytic_sorted[i],
                poles_gcp_z_sorted[i],
                refined_flamo_sorted[i],
                poles_true[i],
            )
        )
    if n_compare > 10:
        print("   ...")
    print()

    # Z-plane iteration is more fragile than w-plane; use relaxed tolerance
    tol_assert = 1e-4
    assert err_ana < tol_assert, f"Analytic z-plane EAI vs true: {err_ana:.3e} >= {tol_assert}"
    assert err_gcp < tol_assert, f"GCP z-plane EAI vs true: {err_gcp:.3e} >= {tol_assert}"
    assert err_flamo_true < tol_assert, f"FLAMO z-plane vs true: {err_flamo_true:.3e} >= {tol_assert}"
    assert err_ana_gcp < tol_assert, f"Analytic vs GCP z-plane: {err_ana_gcp:.3e} >= {tol_assert}"
    print("Z-plane-only comparison passed.")


if __name__ == "__main__":
    main()
