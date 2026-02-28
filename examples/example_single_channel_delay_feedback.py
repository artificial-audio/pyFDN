"""
Single-channel delay feedback: analytic vs FLAMO.

System: x[n] = a * x[n-m], so P(z) = 1 - a z^{-m}. Everything is computed
analytically and compared to the FLAMO Recursion (w = z^{-1} probe convention).
EAI is run with the analytic (d/dz) log det P (no delay term) and with the
FLAMO loop; both are compared to the true analytic poles.

Requires: pyFDN and FLAMO (e.g. local flamo-probe). The denominator in EAI
is (d/dz) log det P only; no delay term is used (w-convention in FLAMO).
"""

from __future__ import annotations

import numpy as np

import pyFDN


# -----------------------------------------------------------------------------
# Analytic formulas (single channel: delay m, feedback gain a)
# -----------------------------------------------------------------------------


def analytic_poles(a: float, m: int) -> np.ndarray:
    """True poles: 1 - a z^{-m} = 0 => z^m = a => z_k = a^{1/m} exp(2 pi i k / m)."""
    r = (a + 0.0) ** (1.0 / m)  # positive real root when a > 0
    k = np.arange(m, dtype=np.float64)
    return r * np.exp(2.0j * np.pi * k / m)


def analytic_P(z: complex, a: float, m: int) -> complex:
    """Characteristic (scalar): P(z) = 1 - a z^{-m}."""
    return 1.0 - a * (z ** (-m))


def analytic_log_det_derivative(z: complex, a: float, m: int) -> complex:
    """(d/dz) log det P(z) = (d/dz) log(1 - a z^{-m}) = a m z^{-m-1} / (1 - a z^{-m}). No delay term."""
    zm = z ** (-m)
    denom = 1.0 - a * zm
    if abs(denom) < 1e-15:
        return np.inf + 0.0j
    return (a * m * (z ** (-m - 1))) / denom


def eai_refine_analytic(
    poles_init: np.ndarray,
    a: float,
    m: int,
    *,
    max_iter: int = 200,
    tol: float = 1e-10,
    verbose: bool = False,
    sequential: bool = True,
    sort_every_step: bool = False,
    exact_iter: int | None = None,
) -> np.ndarray:
    """
    Ehrlich-Aberth iteration using analytic inv_newton (no delay term).
    Full deflation: deflation_i = sum_{j != i} 1 / (z_i - z_j).
    If sequential=True (default), update poles in place (matches FLAMO).
    If sort_every_step=False (default), sort by angle only once at start so
    pole-index mapping stays fixed (matches FLAMO with fullDeflation).
    If exact_iter is set, run exactly that many iterations (no tol early exit).
    """
    poles = np.asarray(poles_init, dtype=np.complex128).ravel().copy()
    n = len(poles)
    # Match FLAMO: sort once by angle at start; with fullDeflation FLAMO never re-sorts
    order = np.argsort(np.angle(poles))
    poles = poles[order]
    n_iters = exact_iter if exact_iter is not None else max_iter
    for step in range(n_iters):
        if sort_every_step:
            order = np.argsort(np.angle(poles))
            poles = poles[order]
        poles_old = poles.copy()
        if sequential:
            for i in range(n):
                z_i = poles[i]
                inv_newton = analytic_log_det_derivative(z_i, a, m)
                if not np.isfinite(inv_newton):
                    continue
                deflation = 0.0 + 0.0j
                for j in range(n):
                    if j != i:
                        deflation += 1.0 / (z_i - poles[j])
                denom = inv_newton - deflation
                if np.isfinite(denom) and abs(denom) > 1e-20:
                    poles[i] = z_i - 1.0 / denom
        else:
            for i in range(n):
                z_i = poles[i]
                inv_newton = analytic_log_det_derivative(z_i, a, m)
                if not np.isfinite(inv_newton):
                    continue
                deflation = 0.0 + 0.0j
                for j in range(n):
                    if j != i:
                        deflation += 1.0 / (z_i - poles_old[j])
                denom = inv_newton - deflation
                if np.isfinite(denom) and abs(denom) > 1e-20:
                    poles[i] = z_i - 1.0 / denom
        err = np.max(np.abs(poles - poles_old))
        if verbose:
            print(f"  step {step + 1}: max |Δz| = {err:.3e}")
        if exact_iter is None and err < tol:
            break
    return poles


def main() -> None:
    # Single-channel: delay m, feedback gain a (real, 0 < a < 1 for stability)
    m = 5
    a = 0.6

    print("Single-channel delay feedback: P(z) = 1 - a z^{-m}")
    print(f"  m = {m}, a = {a}")
    print()

    # -------------------------------------------------------------------------
    # 1) Analytic poles
    # -------------------------------------------------------------------------
    poles_true = analytic_poles(a, m)
    print("1) Analytic poles (true):")
    for k, z in enumerate(poles_true):
        print(f"   z_{k} = {z:.10f}  (|z| = {np.abs(z):.10f})")
    print()

    # -------------------------------------------------------------------------
    # 2) Build FLAMO model (1x1: one delay, one gain)
    # -------------------------------------------------------------------------
    delays = np.array([m], dtype=int)
    A = np.array([[a]], dtype=np.float64)
    B = np.array([[1.0]], dtype=np.float64)
    C = np.array([[1.0]], dtype=np.float64)
    D = np.array([[0.0]], dtype=np.float64)

    model = pyFDN.dss_to_flamo(
        A=A, B=B, C=C, D=D, m=delays, Fs=1.0, nfft=256, shell=False
    )
    core = model.get_core() if callable(getattr(model, "get_core", None)) else model
    recursion = list(core.branchA)[1]
    decomposition = pyFDN.flamo_extract_pr_decomposition(
        model, delays, recursion_module=recursion
    )
    # decomposition is a dict with keys "P", "B", "C", "D"
    p_probe = decomposition["P"]

    # -------------------------------------------------------------------------
    # 2) Compare P(z) and (d/dz) log det P at sample points (not at poles)
    # -------------------------------------------------------------------------
    test_z = [0.5 + 0.3j, 1.2 - 0.4j, 0.8 * np.exp(0.5j)]
    print("2) Compare P(z) and (d/dz) log det P at sample z (FLAMO vs analytic):")
    for z in test_z:
        P_ana = analytic_P(z, a, m)
        P_flamo = p_probe.at(z)
        P_flamo = np.asarray(P_flamo).reshape(-1)[0]

        d_ana = analytic_log_det_derivative(z, a, m)
        d_flamo = p_probe.log_det_derivative(z)  # Python complex

        err_P = abs(P_ana - P_flamo)
        err_d = abs(d_ana - d_flamo)
        print(f"   z = {z}")
        print(f"      P:     analytic = {P_ana:.4f},  FLAMO = {P_flamo:.4f},  |Δ| = {err_P:.2e}")
        print(f"      d/dz:  analytic = {d_ana:.4f},  FLAMO = {d_flamo:.4f},  |Δ| = {err_d:.2e}")
    print()

    # -------------------------------------------------------------------------
    # 3) EAI with analytic inv_newton (no delay term) -> compare to true poles
    # -------------------------------------------------------------------------
    # Initial poles on unit circle (same convention as flamo_to_pr)
    poles_init = np.exp(2.0j * np.pi * np.arange(m, dtype=np.float64) / m)
    poles_analytic_eai = eai_refine_analytic(
        poles_init, a, m, max_iter=200, tol=1e-12, verbose=False
    )
    # Sort by angle for comparison (same root order)
    def sort_by_angle(p: np.ndarray) -> np.ndarray:
        return p[np.argsort(np.angle(p))]
    poles_true_sorted = sort_by_angle(poles_true)
    poles_analytic_eai_sorted = sort_by_angle(poles_analytic_eai)
    err_eai_analytic = np.max(np.abs(poles_true_sorted - poles_analytic_eai_sorted))
    print("3) EAI with analytic (d/dz) log det P (no delay term):")
    print(f"   max |poles_EAI - poles_true| = {err_eai_analytic:.3e}")
    assert err_eai_analytic < 1e-8, "Analytic EAI should match true poles"
    print()

    # -------------------------------------------------------------------------
    # 4) FLAMO flamo_to_pr (uses FLAMO loop, no delay term in inverse_newton_step)
    # -------------------------------------------------------------------------
    # Diagnostic: compare EAI denominator (inv_newton) from analytic vs FLAMO at same z
    z_test = poles_init[0]
    inv_ana = analytic_log_det_derivative(z_test, a, m)
    P_z = np.asarray(p_probe.at(z_test)).reshape(-1)
    dP_z = np.asarray(p_probe.der(z_test)).reshape(-1)
    inv_flamo_trace = np.trace(np.linalg.solve(P_z.reshape(1, 1), dP_z.reshape(1, 1)))
    inv_flamo_log = p_probe.log_det_derivative(z_test)
    print("4) FLAMO flamo_to_pr (no delay term in denominator):")
    print("   Diagnostic at first initial pole z =", z_test)
    print("      analytic (d/dz)log det P =", inv_ana)
    print("      FLAMO trace(P^{-1} dP/dz) =", inv_flamo_trace)
    print("      FLAMO log_det_derivative  =", inv_flamo_log)
    print("      |analytic - FLAMO trace| =", abs(inv_ana - inv_flamo_trace))
    print()

    # -------------------------------------------------------------------------
    # 4a) Single-iteration diagnostic: one EAI step, same initial poles
    # -------------------------------------------------------------------------
    # Initial poles exactly as FLAMO uses (then sort by angle once)
    pole_angles = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
    poles_one_step = np.exp(1j * pole_angles)
    order = np.argsort(np.angle(poles_one_step))
    poles_one_step = poles_one_step[order]
    idx = 0  # first pole
    z_i = poles_one_step[idx]
    # Deflation (same for both): sum_{j != idx} 1 / (z_i - poles[j])
    deflation_one = 0.0 + 0.0j
    for j in range(m):
        if j != idx:
            deflation_one += 1.0 / (z_i - poles_one_step[j])
    # Analytic
    inv_ana_step = analytic_log_det_derivative(z_i, a, m)
    denom_ana_step = inv_ana_step - deflation_one
    z_new_ana = z_i - 1.0 / denom_ana_step if np.isfinite(denom_ana_step) and abs(denom_ana_step) > 1e-20 else z_i
    # FLAMO (trace = (d/dz) log det P from probe)
    P_i = np.asarray(p_probe.at(z_i)).reshape(-1)
    dP_i = np.asarray(p_probe.der(z_i)).reshape(-1)
    inv_flamo_step = np.trace(np.linalg.solve(P_i.reshape(1, 1), dP_i.reshape(1, 1)))
    denom_flamo_step = inv_flamo_step - deflation_one
    z_new_flamo = z_i - 1.0 / denom_flamo_step if np.isfinite(denom_flamo_step) and abs(denom_flamo_step) > 1e-20 else z_i
    print("4a) Single-iteration diagnostic (first pole only, same initial poles):")
    print(f"   z_old     = {z_i:.4f}")
    print("   Analytic: inv_newton = {:.4g},  deflation = {:.4g},  denom = {:.4g}  -> z_new = {:.4g}".format(inv_ana_step, deflation_one, denom_ana_step, z_new_ana))
    print("   FLAMO:    inv_newton = {:.4g},  deflation = {:.4g},  denom = {:.4g}  -> z_new = {:.4g}".format(inv_flamo_step, deflation_one, denom_flamo_step, z_new_flamo))
    print("   |z_new_analytic - z_new_FLAMO| = {:.3e}".format(abs(z_new_ana - z_new_flamo)))
    print()

    residues_flamo, poles_flamo, direct_flamo, is_pair, meta = pyFDN.flamo_to_pr(
        delays=delays,
        decomposition=decomposition,
        feedback_delay_units=0,
        absorption_delay_units=0,
        maximum_iterations=200,
        reject_unstable_poles=False,
        deflation_type="fullDeflation",
        quality_threshold=1e-15,
        refinement_tol=1e-12,  # stop when max|Δz| < 1e-12 (like analytic EAI)
        verbose=True,
    )
    # 4b: FLAMO now runs to convergence (refinement_tol); compare to analytic EAI and true poles
    refined_flamo = np.asarray(meta["refinedPoles"], dtype=np.complex128).ravel()
    n_refined = len(refined_flamo)
    if n_refined != m:
        print(f"   [Warning] FLAMO refinedPoles has {n_refined} poles, expected {m}")
    refined_flamo_sorted = sort_by_angle(refined_flamo)
    err_flamo_analytic = np.max(np.abs(refined_flamo_sorted - poles_analytic_eai_sorted))
    err_flamo_true = np.max(np.abs(refined_flamo_sorted - poles_true_sorted))
    print("4b) FLAMO refined (refinement_tol=1e-12) vs analytic EAI vs true poles:")
    print(f"   FLAMO iterations = {meta.get('iterations', 'N/A')},  stepCounter = {meta.get('stepCounter', 'N/A')}")
    print(f"   max |refined_FLAMO - analytic_EAI| = {err_flamo_analytic:.3e}")
    print(f"   max |refined_FLAMO - true_poles|   = {err_flamo_true:.3e}")
    print()

    print("{:<22} {:<22} {:<22}".format("FLAMO refinedPoles", "Analytic EAI", "True Poles"))
    for flamo_pole, ana_pole, true_pole in zip(refined_flamo_sorted, poles_analytic_eai_sorted, poles_true_sorted):
        print("{:<22.6g} {:<22.6g} {:<22.6g}".format(flamo_pole, ana_pole, true_pole))
    assert err_flamo_analytic < 1e-6, "FLAMO refined should match analytic EAI"
    assert err_flamo_true < 1e-6, "FLAMO refined should match true poles"
    print()

    # # -------------------------------------------------------------------------
    # # 5) Sanity: IR reconstruction
    # # -------------------------------------------------------------------------
    # ir_len = 256
    # ir_time = pyFDN.dss_to_impz(ir_len, delays, A, B, C, D)[:, 0, 0]
    # ir_modal = pyFDN.pr_to_impz(residues_flamo, poles_flamo, direct_flamo, is_pair, ir_len)[:, 0, 0]
    # err_ir = np.max(np.abs(ir_time - ir_modal))
    # print("5) IR reconstruction (time-domain vs modal from FLAMO PR):")
    # print(f"   max |IR_time - IR_modal| = {err_ir:.3e}")
    # assert err_ir < 1e-5
    # print("Done. Analytic and FLAMO agree; no delay term used.")


if __name__ == "__main__":
    main()
