"""
Dedicated FLAMO/autograd DSS2PR example.

Design:
- Normal P(z) and dP/dz in FLAMO: characteristic matrix is P(z) = I - A @ D(z)
  (delays on columns, same as analytic/GCP). FLAMO Recursion exposes this via
  probe_recursion(z) and log_det_derivative(z).
- EAI (Ehrlich-Aberth) for pole refinement is run in the w-plane (w = z^{-1}):
  Newton term Q'(w)/Q(w), deflation in w, step in w, then z_new = 1/w_new.
  This is the default in flamo_to_pr(..., use_w_plane_step=True).

This example: build DSS -> FLAMO -> PR, verify P(z) and (d/dz)log det P match
analytic, then run pole refinement (EAI in w) and compare IR.

For higher numerical accuracy, use dtype=torch.float64 when building the model
(dss_to_flamo(..., dtype=torch.float64) or dss_to_pr_flamo(..., dtype=torch.float64)).
"""

from __future__ import annotations

import numpy as np
import torch

import pyFDN

import matplotlib.pyplot as plt


def analytic_P(z: complex, A: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Normal convention: P(z) = I - A @ diag(z^{-m1}, ..., z^{-mN})."""
    m = np.asarray(m, dtype=int).ravel()
    N = len(m)
    D = np.diag([z ** (-int(mk)) for mk in m])
    return np.eye(N, dtype=np.complex128) - (A.astype(np.complex128) @ D)


def analytic_dP_dz(z: complex, A: np.ndarray, m: np.ndarray) -> np.ndarray:
    """dP/dz for P(z) = I - A @ diag(z^{-m_k})."""
    m = np.asarray(m, dtype=int).ravel()
    N = len(m)
    dP = np.zeros((N, N), dtype=np.complex128)
    for j in range(N):
        k = int(m[j])
        dP[:, j] = A[:, j] * (k * (z ** (-k - 1)))
    return dP


def analytic_log_det_derivative(z: complex, A: np.ndarray, m: np.ndarray) -> complex:
    """(d/dz) log det P(z) = trace(P^{-1} dP/dz)."""
    P = analytic_P(z, A, m)
    dP = analytic_dP_dz(z, A, m)
    detP = np.linalg.det(P)
    if np.abs(detP) < 1e-20:
        return np.inf + 0.0j
    return np.trace(np.linalg.solve(P, dP))


def main() -> None:
    np.random.seed(7)

    # Small stable FDN
    n = 4
    delays = np.array([53, 67, 79, 97], dtype=int)
    a_num = 0.7 * pyFDN.random_orthogonal(n)
    b = np.ones((n, 1))
    c = np.ones((1, n))
    d = np.zeros((1, 1))

    # DSS -> FLAMO model -> decomposition (P probe = normal P(z), dP/dz).
    # Optional: dtype=torch.float64 for higher numerical accuracy in probing and poles.
    model = pyFDN.dss_to_flamo(
        A=a_num,
        B=b,
        C=c,
        D=d,
        m=delays,
        Fs=1.0,
        nfft=2**12,
        shell=False,
        dtype=torch.float64,  # increase numerical accuracy
    )
    core = model.get_core() if callable(getattr(model, "get_core", None)) else model
    recursion_module = list(core.branchA)[1]
    decomposition = pyFDN.flamo_extract_pr_decomposition(
        model,
        delays,
        recursion_module=recursion_module,
    )
    p_probe = decomposition["P"]

    # Verify: FLAMO P(z) and (d/dz)log det P match analytic (normal convention)
    print("1) Verify normal P(z) and dP/dz in FLAMO (vs analytic)")
    test_z = [0.6 + 0.2j, 0.9 - 0.3j, 1.1 * np.exp(0.5j)]
    for z in test_z:
        P_ana = analytic_P(z, a_num, delays)
        P_flamo = np.asarray(p_probe.at(z)).reshape(n, n)
        # print(f"P_ana: {P_ana}")
        # print(f"P_flamo: {P_flamo}")
        # print(f"P_flamo / P_ana: {P_flamo / P_ana}")
        d_ana = analytic_log_det_derivative(z, a_num, delays)
        d_flamo = p_probe.log_det_derivative(z)
        err_P = np.max(np.abs(P_ana - P_flamo))
        err_d = abs(d_ana - d_flamo)
        print(f"   z = {z:.2e}: max|P_ana - P_flamo| = {err_P:.2e},  |(d/dz)ana - (d/dz)flamo| = {err_d:.2e}")
    print()

    # Pole refinement: EAI in w-plane (default use_w_plane_step=True).
    # Quality = rcond(P(pole)); converged when quality <= quality_threshold. For many
    # poles (e.g. 296), use enough iterations so non-converged count can drop.
    print("2) FLAMO -> PR (EAI in w-plane, use_w_plane_step=True)")
    residues, poles, direct, is_pair, meta = pyFDN.flamo_to_pr(
        delays=delays,
        decomposition=decomposition,
        feedback_delay_units=0,
        maximum_iterations=30,
        reject_unstable_poles=False,
        deflation_type="fullDeflation",
        quality_threshold=1e-10,
        refinement_tol=1e-7,
        verbose=True,
        use_w_plane_step=True,  # EAI in w: Q'(w)/Q(w), deflation in w, z_new = 1/w_new
    )

    ir_len = 1024
    ir_time = pyFDN.dss_to_impz(ir_len, delays, a_num, b, c, d)[:, 0, 0]
    ir_modal = pyFDN.pr_to_impz(residues, poles, direct, is_pair, ir_len)[:, 0, 0]

    err = np.max(np.abs(ir_time - ir_modal))
    print("max |IR_time - IR_modal|:", err)

    plt.plot(ir_time)
    plt.plot(ir_modal)
    plt.legend(["IR_time", "IR_modal"])
    plt.show()


if __name__ == "__main__":
    main()
