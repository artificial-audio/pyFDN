"""Tests for dss_to_pr (modes: eig, roots, eai).

Covers:
- IR round-trip for each mode (modal reconstruction ≈ time-domain simulation)
- Cross-mode agreement (eai poles/residues match the roots backend)
- impz_to_res closes the loop (residues recovered from an IR match those
  produced directly)
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment

from pyFDN.generate.random_orthogonal import random_orthogonal
from pyFDN.translate.dss_to_impz import dss_to_impz
from pyFDN.translate.dss_to_pr import dss_to_pr
from pyFDN.translate.impz_to_res import impz_to_res
from pyFDN.translate.pr_to_impz import pr_to_impz


def _hungarian_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair entries of two complex pole arrays by minimum Euclidean distance.

    Robust to degenerate sort orders (e.g. multiple poles at the same angle)
    that would break a naive sort-by-angle element-wise comparison.
    """
    cost = np.abs(a[:, None] - b[None, :])
    return linear_sum_assignment(cost)


# ───────────────────────────────────────────────────────────────────────────
# Mode round-trips: each mode must reconstruct the time-domain IR exactly.
# ───────────────────────────────────────────────────────────────────────────


def test_dss_to_pr_eig_mode_reconstructs_ir():
    delays = np.array([4, 5, 6, 7], dtype=int)
    a = 0.55 * random_orthogonal(delays.size)
    b = np.eye(delays.size, 1)
    c = np.eye(1, delays.size)
    d = np.zeros((1, 1))

    residues, poles, direct, is_pair, _ = dss_to_pr(delays, a, b, c, d, mode="eig")

    ir_len = 512
    ir_time = dss_to_impz(ir_len, delays, a, b, c, d)[:, 0, 0]
    ir_modal = pr_to_impz(residues, poles, direct, is_pair, ir_len)[:, 0, 0]

    np.testing.assert_allclose(ir_modal, ir_time, rtol=1e-7, atol=1e-8)


def test_dss_to_pr_roots_mode_reconstructs_ir():
    delays = np.array([3, 4, 5, 6], dtype=int)
    a = 0.65 * random_orthogonal(delays.size)
    b = np.eye(delays.size, 1)
    c = np.eye(1, delays.size)
    d = np.zeros((1, 1))

    residues, poles, direct, is_conj, _ = dss_to_pr(
        delays,
        a,
        b,
        c,
        d,
        mode="roots",
    )

    ir_len = 512
    ir_time = dss_to_impz(ir_len, delays, a, b, c, d)[:, 0, 0]
    ir_modal = pr_to_impz(residues, poles, direct, is_conj, ir_len)[:, 0, 0]

    np.testing.assert_allclose(ir_modal, ir_time, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(direct, d, rtol=0, atol=0)
    assert poles.ndim == 1
    assert poles.size > 0


def test_dss_to_pr_roots_mode_meta_shape():
    """Surface-level sanity: shapes/keys of the meta tuple."""
    delays = np.array([4, 5, 6], dtype=int)
    a = 0.5 * random_orthogonal(delays.size)
    b = np.eye(delays.size, 1)
    c = np.eye(1, delays.size)
    d = np.zeros((1, 1))

    residues, poles, direct, is_pair, meta = dss_to_pr(delays, a, b, c, d, mode="roots")

    assert residues.ndim == 3
    assert poles.ndim == 1
    assert direct.shape == (1, 1)
    assert is_pair.shape == poles.shape
    assert "undrivenResidues" in meta


# ───────────────────────────────────────────────────────────────────────────
# EAI mode: agrees with the direct roots backend at machine precision.
# ───────────────────────────────────────────────────────────────────────────


def test_dss_to_pr_eai_matches_roots():
    """EAI mode (float64 + SVD post-refinement) must agree with the roots
    backend at near-machine precision in poles, residues and reconstructed IR."""
    delays = np.array([4, 5, 6], dtype=int)
    a = np.array([[0.25, -0.1, 0.05], [0.15, 0.3, -0.2], [0.1, 0.05, 0.2]])
    b = np.eye(3, 1)
    c = np.eye(1, 3)
    d = np.zeros((1, 1))

    res_ref, pol_ref, direct_ref, pair_ref, _ = dss_to_pr(
        delays, a, b, c, d, mode="roots"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_new, pol_new, direct_new, pair_new, _ = dss_to_pr(
            delays,
            a,
            b,
            c,
            d,
            mode="eai",
            Fs=1.0,
            nfft=1024,
            verbose=False,
        )

    # Same pole-set cardinality (after conjugate-pair reduction).
    assert pol_new.size == pol_ref.size

    # Hungarian-matched pole/residue error: tolerates degenerate sort orders.
    row, col = _hungarian_match(pol_ref, pol_new)
    np.testing.assert_allclose(pol_new[col], pol_ref[row], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(res_new[col], res_ref[row], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(direct_new, direct_ref, rtol=0, atol=0)

    # IR round-trip
    ir_len = 512
    ir_time = dss_to_impz(ir_len, delays, a, b, c, d)[:, 0, 0]
    ir_modal = pr_to_impz(res_new, pol_new, direct_new, pair_new, ir_len)[:, 0, 0]
    np.testing.assert_allclose(ir_modal, ir_time, rtol=1e-10, atol=1e-10)


# ───────────────────────────────────────────────────────────────────────────
# Residue closure: impz_to_res recovers residues from the IR.
# ───────────────────────────────────────────────────────────────────────────


def test_dss_to_pr_residue_lstsq_match():
    np.random.seed(5)
    n = 4
    delays = np.random.randint(50, 101, size=n)
    a = random_orthogonal(n)
    b = np.eye(n, 1)
    c = np.eye(1, n)
    d = np.zeros((1, 1))

    ir_len = 4 * int(np.sum(delays))
    ir_time = dss_to_impz(ir_len, delays, a, b, c, d)[:, 0, 0]
    residues, poles, direct, is_conj, _ = dss_to_pr(
        delays,
        a,
        b,
        c,
        d,
        mode="eig",
    )
    ir_modal = pr_to_impz(residues, poles, direct, is_conj, ir_len)[:, 0, 0]
    residues_from_ir, _, _ = impz_to_res(ir_time, poles, is_conj)

    np.testing.assert_allclose(ir_modal, ir_time, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        residues[:, 0, 0],
        residues_from_ir,
        rtol=1e-6,
        atol=1e-7,
    )
