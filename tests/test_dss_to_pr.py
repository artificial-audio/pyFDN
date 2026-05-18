"""Tests for dss_to_pr_direct / dss_to_pr_flamo and pr_to_impz / impz_to_res pipeline."""

from __future__ import annotations

import numpy as np

from pyFDN.generate.random_orthogonal import random_orthogonal
from pyFDN.translate.dss_to_impz import dss_to_impz
from pyFDN.translate.dss_to_pr_direct import dss_to_pr_direct
from pyFDN.translate.impz_to_res import impz_to_res
from pyFDN.translate.pr_to_impz import pr_to_impz


def test_dss_to_pr_direct_reconstructs_impulse_response():
    delays = np.array([3, 4, 5, 6], dtype=int)
    a = 0.65 * random_orthogonal(delays.size)
    b = np.eye(delays.size, 1)
    c = np.eye(1, delays.size)
    d = np.zeros((1, 1))

    residues, poles, direct, is_conj, _ = dss_to_pr_direct(
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


def test_example_dss_to_pr_residue_lstsq_match():
    np.random.seed(5)
    n = 4
    delays = np.random.randint(50, 101, size=n)
    a = random_orthogonal(n)
    b = np.eye(n, 1)
    c = np.eye(1, n)
    d = np.zeros((1, 1))

    ir_len = 4 * int(np.sum(delays))
    ir_time = dss_to_impz(ir_len, delays, a, b, c, d)[:, 0, 0]
    residues, poles, direct, is_conj, _ = dss_to_pr_direct(
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
