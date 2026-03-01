"""Tests for DSS-only direct dss_to_pr_direct path."""

from __future__ import annotations

import numpy as np

from pyFDN.generate.random_orthogonal import random_orthogonal
from pyFDN.translate.dss_to_impz import dss_to_impz
from pyFDN.translate.dss_to_pr_direct import dss_to_pr_direct
from pyFDN.translate.pr_to_impz import pr_to_impz


def test_dss_to_pr_direct_reconstructs_ir_eig_mode():
    delays = np.array([4, 5, 6, 7], dtype=int)
    a = 0.55 * random_orthogonal(delays.size)
    b = np.eye(delays.size, 1)
    c = np.eye(1, delays.size)
    d = np.zeros((1, 1))

    residues, poles, direct, is_pair, _ = dss_to_pr_direct(
        delays, a, b, c, d, mode="eig"
    )

    ir_len = 512
    ir_time = dss_to_impz(ir_len, delays, a, b, c, d)[:, 0, 0]
    ir_modal = pr_to_impz(residues, poles, direct, is_pair, ir_len)[:, 0, 0]

    np.testing.assert_allclose(ir_modal, ir_time, rtol=1e-7, atol=1e-8)


def test_dss_to_pr_direct_roots_mode_runs():
    delays = np.array([4, 5, 6], dtype=int)
    a = 0.5 * random_orthogonal(delays.size)
    b = np.eye(delays.size, 1)
    c = np.eye(1, delays.size)
    d = np.zeros((1, 1))

    residues, poles, direct, is_pair, meta = dss_to_pr_direct(
        delays, a, b, c, d, mode="roots"
    )

    assert residues.ndim == 3
    assert poles.ndim == 1
    assert direct.shape == (1, 1)
    assert is_pair.shape == poles.shape
    assert "undrivenResidues" in meta

