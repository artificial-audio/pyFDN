"""Tests for the model-driven flamo_to_pr entry point."""

from __future__ import annotations

import warnings

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pyFDN.translate.dss_to_flamo import dss_to_flamo
from pyFDN.translate.dss_to_pr import dss_to_pr
from pyFDN.translate.flamo_to_pr import flamo_to_pr


def _hungarian_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair entries of two complex pole arrays by minimum Euclidean distance."""
    cost = np.abs(a[:, None] - b[None, :])
    return linear_sum_assignment(cost)


def test_flamo_to_pr_matches_dss_to_pr_eai():
    """The model-driven entry and the numeric-matrix entry (mode="eai") run the
    same algorithm — they must produce numerically equivalent poles/residues."""
    delays = np.array([2, 3], dtype=int)
    a = np.array([[0.25, -0.1], [0.15, 0.3]])
    b = np.eye(2, 1)
    c = np.eye(1, 2)
    d = np.zeros((1, 1))

    # Build the model in float64 so flamo_to_pr matches dss_to_pr's
    # (now-default) machine-precision behavior.
    model = dss_to_flamo(
        A=a, B=b, C=c, D=d, m=delays, Fs=1.0, nfft=1024,
        shell=False, dtype=torch.float64,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_model, pol_model, direct_model, pair_model, meta_model = flamo_to_pr(
            model, verbose=False
        )
        res_wrap, pol_wrap, direct_wrap, pair_wrap, _ = dss_to_pr(
            delays, a, b, c, d, mode="eai", Fs=1.0, nfft=1024, verbose=False,
        )

    assert pol_model.size == pol_wrap.size

    row, col = _hungarian_match(pol_model, pol_wrap)
    np.testing.assert_allclose(pol_wrap[col], pol_model[row], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(res_wrap[col], res_model[row], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(direct_model, direct_wrap, rtol=0, atol=0)
    assert {"P", "B", "C", "D"}.issubset(set(meta_model["decomposition"].keys()))
