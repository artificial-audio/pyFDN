"""Tests for the dedicated FLAMO/autograd dss_to_pr_flamo entry point."""

from __future__ import annotations

import numpy as np
import pytest

from pyFDN.translate.dss_to_pr import dss_to_pr
from pyFDN.translate.dss_to_pr_flamo import dss_to_pr_flamo


class ConstantFeedbackGraph:
    """Minimal graph-like object accepted by FLAMO probe adapters."""

    def __init__(self, matrix: np.ndarray):
        self._matrix = np.asarray(matrix, dtype=np.float64)
        self.input_channels = self._matrix.shape[1]
        self.output_channels = self._matrix.shape[0]

    def at(self, z):
        return self._matrix

    def der(self, z):
        return np.zeros_like(self._matrix)


def test_dss_to_pr_flamo_matches_autograd_backend():
    delays = np.array([2, 3], dtype=int)
    a = np.array([[0.25, -0.1], [0.15, 0.3]])
    b = np.eye(2, 1)
    c = np.eye(1, 2)
    d = np.zeros((1, 1))
    graph = ConstantFeedbackGraph(a)

    res_ref, pol_ref, direct_ref, pair_ref, _ = dss_to_pr(
        delays,
        graph,
        b,
        c,
        d,
        feedback_delay_units=0,
        probe_backend="autograd",
        verbose=False,
    )
    res_new, pol_new, direct_new, pair_new, _ = dss_to_pr_flamo(
        delays,
        graph,
        b,
        c,
        d,
        feedback_delay_units=0,
        verbose=False,
    )

    np.testing.assert_allclose(pol_new, pol_ref, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(res_new, res_ref, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(direct_new, direct_ref, rtol=0, atol=0)
    np.testing.assert_array_equal(pair_new, pair_ref)


def test_dss_to_pr_flamo_rejects_backend_override():
    delays = np.array([2, 3], dtype=int)
    a = np.eye(2)
    b = np.eye(2, 1)
    c = np.eye(1, 2)
    d = np.zeros((1, 1))

    with pytest.raises(TypeError):
        dss_to_pr_flamo(
            delays,
            ConstantFeedbackGraph(a),
            b,
            c,
            d,
            probe_backend="manual",
        )

