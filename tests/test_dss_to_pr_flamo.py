"""Tests for FLAMO-first dss_to_pr_flamo / flamo_to_pr entrypoints."""

from __future__ import annotations

import numpy as np
import pytest

from pyFDN.translate.dss_to_flamo import dss_to_flamo
from pyFDN.translate.dss_to_pr import dss_to_pr
from pyFDN.translate.dss_to_pr_flamo import (
    dss_to_pr_flamo,
    flamo_extract_pr_decomposition,
    flamo_to_pr,
)


def test_dss_to_pr_flamo_matches_autograd_backend():
    delays = np.array([2, 3], dtype=int)
    a = np.array([[0.25, -0.1], [0.15, 0.3]])
    b = np.eye(2, 1)
    c = np.eye(1, 2)
    d = np.zeros((1, 1))

    res_ref, pol_ref, direct_ref, pair_ref, _ = dss_to_pr(
        delays,
        a,
        b,
        c,
        d,
        feedback_delay_units=0,
        probe_backend="autograd",
        verbose=False,
    )
    res_new, pol_new, direct_new, pair_new, _ = dss_to_pr_flamo(
        delays,
        a,
        b,
        c,
        d,
        feedback_delay_units=0,
        absorption_delay_units=0,
        Fs=1.0,
        nfft=1024,
        verbose=False,
    )

    np.testing.assert_allclose(pol_new, pol_ref, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(res_new, res_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(direct_new, direct_ref, rtol=0, atol=0)
    np.testing.assert_array_equal(pair_new, pair_ref)


def test_flamo_to_pr_matches_dss_to_pr_flamo_wrapper():
    delays = np.array([2, 3], dtype=int)
    a = np.array([[0.25, -0.1], [0.15, 0.3]])
    b = np.eye(2, 1)
    c = np.eye(1, 2)
    d = np.zeros((1, 1))

    model = dss_to_flamo(
        A=a,
        B=b,
        C=c,
        D=d,
        m=delays,
        Fs=1.0,
        nfft=1024,
        shell=False,
    )
    core = model.get_core() if callable(getattr(model, "get_core", None)) else model
    recursion_module = list(core.branchA)[1]

    decomposition = flamo_extract_pr_decomposition(
        model,
        delays,
        recursion_module=recursion_module,
    )
    res_model, pol_model, direct_model, pair_model, meta_model = flamo_to_pr(
        delays=delays,
        decomposition=decomposition,
        feedback_delay_units=0,
        absorption_delay_units=0,
        verbose=False,
    )
    res_wrap, pol_wrap, direct_wrap, pair_wrap, _ = dss_to_pr_flamo(
        delays,
        a,
        b,
        c,
        d,
        feedback_delay_units=0,
        absorption_delay_units=0,
        Fs=1.0,
        nfft=1024,
        verbose=False,
    )

    np.testing.assert_allclose(pol_model, pol_wrap, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(res_model, res_wrap, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(direct_model, direct_wrap, rtol=0, atol=0)
    np.testing.assert_array_equal(pair_model, pair_wrap)
    assert {"P", "B", "C", "D"}.issubset(set(meta_model["decomposition"].keys()))


def test_dss_to_pr_flamo_rejects_absorption_filters():
    delays = np.array([2, 3], dtype=int)
    a = np.eye(2)
    b = np.eye(2, 1)
    c = np.eye(1, 2)
    d = np.zeros((1, 1))

    with pytest.raises(ValueError):
        dss_to_pr_flamo(
            delays,
            a,
            b,
            c,
            d,
            absorption_filters=np.eye(2),
        )

