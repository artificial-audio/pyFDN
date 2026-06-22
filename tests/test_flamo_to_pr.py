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
        A=a,
        B=b,
        C=c,
        D=d,
        m=delays,
        Fs=1.0,
        nfft=1024,
        shell=False,
        dtype=torch.float64,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_model, pol_model, direct_model, pair_model, meta_model = flamo_to_pr(
            model, verbose=False
        )
        res_wrap, pol_wrap, direct_wrap, pair_wrap, _ = dss_to_pr(
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

    assert pol_model.size == pol_wrap.size

    row, col = _hungarian_match(pol_model, pol_wrap)
    np.testing.assert_allclose(pol_wrap[col], pol_model[row], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(res_wrap[col], res_model[row], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(direct_model, direct_wrap, rtol=0, atol=0)
    assert {"P", "B", "C", "D"}.issubset(set(meta_model["decomposition"].keys()))


def test_flamo_to_pr_biquad_in_loop_reconstructs_ir():
    """A genuine 2nd-order section (biquad, ``a2 != 0``) in the recursion loop
    adds 2 poles per delay line. flamo_to_pr must (a) size the pole search to
    include those filter poles — not just the delay-line poles — and (b)
    reconstruct FLAMO's impulse response from the modal data at machine precision.
    """
    from scipy.signal import butter

    from pyFDN.generate.random_orthogonal import random_orthogonal
    from pyFDN.translate.pr_to_impz import pr_to_impz

    Fs = 48000.0
    nfft = 2**13
    delays = np.array([13, 17, 19, 23], dtype=int)
    n = delays.size

    np.random.seed(0)
    a = 0.5 * random_orthogonal(n)
    b = np.eye(n, 1)
    c = np.eye(1, n)
    d = np.zeros((1, 1))

    # Same genuine biquad lowpass (a2 != 0) on every delay line, as (1, 6, N).
    bq_b, bq_a = butter(2, 0.4)  # [b0, b1, b2], [a0, a1, a2]
    sos_loop = np.tile(np.concatenate([bq_b, bq_a])[:, None], (1, n))[np.newaxis, :, :]

    model = dss_to_flamo(
        A=a,
        B=b,
        C=c,
        D=d,
        m=delays,
        Fs=Fs,
        nfft=nfft,
        shell=True,
        sos_filter=sos_loop,
        dtype=torch.float64,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        residues, poles, direct, is_pair, _ = flamo_to_pr(
            model,
            reject_unstable_poles=True,
            maximum_iterations=300,
            quality_threshold=1e-10,
            refinement_tol=1e-12,
            verbose=False,
        )

    # Pole count includes the biquad's 2 poles per line (a2 != 0), not just the
    # delay-line poles. is_pair counts a real pole as 1 and a conjugate pair as 2.
    expanded = int(np.sum(np.asarray(is_pair).astype(int) + 1))
    assert expanded == int(delays.sum()) + 2 * n
    assert np.all(np.abs(poles) < 1.0 + 1e-9)  # stable FDN

    # Modal reconstruction matches FLAMO's true impulse response (the only IR
    # reference once there is an IIR filter in the loop).
    ir_flamo = np.asarray(
        model.get_time_response(fs=int(Fs)).squeeze(), dtype=np.float64
    )
    if ir_flamo.ndim == 3:
        ir_flamo = ir_flamo[:, 0, 0]
    elif ir_flamo.ndim == 2:
        ir_flamo = ir_flamo[:, 0]
    ir_len = min(4000, ir_flamo.size)
    ir_modal = pr_to_impz(residues, poles, direct, is_pair, ir_len)[:, 0, 0]
    np.testing.assert_allclose(ir_modal, ir_flamo[:ir_len], rtol=1e-6, atol=1e-9)
