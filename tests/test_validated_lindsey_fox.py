"""Acceptance and failure-path tests for validated modal decomposition."""

from importlib import import_module

import numpy as np
import pytest
import torch

import pyFDN
from pyFDN.auxiliary.lindsey_fox import lindsey_fox_roots
from pyFDN.auxiliary.modal import validate_modal_response


def test_kronecker_close_pairs_complete_without_fallback():
    delays = np.array([53, 61, 71, 79, 89, 101, 113, 127])
    a = pyFDN.kronecker_matrix(np.full(3, np.pi / 4))
    result = pyFDN.dss_to_pr(
        delays,
        a,
        np.ones((8, 1)),
        np.ones((1, 8)),
        [[0]],
        mode="lindsey_fox",
        lf_max_seconds=10,
    )
    assert result[4]["solver"] == "lindsey_fox"
    assert result[4]["complete"] and result[4]["returnedPoles"] == 694
    reference = pyFDN.dss_to_impz(
        delays, a, np.ones((8, 1)), np.ones((1, 8)), [[0]], 4096
    )
    np.testing.assert_allclose(
        pyFDN.pr_to_impz(*result[:4], 4096), reference, atol=1e-9, rtol=1e-9
    )


def test_failed_search_uses_eai_and_reports_reason(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("forced grid budget exhaustion")

    monkeypatch.setattr("pyFDN.auxiliary.lindsey_fox.lindsey_fox_roots", fail)
    a = np.array([[0.4, 0.1], [-0.1, 0.3]])
    with pytest.warns(RuntimeWarning, match="falling back to EAI"):
        result = pyFDN.dss_to_pr(
            [3, 5], a, np.eye(2), np.eye(2), np.zeros((2, 2)), mode="lindsey_fox"
        )
    meta = result[4]
    assert meta["solver"] == "eai" and meta["complete"]
    assert "grid budget" in meta["fallbackReason"]
    assert meta["responseValidation"]["relativeL2"] < 1e-8


def test_all_port_check_rejects_hidden_missing_mode():
    # C=[1,0] would hide the second mode entirely; validate the internal ports.
    poles = np.array([0.8])
    residues = np.array([[[1.0, 0.0], [0.0, 0.0]]])
    with pytest.raises(RuntimeError, match="All-port"):
        validate_modal_response(
            poles,
            residues,
            np.array([False]),
            lambda z: np.diag(1 / (z - np.array([0.8, 0.7]))),
            2,
            1e-8,
        )


def test_bad_residues_trigger_fallback(monkeypatch):
    module = import_module("pyFDN.translate.dss_to_pr")
    original = module._dss_to_res
    calls = []

    def corrupt_first(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            return (2 * result[0], *result[1:])
        return result

    monkeypatch.setattr(module, "_dss_to_res", corrupt_first)
    with pytest.warns(RuntimeWarning, match="falling back to EAI"):
        result = pyFDN.dss_to_pr(
            [3, 5],
            [[0.4, 0.1], [-0.1, 0.3]],
            np.eye(2),
            np.eye(2),
            np.zeros((2, 2)),
            mode="lindsey_fox",
        )
    assert result[4]["solver"] == "eai"
    assert "residues disagree" in result[4]["fallbackReason"]


def test_kernel_budget_and_fft_limits():
    coefficients = np.zeros(128)
    coefficients[[0, -1]] = [-0.7, 1]
    with pytest.raises(RuntimeError, match="time budget"):
        lindsey_fox_roots(coefficients, 0.99, 1.0, max_seconds=1e-12)
    with pytest.raises(RuntimeError, match="completeness/reconstruction"):
        lindsey_fox_roots(coefficients, 0.99, 1.0, max_fft_size=8)


def test_invalid_eai_fallback_is_rejected(monkeypatch):
    module = import_module("pyFDN.translate.dss_to_pr")
    original = module.dss_to_pr

    def fail(*args, **kwargs):
        raise RuntimeError("forced failure")

    def corrupted_eai(*args, **kwargs):
        result = original(*args, **kwargs)
        return (2 * result[0], *result[1:])

    monkeypatch.setattr("pyFDN.auxiliary.lindsey_fox.lindsey_fox_roots", fail)
    monkeypatch.setattr(module, "dss_to_pr", corrupted_eai)
    with pytest.warns(RuntimeWarning, match="falling back to EAI"):
        with pytest.raises(RuntimeError, match="residues disagree"):
            original(
                [3, 5],
                [[0.4, 0.1], [-0.1, 0.3]],
                np.eye(2),
                np.eye(2),
                np.zeros((2, 2)),
                mode="lindsey_fox",
            )


def _absorption_model():
    sos = np.zeros((1, 6, 2))
    sos[0, 0] = 0.6
    sos[0, 1] = 0.001
    sos[0, 3] = 1
    build = pyFDN.FDNBuild(
        A=np.array([[0.4, 0.1], [-0.1, 0.3]]),
        B=np.eye(2),
        C=np.eye(2),
        D=np.zeros((2, 2)),
        delays=np.array([17, 23]),
        fs=48000,
        post_delay=sos,
    )
    return build, pyFDN.build_to_flamo(build, dtype=torch.float64, device="cpu")


def test_absorption_approximation_requires_opt_in_and_response_check():
    build, model = _absorption_model()
    with pytest.warns(RuntimeWarning, match="falling back to EAI"):
        with pytest.raises(RuntimeError, match="allow_approximate=True"):
            pyFDN.flamo_to_pr(model, mode="lindsey_fox", lf_max_seconds=1e-12)
    with pytest.warns(RuntimeWarning) as captured:
        result = pyFDN.flamo_to_pr(
            model, mode="lindsey_fox", lf_max_seconds=1e-12, allow_approximate=True
        )
    assert len(captured) >= 2
    meta = result[4]
    assert meta["approximate"] and not meta["complete"]
    assert meta["omittedPoleCount"] == 2
    np.testing.assert_allclose(
        pyFDN.pr_to_impz(*result[:4], 512),
        pyFDN.build_to_impz(build, 512),
        atol=1e-9,
        rtol=1e-9,
    )


def test_flamo_lf_static_and_sos_complete():
    delays = np.array([3, 5])
    for sos in [
        None,
        np.array([[[0.6, 0.5], [0.1, -0.05], [0, 0], [1, 1], [-0.2, -0.3], [0, 0]]]),
    ]:
        build = pyFDN.FDNBuild(
            A=np.array([[0.4, 0.1], [-0.15, 0.3]]),
            B=np.eye(2),
            C=np.eye(2),
            D=np.zeros((2, 2)),
            delays=delays,
            fs=48000,
            post_delay=sos,
        )
        model = pyFDN.build_to_flamo(build, dtype=torch.float64, device="cpu")
        result = pyFDN.flamo_to_pr(model, mode="lindsey_fox")
        assert result[4]["solver"] == "lindsey_fox" and result[4]["complete"]
        np.testing.assert_allclose(
            pyFDN.pr_to_impz(*result[:4], 512),
            pyFDN.build_to_impz(build, 512),
            atol=1e-9,
            rtol=1e-9,
        )


def test_approximation_flag_does_not_allow_missing_outer_modes(monkeypatch):
    _, model = _absorption_model()
    module = import_module("pyFDN.translate._lindsey_fox_flamo")
    original = module.flamo_to_pr

    def omit_mode(*args, **kwargs):
        r, p, d, pairs, meta = original(*args, **kwargs)
        meta["eigenvectors"] = {
            key: value[:, 1:] for key, value in meta["eigenvectors"].items()
        }
        meta["undrivenResidues"] = meta["undrivenResidues"][1:]
        return r[1:], p[1:], d, pairs[1:], meta

    monkeypatch.setattr(module, "flamo_to_pr", omit_mode)
    with pytest.warns(RuntimeWarning, match="falling back to EAI"):
        with pytest.raises(RuntimeError, match="All-port"):
            pyFDN.flamo_to_pr(
                model, mode="lindsey_fox", lf_max_seconds=1e-12, allow_approximate=True
            )


def test_eai_absorption_approximation_uses_the_same_acceptance_policy():
    build, model = _absorption_model()
    with pytest.raises(RuntimeError, match="allow_approximate=True"):
        pyFDN.flamo_to_pr(model)
    with pytest.warns(RuntimeWarning, match="validated approximate"):
        result = pyFDN.flamo_to_pr(model, allow_approximate=True)
    meta = result[4]
    assert meta["solver"] == meta["requestedSolver"] == "eai"
    assert meta["omittedPoleCount"] == 2 and not meta["complete"]
    assert meta["responseValidation"]["relativeL2"] < 1e-8
    assert meta["externalResponseValidation"]["relativeL2"] < 1e-8
    np.testing.assert_allclose(
        pyFDN.pr_to_impz(*result[:4], 512),
        pyFDN.build_to_impz(build, 512),
        atol=1e-9,
    )
