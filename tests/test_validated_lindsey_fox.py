"""Acceptance and failure-path tests for validated modal decomposition."""

from functools import partial
from importlib import import_module

import numpy as np
import pytest
import torch

import pyFDN
from pyFDN.auxiliary.lindsey_fox import lindsey_fox_roots
from pyFDN.auxiliary.modal import validate_modal_response


def test_kronecker_close_pairs_complete():
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


@pytest.mark.parametrize("flamo", [False, True])
@pytest.mark.parametrize("failure", [None, "budget", "incomplete"])
def test_lf_never_calls_eai(monkeypatch, flamo, failure):
    module = import_module("pyFDN.translate.flamo_to_pr")

    def forbidden_eai(*args, **kwargs):
        raise AssertionError("LF must never run EAI")

    monkeypatch.setattr(module, "_refine_pole_positions_w", forbidden_eai)
    if failure:

        def fail(*args, **kwargs):
            if failure == "incomplete":
                return np.array([0.7])
            raise RuntimeError("forced grid budget exhaustion")

        monkeypatch.setattr("pyFDN.auxiliary.lindsey_fox.lindsey_fox_roots", fail)
        monkeypatch.setattr(
            "pyFDN.translate._lindsey_fox_flamo.lindsey_fox_roots", fail
        )
    args = ([3, 5], [[0.4, 0.1], [-0.1, 0.3]], np.eye(2), np.eye(2), np.zeros((2, 2)))
    if flamo:
        model = pyFDN.dss_to_flamo(
            delays=args[0],
            A=args[1],
            B=args[2],
            C=args[3],
            D=args[4],
            fs=48000,
            dtype=torch.float64,
            shell=False,
        )
        solve = partial(
            pyFDN.flamo_to_pr, model, mode="lindsey_fox", maximum_iterations=0
        )
    else:
        solve = partial(
            pyFDN.dss_to_pr, *args, mode="lindsey_fox", maximum_iterations=0
        )
    if failure:
        message = (
            "grid budget" if failure == "budget" else "1/8 poles.*complete pole set"
        )
        with pytest.raises(RuntimeError, match=message):
            solve()
    else:
        meta = solve()[4]
        assert meta["solver"] == "lindsey_fox" and meta["complete"]
        assert "fallbackReason" not in meta


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


def test_bad_lf_residues_raise(monkeypatch):
    module = import_module("pyFDN.translate.dss_to_pr")
    original = module._dss_to_res

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        return (2 * result[0], *result[1:])

    monkeypatch.setattr(module, "_dss_to_res", corrupt)
    with pytest.raises(RuntimeError, match="residues disagree"):
        pyFDN.dss_to_pr(
            [3, 5],
            [[0.4, 0.1], [-0.1, 0.3]],
            np.eye(2),
            np.eye(2),
            np.zeros((2, 2)),
            mode="lindsey_fox",
        )


def test_kernel_budget_and_fft_limits():
    coefficients = np.zeros(128)
    coefficients[[0, -1]] = [-0.7, 1]
    with pytest.raises(RuntimeError, match="time budget"):
        lindsey_fox_roots(coefficients, 0.99, 1.0, max_seconds=1e-12)
    with pytest.raises(RuntimeError, match="completeness/reconstruction"):
        lindsey_fox_roots(coefficients, 0.99, 1.0, max_fft_size=8)


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


@pytest.mark.parametrize("allow_approximate", [False, True])
def test_lf_budget_failure_raises_even_with_approximation(allow_approximate):
    _, model = _absorption_model()
    with pytest.raises(RuntimeError, match="time budget"):
        pyFDN.flamo_to_pr(
            model,
            mode="lindsey_fox",
            lf_max_seconds=1e-12,
            allow_approximate=allow_approximate,
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


def test_eai_approximation_still_checks_external_residues(monkeypatch):
    _, model = _absorption_model()
    module = import_module("pyFDN.translate.flamo_to_pr")
    original = module._dss_to_res_flamo

    def corrupt(*args, **kwargs):
        r, d, undriven, vectors = original(*args, **kwargs)
        return 2 * r, d, undriven, vectors

    monkeypatch.setattr(module, "_dss_to_res_flamo", corrupt)
    with pytest.raises(RuntimeError, match="All-port"):
        pyFDN.flamo_to_pr(model, allow_approximate=True)


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
