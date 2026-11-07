"""Check denominator recovery and real example setups through production code."""

import numpy as np
import torch

import pyFDN
from pyFDN.translate._lindsey_fox_flamo import _model_polynomial
from tests.modal_examples import setup_cell


def test_model_polynomial_clears_sos_denominators():
    delays = np.array([3, 5])
    a = np.array([[0.4, 0.1], [-0.15, 0.3]])
    sos = np.zeros((1, 6, 2))
    sos[0, 0] = [0.6, 0.5]
    sos[0, 1] = [0.1, -0.05]
    sos[0, 3] = 1
    sos[0, 4] = [-0.2, -0.3]
    model = pyFDN.dss_to_flamo(
        delays=delays,
        A=a,
        B=np.ones((2, 1)),
        C=np.ones((1, 2)),
        D=[[0]],
        fs=48000,
        post_delay=sos,
        dtype=torch.float64,
        shell=False,
    )
    coefficients = _model_polynomial(pyFDN.flamo_decompose_for_pr(model), 10)
    for w in [0, 0.7 + 0.2j, 0.3 - 0.9j]:
        denominator = 1 + sos[0, 4] * w
        numerator = sos[0, 0] + sos[0, 1] * w
        expected = np.linalg.det(
            np.diag(denominator) - np.diag(w**delays * numerator) @ a
        )
        np.testing.assert_allclose(
            np.polynomial.polynomial.polyval(w, coefficients), expected, atol=1e-12
        )
    result = pyFDN.flamo_to_pr(model, mode="lindsey_fox", num_poles=10)
    assert np.sum(1 + result[3]) == 10
    dss = pyFDN.DSS(delays, a, np.ones((2, 1)), np.ones((1, 2)), np.zeros((1, 1)))
    np.testing.assert_allclose(
        pyFDN.pr_to_impz(*result[:4], 256),
        pyFDN.dss_to_td(*dss, post_delay=sos).impulse_response(256),
        atol=1e-10,
    )


def test_model_adapter_reconstructs_static_and_fir_models():
    delays = np.array([3, 5])
    a_static = np.array([[0.4, 0.1], [-0.15, 0.3]])
    for a in [a_static, np.stack([a_static, 0.05 * np.eye(2)], axis=2)]:
        model = pyFDN.dss_to_flamo(
            delays=delays,
            A=a,
            B=np.ones((2, 1)),
            C=np.ones((1, 2)),
            D=[[0]],
            fs=48000,
            dtype=torch.float64,
            shell=False,
        )
        degree = 8 if a.ndim == 2 else 10
        result = pyFDN.flamo_to_pr(model, mode="lindsey_fox", num_poles=degree)
        assert np.sum(1 + result[3]) == degree
        np.testing.assert_allclose(
            pyFDN.pr_to_impz(*result[:4], 256),
            pyFDN.dss_to_impz(delays, a, np.ones((2, 1)), np.ones((1, 2)), [[0]], 256),
            atol=1e-10,
        )


def test_setup_loader_uses_the_original_notebook_cell():
    setup = setup_cell("example_dss_to_pr_direct.py", "A")
    np.testing.assert_array_equal(setup["delays"], [41, 53, 67, 79])
    np.testing.assert_allclose(
        setup["A"].T @ setup["A"], 0.65**2 * np.eye(4), atol=1e-14
    )
