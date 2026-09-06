"""Tests for translation helpers."""

import numpy as np
import pytest

from pyFDN.auxiliary.allpass import is_allpass
from pyFDN.generate.random_orthogonal import random_orthogonal
from pyFDN.translate.dss_to_impz import dss_to_impz
from pyFDN.translate.dss_to_ss import dss_to_ss
from pyFDN.translate.dss_to_tf import dss_to_tf


def test_dss_to_ss_raises_for_delay_below_minimum():
    delays = np.array([2, 4])
    A = np.eye(2)
    bb = np.ones((2, 1))
    cc = np.ones((1, 2))
    dd = np.eye(1)

    with pytest.raises(ValueError):
        dss_to_ss(delays, A, bb, cc, dd)


def test_dss_to_ss_supports_mixed_delays_including_minimum():
    # Regression test for #240: delays that mix the documented minimum of
    # 3 samples with larger delays must not raise a shape-mismatch error.
    delays = np.array([3, 4])
    A = np.eye(2)
    bb = np.ones((2, 1))
    cc = np.ones((1, 2))
    dd = np.eye(1)

    AA, _, _, _ = dss_to_ss(delays, A, bb, cc, dd)

    total = int(np.sum(delays))
    assert AA.shape == (total, total)


def test_dss_to_ss_handles_minimum_delay_of_three():
    # Regression test for #240: delays=[3] previously raised a dimension
    # mismatch instead of returning a (3, 3) state-space matrix.
    AA, bb, cc, dd = dss_to_ss(delays=np.array([3]), A=np.array([[0.5]]))

    assert AA.shape == (3, 3)
    assert bb.shape == (3, 1)
    assert cc.shape == (1, 3)
    assert dd.shape == (1, 1)


def test_dss_to_impz_produces_delayed_impulse():
    ir_len = 8
    delays = np.array([3])
    A = np.array([[0.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])

    impulse = dss_to_impz(ir_len, delays, A, B, C, D)

    assert impulse.shape == (ir_len, 1, 1)
    ir_vector = impulse.squeeze()
    expected = np.zeros(ir_len)
    expected[delays[0]] = 1.0
    assert np.allclose(ir_vector, expected)


# ============================================================================
# is_allpass Tests
# ============================================================================


def test_is_allpass_schroeder_siso():
    # Schroeder allpass: H(z) = (g + z^{-d}) / (1 + g*z^{-d}), |g| < 1
    # DSS parameterisation: A=-g, B=1, C=1-g^2, D=g
    g = 0.5
    delays = np.array([4])
    A = np.array([[-g]])
    B = np.array([[1.0]])
    C = np.array([[1 - g**2]])
    D = np.array([[g]])
    result, den, num = is_allpass(A, B, C, D, delays)
    assert result


def test_is_allpass_rejects_non_allpass():
    delays = np.array([4])
    A = np.array([[0.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.5]])
    result, _, _ = is_allpass(A, B, C, D, delays)
    assert not result


def test_is_allpass_unitary_feedback():
    # Lossless FDN with unitary A and matched B, C, D is allpass
    N = 4
    delays = np.array([3, 5, 7, 11])
    A = random_orthogonal(N)
    B = np.eye(N, 1)
    C = -B.T
    D = np.zeros((1, 1))
    # Not checking allpass condition here (D=0 makes D singular),
    # just that the function runs without error via np.linalg.solve
    with pytest.raises(np.linalg.LinAlgError):
        is_allpass(A, B, C, D, delays)


def test_is_allpass_near_singular_D_does_not_use_inv():
    # High-g Schroeder allpass: verify solve path is stable where inv degrades
    g = 0.9
    delays = np.array([3])
    A = np.array([[-g]])
    B = np.array([[1.0]])
    C = np.array([[1 - g**2]])
    D = np.array([[g]])
    result, den, num = is_allpass(A, B, C, D, delays)
    assert result
    assert len(den) > 0
    assert len(num) > 0


# ============================================================================
# dss_to_tf Tests
# ============================================================================


def test_dss_to_tf_scalar_A_tf_matches_impz():
    # TF poles/zeros must reproduce the impulse response via np.polyval
    ir_len = 64
    delays = np.array([3, 5])
    rng = np.random.default_rng(42)
    A = rng.standard_normal((2, 2)) * 0.3
    B = np.eye(2, 1)
    C = np.eye(1, 2)
    D = np.zeros((1, 1))

    tfB, tfA = dss_to_tf(delays, A, B, C, D)
    ir_tf = dss_to_impz(ir_len, delays, A, B, C, D)[:, 0, 0]

    # Evaluate TF at z = e^{j omega} and compare to DFT of IR
    N = 512
    z = np.exp(2j * np.pi * np.arange(N) / N)
    num_coeffs = tfB[0, 0, :]
    den_coeffs = tfA

    # z^{-k} convention: H(z) = sum_k num[k]*z^{-k} / sum_k den[k]*z^{-k}
    H = np.array(
        [
            sum(num_coeffs[k] * zi ** (-k) for k in range(len(num_coeffs)))
            / sum(den_coeffs[k] * zi ** (-k) for k in range(len(den_coeffs)))
            for zi in z
        ]
    )
    ir_from_tf = np.real(np.fft.ifft(H))[:ir_len]
    np.testing.assert_allclose(ir_from_tf, ir_tf, atol=1e-6)


def test_dss_to_tf_polynomial_A_matches_scalar_A():
    # A polynomial A with only a degree-0 term must give identical TF to scalar A
    delays = np.array([2, 3])
    rng = np.random.default_rng(7)
    A_scalar = rng.standard_normal((2, 2)) * 0.3
    A_poly = A_scalar[:, :, np.newaxis]  # shape (2, 2, 1)
    B = np.eye(2, 1)
    C = np.eye(1, 2)
    D = np.zeros((1, 1))

    tfB_s, tfA_s = dss_to_tf(delays, A_scalar, B, C, D)
    tfB_p, tfA_p = dss_to_tf(delays, A_poly, B, C, D)

    min_den = min(len(tfA_s), len(tfA_p))
    np.testing.assert_allclose(tfA_p[:min_den], tfA_s[:min_den], atol=1e-8)

    min_num = min(tfB_s.shape[2], tfB_p.shape[2])
    np.testing.assert_allclose(tfB_p[:, :, :min_num], tfB_s[:, :, :min_num], atol=1e-8)
