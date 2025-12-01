"""Tests for auxiliary.filters module."""

import numpy as np
import pytest

from pyFDN.auxiliary.filters import TFMatrix
from pyFDN.auxiliary.filters import ZFilter
from pyFDN.auxiliary.filters import ZFIR
from pyFDN.auxiliary.filters import ZScalar
from pyFDN.auxiliary.filters import ZSOS
from pyFDN.auxiliary.filters import ZTF


# ============================================================================
# ZSOS Tests
# ============================================================================

def _simple_sos():
    # Single SOS section with a mild pole/zero pair
    sos = np.zeros((1, 1, 1, 6), dtype=float)
    sos[0, 0, 0, :3] = [1.0, 0.0, 0.0]
    sos[0, 0, 0, 3:] = [1.0, -0.2, 0.0]
    return sos


def test_zsos_evaluation_and_derivative():
    sos = _simple_sos()
    filt = ZSOS(sos)
    val = filt.at(1.0)
    der = filt.der(1.0)
    assert np.allclose(val, np.array([[1.25]]))
    assert der.shape == (1, 1)


def test_zsos_inverse_swaps_sections():
    sos = _simple_sos()
    filt = ZSOS(sos)
    inv = filt.inverse()
    assert np.allclose(inv.at(1.0), np.array([[0.8]]))


def test_zsos_dfilt_metadata():
    sos = _simple_sos()
    filt = ZSOS(sos)
    assert filt.dfilt_type() == "df2sos"
    params = filt.dfilt_parameter(0, 0)
    assert "sos" in params
    assert params["sos"].shape == (1, 6)


# ============================================================================
# ZFilter Structure Tests
# ============================================================================

def test_convert2zfilter_returns_zscalar_for_static_matrices():
    matrix = np.array([[1.0, 0.0], [0.5, -0.25]])
    zf = ZFilter.from_any(matrix)
    assert isinstance(zf, ZScalar)
    assert np.allclose(zf.at(1.0), matrix)


def test_convert2zfilter_returns_zfir_for_polynomial_data():
    coeffs = np.zeros((1, 1, 3))
    coeffs[0, 0, :] = [1.0, 0.5, -0.25]

    zf = ZFilter.from_any(coeffs)
    assert isinstance(zf, ZFIR)
    expected = (coeffs[:, :, 0] + coeffs[:, :, 1] + coeffs[:, :, 2]).reshape(-1, 1)
    assert np.allclose(zf.at(1.0), expected)


def test_convert2zfilter_round_trips_zfilter_instance():
    numerator = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    denominator = np.ones_like(numerator)
    ztf = ZTF(numerator, denominator)

    converted = ZFilter.from_any(ztf)
    assert converted is ztf


def test_convert2zfilter_rejects_unknown_types():
    with pytest.raises(TypeError):
        ZFilter.from_any("not-a-filter")


def test_ztf_matches_matrix_polyval():
    numerator = np.array([[[1.0, -0.5]]])
    denominator = np.array([[[1.0, -0.25]]])
    ztf = ZTF(numerator, denominator, is_diagonal=True)

    value = ztf.at(1.0)
    expected = (1.0 - 0.5) / (1.0 - 0.25)
    assert np.allclose(np.diag(value), expected)


def test_ztf_inverse_swaps_polynomials():
    numerator = np.zeros((2, 1, 2))
    numerator[:, 0, 0] = [1.0, 0.9]
    numerator[:, 0, 1] = [0.2, 0.1]
    denominator = np.zeros_like(numerator)
    denominator[:, 0, 0] = [2.0, 1.5]
    denominator[:, 0, 1] = [0.2, 0.1]
    ztf = ZTF(numerator, denominator, is_diagonal=True)
    inv = ztf.inverse()

    assert isinstance(inv, ZTF)
    val = np.diag(inv.at(1.0))
    expected = (denominator[:, 0, 0] + denominator[:, 0, 1]) / (
        numerator[:, 0, 0] + numerator[:, 0, 1]
    )
    assert np.allclose(val, expected)


def test_zscalar_inverse_handles_diagonal_case():
    gains = np.array([[2.0], [4.0]])
    zscalar = ZScalar(gains, isDiagonal=True)
    inv = zscalar.inverse()

    assert inv.is_diagonal is True
    assert np.allclose(np.diag(inv.at(1.0)), [0.5, 0.25])


def test_zfir_derivative_matches_finite_difference():
    coeffs = np.array([
        [[1.0, -0.5, 0.25]],
    ])
    zfir = ZFIR(coeffs)

    z = 0.9
    value = zfir.at(z)
    deriv = zfir.der(z)

    assert deriv.shape == value.shape
    assert np.all(np.isfinite(deriv))


def test_zfilter_interface_requires_scalar_argument():
    numerator = np.array([[[1.0, 0.0]]])
    denominator = np.array([[[1.0, 0.0]]])
    ztf = ZTF(numerator, denominator)

    with pytest.raises(ValueError):
        ztf.at(np.array([1.0, 0.5]))


# ============================================================================
# TFMatrix Tests
# ============================================================================

def test_tfmatrix_init_and_at():
    num = np.ones((2, 2, 3))
    den = np.ones((2, 2, 3))
    tfm = TFMatrix(num, den, var="z^-1")
    z = 1.0
    result = tfm.at(z)
    assert result.shape == (2, 2)


def test_tfmatrix_poles():
    num = np.ones((2, 2, 3))
    den = np.zeros((2, 2, 3))
    den[..., 0] = 1
    tfm = TFMatrix(num, den)
    poles = tfm.poles()
    assert isinstance(poles, np.ndarray)
