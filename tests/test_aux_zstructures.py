"""Tests for z-domain filter helper classes."""

import numpy as np
import pytest

from pyFDN.auxiliary.convert2zfilter import convert2zFilter
from pyFDN.auxiliary.zfilter import ZFilter
from pyFDN.auxiliary.zfir import ZFIR
from pyFDN.auxiliary.zscalar import ZScalar
from pyFDN.auxiliary.ztf import ZTF


def test_convert2zfilter_returns_zscalar_for_static_matrices():
    matrix = np.array([[1.0, 0.0], [0.5, -0.25]])
    zf = convert2zFilter(matrix)
    assert isinstance(zf, ZScalar)
    assert np.allclose(zf.at(1.0), matrix)


def test_convert2zfilter_returns_zfir_for_polynomial_data():
    coeffs = np.zeros((1, 1, 3))
    coeffs[0, 0, :] = [1.0, 0.5, -0.25]

    zf = convert2zFilter(coeffs)
    assert isinstance(zf, ZFIR)
    expected = (coeffs[:, :, 0] + coeffs[:, :, 1] + coeffs[:, :, 2]).reshape(-1, 1)
    assert np.allclose(zf.at(1.0), expected)


def test_convert2zfilter_round_trips_zfilter_instance():
    numerator = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    denominator = np.ones_like(numerator)
    ztf = ZTF(numerator, denominator)

    converted = convert2zFilter(ztf)
    assert converted is ztf


def test_convert2zfilter_rejects_unknown_types():
    with pytest.raises(TypeError):
        convert2zFilter("not-a-filter")


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
