"""Tests for dsp module."""

import numpy as np
import pytest
from scipy.signal import sos2tf

from pyFDN import TFMatrix
from pyFDN import ZScalar
from pyFDN import ZSOS
from pyFDN import ZTF
from pyFDN import FilterMatrix


# ============================================================================
# FilterMatrix Tests
# ============================================================================

def test_from_data_returns_same_instance():
    base = FilterMatrix.from_data(np.eye(2))
    wrapped = FilterMatrix.from_data(base)
    assert wrapped is base


def test_static_matrix_full_multiplication():
    fm = FilterMatrix.from_data(np.array([[1.0, 2.0], [0.0, -1.0]]))
    block = np.array([[1.0, 0.5], [0.0, 1.0]])
    out = fm.filter(block)
    expected = block @ fm.matrix.T  # static case should multiply by transpose of stored matrix
    assert np.allclose(out, expected)


def test_static_diagonal_from_zscalar():
    zsc = ZScalar(np.diag([2.0, 4.0]))
    fm = FilterMatrix.from_data(zsc, is_diagonal=True)
    block = np.ones((3, 2))
    out = fm.filter(block)
    assert np.allclose(out, block * np.array([2.0, 4.0]))


def test_iir_from_ztf_filters_input():
    numerator = np.array([[[1.0, -0.5]]])
    denominator = np.array([[[1.0, -0.25]]])
    ztf = ZTF(numerator, denominator, is_diagonal=True)
    fm = FilterMatrix.from_data(ztf)
    impulse = np.zeros((16, 1))
    impulse[0, 0] = 1.0
    response = fm.filter(impulse)
    assert response.shape == impulse.shape
    assert np.isclose(response[0, 0], 1.0)


def test_iir_from_array_multiple_inputs():
    fir = np.array(
        [
            [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )
    fm = FilterMatrix.from_data(fir)
    block = np.eye(2, dtype=float)
    out = fm.filter(block)
    assert out.shape == (2, 2)


def test_filter_raises_on_channel_mismatch():
    fm = FilterMatrix.from_data(np.eye(2))
    with pytest.raises(ValueError):
        fm.filter(np.ones((4, 3)))


def test_from_data_requires_supported_type():
    with pytest.raises(ValueError):
        FilterMatrix.from_data(None)  # type: ignore[arg-type]


def test_from_data_rejects_mismatched_diagonal_shape():
    with pytest.raises(ValueError):
        FilterMatrix.from_data(np.ones((2, 3)), is_diagonal=True)


def _simple_sos():
    """Single SOS section (n, m, nsos, 6); same convention as test_auxiliary_filters."""
    sos = np.zeros((1, 1, 1, 6), dtype=float)
    sos[0, 0, 0, :3] = [1.0, 0.0, 0.0]
    sos[0, 0, 0, 3:] = [1.0, -0.2, 0.0]
    return sos


def test_iir_from_zsos_filters_input():
    sos = _simple_sos()
    zsos = ZSOS(sos, is_diagonal=True)
    fm = FilterMatrix.from_data(zsos)
    impulse = np.zeros((16, 1))
    impulse[0, 0] = 1.0
    response = fm.filter(impulse)
    assert response.shape == impulse.shape
    assert np.isclose(response[0, 0], 1.0)


def test_iir_from_zsos_matches_ztf():
    """FilterMatrix.from_data(ZSOS(...)) matches FilterMatrix.from_data(ZTF(...)) for same filter."""
    sos = _simple_sos()
    sos_1d = sos[0, 0, :, :]  # shape (1, 6)
    b, a = sos2tf(sos_1d)
    numerator = b.reshape(1, 1, -1)
    denominator = a.reshape(1, 1, -1)
    ztf = ZTF(numerator, denominator, is_diagonal=True)
    fm_ztf = FilterMatrix.from_data(ztf)
    fm_zsos = FilterMatrix.from_data(ZSOS(sos), is_diagonal=True)
    block = np.random.default_rng(42).standard_normal((64, 1))
    out_ztf = fm_ztf.filter(block)
    out_zsos = fm_zsos.filter(block)
    assert np.allclose(out_zsos, out_ztf, rtol=1e-9, atol=1e-12)


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
