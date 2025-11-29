"""Tests for ``pyFDN.dsp.filter_matrix``."""

import numpy as np
import pytest

from pyFDN.auxiliary.zscalar import ZScalar
from pyFDN.auxiliary.ztf import ZTF
from pyFDN.dsp.filter_matrix import FilterMatrix


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
