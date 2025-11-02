"""Tests for the ``ZSOS`` filter wrapper."""

import numpy as np

from pyFDN.auxiliary.zsos import ZSOS


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
