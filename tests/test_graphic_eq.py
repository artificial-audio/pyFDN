"""Tests for the graphicEQ module."""

from __future__ import annotations

import numpy as np
import pytest

from pyFDN.auxiliary.utils import hertz_to_rad
from pyFDN.graphicEQ.absorption_geq import absorption_geq
from pyFDN.graphicEQ.bandpass_filter import bandpass_filter
from pyFDN.graphicEQ.design_geq import design_geq
from pyFDN.graphicEQ.graphic_eq import graphic_eq
from pyFDN.graphicEQ.probe_sos import probe_sos
from pyFDN.graphicEQ.shelving_filter import shelving_filter


@pytest.fixture()
def geq_setup():
    fs = 48000.0
    center_omega = hertz_to_rad(
        np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000.0]), fs
    )
    shelving_omega = hertz_to_rad(np.array([46.0, 11360.0]), fs)
    R = 2.7
    return center_omega, shelving_omega, R, fs


def test_shelving_filter_low_shape():
    b, a = shelving_filter(0.3, 2.0, "low")
    assert b.shape == (3,)
    assert a.shape == (3,)


def test_shelving_filter_high_shape():
    b, a = shelving_filter(0.3, 2.0, "high")
    assert b.shape == (3,)
    assert a.shape == (3,)


def test_shelving_filter_unity_gain():
    b, a = shelving_filter(0.5, 1.0, "low")
    np.testing.assert_allclose(b, a, atol=1e-12)


def test_shelving_filter_invalid_type():
    with pytest.raises(ValueError, match="filter_type"):
        shelving_filter(0.3, 2.0, "band")


def test_bandpass_filter_shape():
    b, a = bandpass_filter(0.5, 2.0, 3.0)
    assert b.shape == (3,)
    assert a.shape == (3,)


def test_bandpass_filter_unity_gain():
    b, a = bandpass_filter(0.5, 1.0, 3.0)
    np.testing.assert_allclose(b, a, atol=1e-12)


def test_graphic_eq_shape(geq_setup):
    center_omega, shelving_omega, R, _ = geq_setup
    sos = graphic_eq(center_omega, shelving_omega, R, np.zeros(11))
    assert sos.shape == (11, 6)


def test_graphic_eq_zero_gains_flat(geq_setup):
    """Zero dB command gains should give a flat (all-pass) response."""
    center_omega, shelving_omega, R, fs = geq_setup
    sos = graphic_eq(center_omega, shelving_omega, R, np.zeros(11))
    ctrl = np.linspace(200, 8000, 20)
    G, _, _ = probe_sos(sos, ctrl, 2**14, fs)
    # Each section at 0 dB should contribute ≈ 0 dB
    np.testing.assert_allclose(G.sum(axis=1), np.zeros(len(ctrl)), atol=0.5)


def test_graphic_eq_wrong_gain_length(geq_setup):
    center_omega, shelving_omega, R, _ = geq_setup
    with pytest.raises(ValueError):
        graphic_eq(center_omega, shelving_omega, R, np.zeros(9))


def test_probe_sos_shapes(geq_setup):
    center_omega, shelving_omega, R, fs = geq_setup
    sos = graphic_eq(center_omega, shelving_omega, R, np.zeros(11))
    ctrl = np.linspace(100, 10000, 30)
    G, H, W = probe_sos(sos, ctrl, 512, fs)
    assert G.shape == (30, 11)
    assert H.shape == (512, 11)
    assert W.shape == (512, 11)


def test_design_geq_shape():
    sos, target_f = design_geq(np.zeros(10))
    assert sos.shape == (11, 6)
    assert target_f.shape == (10,)


def test_design_geq_flat_target():
    """Flat 0 dB target should give ≈ 0 dB total response at all bands."""
    sos, _ = design_geq(np.zeros(10))
    ctrl = np.array([63.0, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
    G, _, _ = probe_sos(sos, ctrl, 2**16, 48000.0)
    total_db = G.sum(axis=1)
    np.testing.assert_allclose(total_db, np.zeros(len(ctrl)), atol=0.5)


def test_design_geq_uniform_target():
    """Uniform -3 dB target should give ≈ -3 dB at all bands."""
    sos, _ = design_geq(np.full(10, -3.0))
    ctrl = np.array([63.0, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
    G, _, _ = probe_sos(sos, ctrl, 2**16, 48000.0)
    total_db = G.sum(axis=1)
    np.testing.assert_allclose(total_db, np.full(len(ctrl), -3.0), atol=0.5)


def test_absorption_geq_shape():
    rt = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.9])
    delays = np.array([1000.0, 1300.0, 1700.0])
    sos = absorption_geq(rt, delays, 48000.0)
    assert sos.shape == (11, 6, 3)


def test_absorption_geq_normalised():
    """All sections should have a₀ = 1 after normalisation."""
    rt = np.ones(10) * 0.5
    delays = np.array([800.0, 1200.0])
    sos = absorption_geq(rt, delays, 48000.0)
    np.testing.assert_allclose(sos[:, 3, :], np.ones((11, 2)), atol=1e-10)
