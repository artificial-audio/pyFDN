"""Tests for the pyFDN.eq designs."""

from __future__ import annotations

import numpy as np
import pytest

import pyFDN
from pyFDN.auxiliary.utils import hertz_to_rad
from pyFDN.eq.absorption_geq import absorption_geq
from pyFDN.eq.bandpass_filter import bandpass_filter
from pyFDN.eq.design_geq import design_geq, geq_sos
from pyFDN.eq.graphic_eq import graphic_eq
from pyFDN.eq.probe_sos import probe_sos
from pyFDN.eq.shelving_filter import shelving_filter


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


def test_geq_sos_matches_design_geq_where_the_bounds_are_slack():
    """The closed form is the same design as the bounded solve, unbounded."""
    target = np.linspace(-6.0, 4.0, 10)
    bounded, _ = design_geq(target, fs=48000.0)
    closed = geq_sos(target, 48000.0)
    ctrl = np.array([63.0, 125, 250, 500, 1000, 2000, 4000, 8000])
    bounded_db = probe_sos(bounded / bounded[:, 3:4], ctrl, 2**16, 48000.0)[0]
    closed_db = probe_sos(closed, ctrl, 2**16, 48000.0)[0]
    np.testing.assert_allclose(bounded_db.sum(axis=1), closed_db.sum(axis=1), atol=0.05)


def test_geq_sos_is_the_same_design_in_torch():
    """One source, two array namespaces: the tensor path must not drift."""
    torch = pytest.importorskip("torch")
    target = np.stack([np.linspace(-6.0, 4.0, 10), np.full(10, -2.0)], axis=1)

    tensor_target = torch.tensor(target, dtype=torch.float64, requires_grad=True)
    tensor_sos = geq_sos(tensor_target, 48000.0)
    assert tensor_sos.shape == (11, 6, 2)

    for channel in range(target.shape[1]):
        np.testing.assert_allclose(
            tensor_sos[:, :, channel].detach().numpy(),
            geq_sos(target[:, channel], 48000.0),
            atol=1e-12,
        )

    tensor_sos.sum().backward()
    assert torch.all(torch.isfinite(tensor_target.grad))
    assert torch.any(tensor_target.grad != 0)


def test_graphic_eq_designs_a_batch_of_gains_at_once():
    """A trailing axis on the gains is a bank of EQs, designed together."""
    fs = 48000.0
    center_omega = hertz_to_rad(
        np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000.0]), fs
    )
    shelving_omega = hertz_to_rad(np.array([46.0, 11360.0]), fs)
    gains = np.stack([np.linspace(-6, 6, 11), np.zeros(11)], axis=1)

    banked = graphic_eq(center_omega, shelving_omega, 2.7, gains)
    assert banked.shape == (11, 6, 2)
    for channel in range(gains.shape[1]):
        np.testing.assert_allclose(
            banked[:, :, channel],
            graphic_eq(center_omega, shelving_omega, 2.7, gains[:, channel]),
        )


# ---------------------------------------------------------------- EQDesign ---


@pytest.mark.parametrize(
    "design",
    [pyFDN.GraphicEQ(), pyFDN.FirstOrderShelf(), pyFDN.OnePole()],
    ids=lambda d: type(d).__name__,
)
def test_every_design_honours_the_same_contract(design):
    """n_params targets in, (n_sections, 6) biquads out, normalized to a0 = 1."""
    fs = 48000.0
    target = np.linspace(-6.0, -1.0, design.n_params)

    sos = design.sos(target, fs, **design.buffers(fs))
    assert sos.shape == (design.n_sections, 6)
    np.testing.assert_allclose(sos[:, 3], 1.0, atol=1e-12)

    banked = design.sos(
        np.stack([target, target * 0.5], axis=1), fs, **design.buffers(fs)
    )
    assert banked.shape == (design.n_sections, 6, 2)
    np.testing.assert_allclose(banked[:, :, 0], sos, atol=1e-12)


@pytest.mark.parametrize(
    "design",
    [pyFDN.GraphicEQ(), pyFDN.FirstOrderShelf(), pyFDN.OnePole()],
    ids=lambda d: type(d).__name__,
)
def test_every_design_runs_in_torch_from_the_same_source(design):
    """One implementation per design, two array namespaces, no drift."""
    torch = pytest.importorskip("torch")
    fs = 48000.0
    target = np.linspace(-6.0, -1.0, design.n_params)

    tensor_target = torch.tensor(target, dtype=torch.float64, requires_grad=True)
    buffers = {
        k: torch.tensor(v, dtype=torch.float64) for k, v in design.buffers(fs).items()
    }
    tensor_sos = design.sos(tensor_target, fs, **buffers)

    np.testing.assert_allclose(
        tensor_sos.detach().numpy(),
        design.sos(target, fs, **design.buffers(fs)),
        atol=1e-12,
    )
    tensor_sos.sum().backward()
    assert torch.all(torch.isfinite(tensor_target.grad))


def test_fit_is_the_solve_and_sos_is_the_map():
    """Only the graphic EQ has a design step its map cannot express exactly."""
    fs = 48000.0
    shelf = pyFDN.FirstOrderShelf()
    endpoints = np.array([-4.0, -9.0])
    # the shelf's parameters are its own endpoints, so fitting is mapping
    np.testing.assert_allclose(
        shelf.fit(endpoints, fs), shelf.sos(endpoints, fs), atol=1e-12
    )

    geq = pyFDN.GraphicEQ()
    target = np.linspace(-6.0, 4.0, geq.n_params)
    # the GEQ's fit is bounded least squares -- a different, unnormalized answer
    fitted = geq.fit(target, fs)
    assert fitted.shape == (geq.n_sections, 6)
    assert not np.allclose(fitted[:, 3], 1.0)


def test_default_design_dispatches_on_length_and_rejects_the_rest():
    assert isinstance(pyFDN.default_design(10), pyFDN.GraphicEQ)
    assert isinstance(pyFDN.default_design(2), pyFDN.FirstOrderShelf)
    assert pyFDN.default_design(2, crossover_frequency=2000.0).crossover_frequency == (
        2000.0
    )
    with pytest.raises(ValueError, match="no EQ design takes 5"):
        pyFDN.default_design(5)


def test_one_pole_design_matches_the_numpy_absorption_design():
    """OnePole().sos is one_pole_absorption, given the same endpoint gains."""
    fs = 48000.0
    delays = np.array([809.0, 1153.0])
    rt_dc, rt_ny = 1.4, 0.7
    designed = pyFDN.one_pole_absorption(rt_dc, rt_ny, delays, fs)

    slope = np.array([pyFDN.rt_to_slope(rt_dc, fs), pyFDN.rt_to_slope(rt_ny, fs)])
    target_db = slope[:, None] * delays[None, :]
    np.testing.assert_allclose(pyFDN.OnePole().sos(target_db, fs), designed, atol=1e-12)


@pytest.mark.parametrize(
    "design_fn", [pyFDN.one_pole_absorption, pyFDN.first_order_absorption]
)
def test_absorption_designs_flatten_the_delays_they_are_given(design_fn):
    """The SOS bank's channel axis is flat whatever shape `delays` arrives in."""
    fs = 48000.0
    flat = np.array([100.0, 150.0, 200.0, 250.0])
    np.testing.assert_allclose(
        design_fn(4.0, 1.0, flat.reshape(2, 2), fs),
        design_fn(4.0, 1.0, flat, fs),
    )
    assert design_fn(4.0, 1.0, flat.reshape(1, 4), fs).shape == (1, 6, 4)
