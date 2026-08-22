"""Tests for the pyFDN.eq designs."""

from __future__ import annotations

import numpy as np
import pytest

import pyFDN
from pyFDN.auxiliary.utils import hertz_to_rad
from pyFDN.eq import (
    absorption_geq,
    bandpass_filter,
    design_geq,
    geq_sos,
    probe_sos,
    shelving_filter,
)
from pyFDN.eq.graphic_eq import _geq_sections


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


def test_geq_shape(geq_setup):
    center_omega, shelving_omega, R, _ = geq_setup
    sos = _geq_sections(center_omega, shelving_omega, R, np.zeros(11))
    assert sos.shape == (11, 6)


def test_geq_zero_gains_flat(geq_setup):
    """Zero dB command gains should give a flat (all-pass) response."""
    center_omega, shelving_omega, R, fs = geq_setup
    sos = _geq_sections(center_omega, shelving_omega, R, np.zeros(11))
    ctrl = np.linspace(200, 8000, 20)
    G, _, _ = probe_sos(sos, ctrl, 2**14, fs)
    # Each section at 0 dB should contribute ≈ 0 dB
    np.testing.assert_allclose(G.sum(axis=1), np.zeros(len(ctrl)), atol=0.5)


def test_geq_wrong_gain_length(geq_setup):
    center_omega, shelving_omega, R, _ = geq_setup
    with pytest.raises(ValueError):
        _geq_sections(center_omega, shelving_omega, R, np.zeros(9))


def test_probe_sos_shapes(geq_setup):
    center_omega, shelving_omega, R, fs = geq_setup
    sos = _geq_sections(center_omega, shelving_omega, R, np.zeros(11))
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


def test_geq_designs_a_batch_of_gains_at_once():
    """A trailing axis on the gains is a bank of EQs, designed together."""
    fs = 48000.0
    center_omega = hertz_to_rad(
        np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000.0]), fs
    )
    shelving_omega = hertz_to_rad(np.array([46.0, 11360.0]), fs)
    gains = np.stack([np.linspace(-6, 6, 11), np.zeros(11)], axis=1)

    banked = _geq_sections(center_omega, shelving_omega, 2.7, gains)
    assert banked.shape == (11, 6, 2)
    for channel in range(gains.shape[1]):
        np.testing.assert_allclose(
            banked[:, :, channel],
            _geq_sections(center_omega, shelving_omega, 2.7, gains[:, channel]),
        )


# ------------------------------------------------------ target-to-EQ API ---


@pytest.mark.parametrize(
    ("design", "n_parameters", "n_sections"),
    [
        ("graphic_eq", 10, 11),
        ("first_order_shelf", 2, 1),
        ("one_pole", 2, 1),
    ],
)
def test_gain_designs_have_one_contract(design, n_parameters, n_sections):
    fs = 48000.0
    gain = np.linspace(-6.0, -1.0, n_parameters)
    if design == "graphic_eq":
        sos = pyFDN.gain_to_geq(gain, fs)
    elif design == "first_order_shelf":
        sos = pyFDN.gain_to_first_order_shelf(gain[0], gain[1], 3000.0, fs)
    else:
        sos = pyFDN.gain_to_one_pole(gain[0], gain[1])
    assert sos.shape == (n_sections, 6)
    np.testing.assert_allclose(sos[:, 3], 1.0, atol=1e-12)


@pytest.mark.parametrize(
    ("design_fn", "args"),
    [
        (pyFDN.gain_to_geq, (np.linspace(-6.0, -1.0, 10), 48000.0)),
        (
            pyFDN.gain_to_first_order_shelf,
            (-6.0, -1.0, 3000.0, 48000.0),
        ),
        (pyFDN.gain_to_one_pole, (-6.0, -1.0)),
    ],
)
def test_gain_designs_run_differentiably_in_torch(design_fn, args):
    torch = pytest.importorskip("torch")
    if design_fn is pyFDN.gain_to_geq:
        tensor_args = [
            torch.tensor(args[0], dtype=torch.float64, requires_grad=True),
            args[1],
        ]
    elif design_fn is pyFDN.gain_to_first_order_shelf:
        tensor_args = [
            torch.tensor(args[0], dtype=torch.float64, requires_grad=True),
            torch.tensor(args[1], dtype=torch.float64, requires_grad=True),
            args[2],
            args[3],
        ]
    else:
        tensor_args = [
            torch.tensor(value, dtype=torch.float64, requires_grad=True)
            for value in args
        ]
    tensor_sos = design_fn(*tensor_args)
    assert torch.all(torch.isfinite(tensor_sos))
    tensor_sos.sum().backward()
    target_args = [value for value in tensor_args if hasattr(value, "grad")]
    assert all(value.grad is not None for value in target_args)
    assert all(torch.all(torch.isfinite(value.grad)) for value in target_args)


def test_decay_and_gain_geq_are_the_same_mapping():
    fs = 48000.0
    rt = np.linspace(2.0, 0.6, 10)
    delays = np.array([809.0, 1153.0])
    gain_db = -60.0 * delays / (rt[:, None] * fs)
    np.testing.assert_allclose(
        pyFDN.decay_to_geq(rt, delays, fs),
        pyFDN.gain_to_geq(gain_db, fs),
        atol=1e-12,
    )


def test_decay_and_gain_first_order_shelf_are_the_same_mapping():
    fs = 48000.0
    delays = np.array([809.0, 1153.0])
    rt_dc, rt_nyquist, crossover = 1.4, 0.7, 3500.0
    np.testing.assert_allclose(
        pyFDN.decay_to_first_order_shelf(rt_dc, rt_nyquist, crossover, delays, fs),
        pyFDN.gain_to_first_order_shelf(
            -60.0 * delays / (rt_dc * fs),
            -60.0 * delays / (rt_nyquist * fs),
            crossover,
            fs,
        ),
        atol=1e-12,
    )


def test_decay_and_gain_one_pole_are_the_same_mapping():
    fs = 48000.0
    delays = np.array([809.0, 1153.0])
    rt_dc, rt_nyquist = 1.4, 0.7
    np.testing.assert_allclose(
        pyFDN.decay_to_one_pole(rt_dc, rt_nyquist, delays, fs),
        pyFDN.gain_to_one_pole(
            -60.0 * delays / (rt_dc * fs),
            -60.0 * delays / (rt_nyquist * fs),
        ),
        atol=1e-12,
    )


def test_geq_rejects_the_wrong_number_of_targets():
    with pytest.raises(ValueError, match="takes 10 gains"):
        pyFDN.gain_to_geq(np.zeros(2), 48000.0)
    with pytest.raises(ValueError, match="expected 10 reverberation times"):
        pyFDN.decay_to_geq(np.zeros(2), [100.0], 48000.0)


@pytest.mark.parametrize(
    "design_fn",
    [pyFDN.decay_to_one_pole, pyFDN.decay_to_first_order_shelf],
)
def test_attenuation_designs_flatten_delays(design_fn):
    fs = 48000.0
    flat = np.array([100.0, 150.0, 200.0, 250.0])
    if design_fn is pyFDN.decay_to_first_order_shelf:
        args = (4.0, 1.0, None)
    else:
        args = (4.0, 1.0)
    np.testing.assert_allclose(
        design_fn(*args, flat.reshape(2, 2), fs),
        design_fn(*args, flat, fs),
    )
    assert design_fn(*args, flat.reshape(1, 4), fs).shape == (1, 6, 4)
