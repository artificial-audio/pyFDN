"""Tests for the Scattering Delay Network geometry."""

import numpy as np
import pytest

import pyFDN
from pyFDN.generate.sdn import SDN, _cuboid_walls


def test_cuboid_walls_use_z_up_coordinates():
    walls = _cuboid_walls(7.0, 9.0, 5.0)

    assert walls == [
        (0, 0, 1, 0),
        (0, 0, 1, -5.0),
        (1, 0, 0, 0),
        (-1, 0, 0, 7.0),
        (0, -1, 0, 9.0),
        (0, 1, 0, 0),
    ]


def test_wall_nodes_follow_z_up_wall_order():
    sdn = SDN(
        room_size=(7.0, 9.0, 5.0),
        source_pos=(2.1, 4.5, 4.0),
        receiver_pos=(2.8, 0.9, 2.0),
    )

    nodes = sdn.node_positions

    np.testing.assert_allclose(nodes[0, 2], 0.0)
    np.testing.assert_allclose(nodes[1, 2], 5.0)
    np.testing.assert_allclose(nodes[2, 0], 0.0)
    np.testing.assert_allclose(nodes[3, 0], 7.0)
    np.testing.assert_allclose(nodes[4, 1], 9.0)
    np.testing.assert_allclose(nodes[5, 1], 0.0)


def test_visualization_marks_z_as_up():
    sdn = SDN(
        room_size=(7.0, 9.0, 5.0),
        source_pos=(2.1, 4.5, 4.0),
        receiver_pos=(2.8, 0.9, 2.0),
    )

    ax = pyFDN.plot_sdn(sdn).axes[0]

    assert ax.get_xlabel() == "x (m)"
    assert ax.get_ylabel() == "y (m)"
    assert ax.get_zlabel() == "z (m, up)"
    assert ax.get_zlim() == (0.0, 5.0)


def _room(**kwargs):
    return SDN((7.0, 9.0, 5.0), (2.1, 4.5, 4.0), (2.8, 0.9, 2.0), **kwargs)


def test_fields_are_in_samples_and_pyfdn_layouts():
    sdn = _room(fs=48_000.0)
    assert sdn.delays.shape == (30,) and sdn.delays.dtype.kind == "i"
    assert sdn.A.shape == (30, 30) and sdn.B.shape == (30, 6) and sdn.C.shape == (6, 30)
    assert sdn.D.shape == (1, 1)
    # isotropic scattering after a permutation: lossless
    np.testing.assert_allclose(sdn.A @ sdn.A.T, np.eye(30), atol=1e-12)
    np.testing.assert_array_equal(sdn.wall_filters[0, [0, 3]], 1.0)  # pass-through
    assert sdn.direct_delay == round(48_000 * np.linalg.norm([0.7, 3.6, 2.0]) / 343)


def test_one_wall_filter_per_wall_filters_the_lines_that_wall_sends_into():
    bank = np.zeros((1, 6, 6))
    bank[0, 0] = np.arange(6)  # b0 = wall index, to tell the walls apart
    bank[0, 3] = 1.0
    sdn = _room(wall_filters=bank)
    np.testing.assert_array_equal(sdn.wall_filters[0, 0], [i for i, _ in sdn.routing])
    with pytest.raises(ValueError, match="sections, 6, 6"):
        _room(wall_filters=np.zeros((1, 6, 5)))


def test_flamo_render_starts_with_the_direct_path():
    pytest.importorskip("flamo")
    absorbing = np.zeros((1, 6, 6))
    absorbing[0, 0], absorbing[0, 3] = 0.7, 1.0  # lossless walls ring forever
    sdn = _room(fs=16_000.0, wall_filters=absorbing)
    ir = pyFDN.flamo_time_response(pyFDN.sdn_to_flamo(sdn, nfft=2**12, device="cpu"))
    ir = ir.reshape(-1)
    # only the wrapped-around tail before the direct sound (circular convolution)
    np.testing.assert_allclose(ir[: sdn.direct_delay], 0.0, atol=1e-5)
    np.testing.assert_allclose(ir[sdn.direct_delay], sdn.D[0, 0], rtol=1e-4)
