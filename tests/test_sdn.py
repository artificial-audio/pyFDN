"""Tests for the Scattering Delay Network geometry."""

import numpy as np

from pyFDN.generate.SDN import SDN, _cuboid_walls


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

    nodes = np.asarray(sdn.compute()["node_positions"])

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

    scene = sdn.visualize(show=False).layout.scene

    assert scene.xaxis.title.text == "x (m)"
    assert scene.yaxis.title.text == "y (m)"
    assert scene.zaxis.title.text == "z (m, up)"
    assert scene.camera.up.x is None
    assert scene.camera.up.y is None
    assert scene.camera.up.z is None
