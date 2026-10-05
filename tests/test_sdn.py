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

    ax = sdn.visualize(show=False).axes[0]

    assert ax.get_xlabel() == "x (m)"
    assert ax.get_ylabel() == "y (m)"
    assert ax.get_zlabel() == "z (m, up)"
    assert ax.get_zlim() == (0.0, 5.0)
