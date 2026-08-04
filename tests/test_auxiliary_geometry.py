import numpy as np

from pyFDN.auxiliary.geometry import (
    Aperture,
    aperture_form_factor,
    get_plane_area,
    get_point_to_room_weights,
)


def test_multi_hop_room_weight_uses_diffuse_chain():
    """
    Calculate the weights from a point to all connected rooms.
    Weights should sum to 1
    """
    room_dims = [[3.0, 6.0, 3.0], [8.0, 4.0, 3.0], [8.0, 4.0, 3.0]]
    room_start = [[0, 0, 0], [-2.0, 6.0, 0], [3.0, 0, 0]]
    aperture_12 = Aperture(
        np.asarray(
            [
                [0.75, 6, 0],
                [0.75, 6, 3],
                [2.25, 6, 3],
                [2.25, 6, 0],
            ],
            dtype=float,
        ),
        0,
        1,
    )
    aperture_13 = Aperture(
        np.asarray(
            [
                [3, 0, 0],
                [3, 0, 3],
                [3, 1.5, 3],
                [3, 1.5, 0],
            ],
            dtype=float,
        ),
        0,
        2,
    )

    point_in_room_3 = np.asarray([9.0, 3.5, 1.5])
    weights = get_point_to_room_weights(
        point_in_room_3,
        room_dims,
        room_start,
        [aperture_12, aperture_13],
    )

    assert weights[1] > 0.0
    assert weights[2] > weights[0] > weights[1]
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)


def test_aperture_form_factor_matches_far_field_limit():
    """Test the form factor from Summers paper JASA, equation 21"""
    aperture_from = np.asarray(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ],
        dtype=float,
    )
    aperture_to = aperture_from + np.asarray([0.0, 0.0, 10.0])

    form_factor = aperture_form_factor(aperture_from, aperture_to, order=8)
    far_field = get_plane_area(aperture_to) / (np.pi * 10.0**2)

    np.testing.assert_allclose(form_factor, far_field, rtol=2e-2)


def test_aperture_form_factor_obeys_reciprocity():
    """Form factor should be reciprocal"""
    aperture_small = np.asarray(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ],
        dtype=float,
    )
    aperture_large = np.asarray(
        [
            [-1.0, -1.0, 2.0],
            [1.0, -1.0, 2.0],
            [1.0, 1.0, 2.0],
            [-1.0, 1.0, 2.0],
        ],
        dtype=float,
    )

    forward = aperture_form_factor(aperture_small, aperture_large, order=8)
    reverse = aperture_form_factor(aperture_large, aperture_small, order=8)

    np.testing.assert_allclose(
        get_plane_area(aperture_small) * forward,
        get_plane_area(aperture_large) * reverse,
        rtol=1e-12,
    )
