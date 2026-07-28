"""Geometry based helper functions"""

import numpy as np
from typing import Optional, Tuple, List, Union
from numpy.typing import NDArray, ArrayLike


def get_plane_area(points: List):
    """Get area from quadrilateral points posA, posB, posC, posD"""
    a, b, c, d = map(np.array, points)

    # Split into two triangles: ABC and ACD
    area1 = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    area2 = 0.5 * np.linalg.norm(np.cross(c - a, d - a))

    return area1 + area2


def get_room_volume(room_dims: List):
    """Get volume of a cuboid room"""
    return room_dims[0] * room_dims[1] * room_dims[2]


def get_room_surface_area(room_dims: List):
    """Get surface area of a cuboid"""
    return 2 * ((room_dims[0] * room_dims[1]) + (room_dims[1] * room_dims[2]) +
                (room_dims[0] * room_dims[2]))


def get_room_absorptive_area(room_dims: List, absorption_coeffs: List):
    """
    Get the absorptive area of a room. 
    Ordering of absorption_coeffs is:
    FLOOR, CEILING, LEFT, RIGHT, FRONT, BACK
    """
    floor_ceiling_abs = (room_dims[0] * room_dims[2]) * (absorption_coeffs[0] +
                                                         absorption_coeffs[1])
    left_right_abs = (room_dims[1] * room_dims[2]) * (absorption_coeffs[2] +
                                                      absorption_coeffs[3])
    front_back_abs = (room_dims[0] * room_dims[1]) * (absorption_coeffs[4] +
                                                      absorption_coeffs[5])
    return floor_ceiling_abs + left_right_abs + front_back_abs


def point_in_room(room_start_coords: List,
                  room_dims: List,
                  point: List,
                  tol=1e-9) -> bool:
    """
    Check if a 3D point lies inside or on the boundary of an axis-aligned cuboid.
    """
    point = np.array(point)
    # minimum corner of room
    p_min = np.array(room_start_coords)
    # maximum corner of room
    p_max = np.array(room_start_coords) + np.array(room_dims)

    # Ensure proper ordering of min/max (in case inputs were swapped)
    lower = np.minimum(p_min, p_max)
    upper = np.maximum(p_min, p_max)

    return np.all(point >= lower - tol) and np.all(point <= upper + tol)


def triangulate_quad(quad: NDArray) -> List[NDArray]:
    """Split one quad into two triangles.
    Args:
        quad (NDArray): quadrilateral containing 4 vertices of shape (4, 3)
    Returns:
        List[NDArray]: list containing 2 triangles, each of shape (3, 3)
    """
    return [
        np.asarray([quad[0], quad[1], quad[2]], dtype=float),
        np.asarray([quad[0], quad[2], quad[3]], dtype=float),
    ]


def solid_angle_triangle(tri: NDArray,
                         x: ArrayLike,
                         eps: float = 1e-12) -> float:
    """Solid angle subtended by point x to triangle tri"""
    assert tri.shape == (3, 3)

    # vectors from point to vertices
    a = tri[0]
    b = tri[1]
    c = tri[2]
    ra = a - x
    rb = b - x
    rc = c - x
    la = np.linalg.norm(ra)
    lb = np.linalg.norm(rb)
    lc = np.linalg.norm(rc)
    # if point is on a vertex or degenerate, solid angle is zero
    if la <= eps or lb <= eps or lc <= eps:
        return 0.0
    triple = np.dot(ra, np.cross(rb, rc))
    denom = (la * lb * lc + np.dot(ra, rb) * lc + np.dot(rb, rc) * la +
             np.dot(rc, ra) * lb)
    # use absolute triple product to return unsigned solid angle
    return float(2.0 * np.arctan2(abs(triple), denom))


def solid_angle_quad(
    quad: NDArray,
    x: ArrayLike,
) -> float:
    """
    Calculate the total solid angly subtended by point x to a patch defined by a quadrilateral.
    Args:
        quad: quadrilateral coordinates of shape 4 x 3
        x : point from which solid angle is subtended
    Returns:
        float: solid angle in steradians
    """
    assert quad.shape == (4, 3)
    tris = triangulate_quad(quad)
    if len(tris) != 2 or tris[0].shape != (3, 3) or tris[1].shape != (3, 3):
        raise ValueError("tris must be list of two arrays with shape (3, 3)")

    tri1, tri2 = tris
    return solid_angle_triangle(tri1, x) + solid_angle_triangle(tri2, x)
