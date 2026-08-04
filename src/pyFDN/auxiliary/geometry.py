"""Geometry based helper functions"""

import numpy as np
from typing import Optional, Tuple, List, Union
from numpy.typing import NDArray, ArrayLike
from collections import deque
from dataclasses import dataclass


@dataclass
class Aperture:
    """Class that defines an aperture connecting any two rooms"""
    poly_points: NDArray  # polygon of shape (4, 3)
    room_a: int  # index of room A
    room_b: int  #index of room B

    @property
    def area(self):
        return get_plane_area(self.poly_points)

    @property
    def centroid(self):
        return np.mean(self.poly_points, axis=0)


def get_plane_area(points: List):
    """Get area from quadrilateral points posA, posB, posC, posD"""
    a, b, c, d = map(np.array, points)

    # Split into two triangles: ABC and ACD
    area1 = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    area2 = 0.5 * np.linalg.norm(np.cross(c - a, d - a))

    return area1 + area2


def get_polygon_normal(points: ArrayLike,
                       reference_point: Optional[ArrayLike] = None,
                       eps: float = 1e-12) -> NDArray:
    """Return the unit normal of a planar polygon.

    If `reference_point` is given, the normal is flipped when needed so it
    points toward that reference.

    Args:
        points: Planar polygon coordinates with shape (N, 3), where N >= 3.
        reference_point: Optional point used to orient the returned normal.
        eps: Tolerance for detecting degenerate polygons.

    Returns:
        Unit-length polygon normal with shape (3,).
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError("points must have shape (N, 3) with N >= 3")

    normal = np.zeros(3, dtype=np.float64)
    for idx in range(points.shape[0]):
        current = points[idx]
        next_point = points[(idx + 1) % points.shape[0]]
        normal += np.cross(current, next_point)

    norm = np.linalg.norm(normal)
    if norm <= eps:
        raise ValueError("points define a degenerate polygon")

    normal /= norm
    if reference_point is not None:
        reference_direction = (np.asarray(reference_point, dtype=np.float64) -
                               np.mean(points, axis=0))
        if np.dot(normal, reference_direction) < 0:
            normal = -normal

    return normal


def _triangle_quadrature_points(triangle: NDArray, order: int):
    """Return Gauss-Legendre quadrature points for one triangle.
    Gauss-Legendre quadrature sampling is a mathematical method used to select 
    the optimal points and weights for numerically calculating a definite integral.

    Args:
        triangle: Triangle coordinates with shape (3, 3).
        order: Gauss-Legendre order per reference-triangle dimension.

    Returns:
        Tuple containing integration points with shape (N, 3) and matching
        differential area weights with shape (N,).
    """
    if order < 1:
        raise ValueError("order must be >= 1")

    triangle = np.asarray(triangle, dtype=np.float64)
    if triangle.shape != (3, 3):
        raise ValueError("triangle must have shape (3, 3)")

    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    area_scale = np.linalg.norm(
        np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
    points = []
    area_weights = []

    for u_idx, u in enumerate(nodes):
        for v_idx, v in enumerate(nodes):
            point = ((1.0 - u) * triangle[0] + u * (1.0 - v) * triangle[1] +
                     u * v * triangle[2])
            points.append(point)
            area_weights.append(weights[u_idx] * weights[v_idx] * area_scale *
                                u)

    return np.asarray(points), np.asarray(area_weights)


def aperture_quadrature_points(aperture_coords: ArrayLike,
                               order: int = 8) -> Tuple[NDArray, NDArray]:
    """Return quadrature points and area weights for an aperture.

    Args:
        aperture_coords: Quadrilateral aperture coordinates with shape (4, 3).
        order: Gauss-Legendre order per reference-triangle dimension.

    Returns:
        Tuple containing integration points with shape (N, 3) and matching
        differential area weights with shape (N,).
    """
    aperture_coords = np.asarray(aperture_coords, dtype=np.float64)
    if aperture_coords.shape != (4, 3):
        raise ValueError("aperture_coords must have shape (4, 3)")

    points = []
    weights = []
    for triangle in triangulate_quad(aperture_coords):
        triangle_points, triangle_weights = _triangle_quadrature_points(
            triangle, order)
        points.append(triangle_points)
        weights.append(triangle_weights)

    return np.vstack(points), np.concatenate(weights)


def get_aperture_form_factor(aperture_from_obj: Aperture,
                             aperture_to_obj: Aperture,
                             order: int = 8,
                             eps: float = 1e-12) -> float:
    """Calculate the aperture-to-aperture form factor from Eq. (21).

    The apertures are quadrilateral coordinate arrays of shape (4, 3). Normals
    are oriented automatically toward the other aperture centroid, so vertex
    winding does not affect the result.

    Args:
        aperture_from: Source aperture coordinates with shape (4, 3).
        aperture_to: Receiving aperture coordinates with shape (4, 3).
        order: Gauss-Legendre order per reference-triangle dimension.
        eps: Tolerance for zero-area and coincident-point checks.

    Returns:
        Form factor from `aperture_from` to `aperture_to`.
    """
    aperture_from = np.asarray(aperture_from_obj.poly_points, dtype=np.float64)
    aperture_to = np.asarray(aperture_to_obj.poly_points, dtype=np.float64)
    if aperture_from.shape != (4, 3) or aperture_to.shape != (4, 3):
        raise ValueError("apertures must both have shape (4, 3)")

    area_from = aperture_from_obj.area
    if area_from <= eps:
        raise ValueError("aperture_from has zero area")
    if get_plane_area(aperture_to) <= eps:
        raise ValueError("aperture_to has zero area")

    centroid_from = aperture_from_obj.centroid
    centroid_to = aperture_to_obj.centroid
    normal_from = get_polygon_normal(aperture_from, centroid_to, eps=eps)
    normal_to = get_polygon_normal(aperture_to, centroid_from, eps=eps)

    points_from, weights_from = aperture_quadrature_points(
        aperture_from, order)
    points_to, weights_to = aperture_quadrature_points(aperture_to, order)

    # solve the integral numerically
    integral = 0.0
    for point_from, weight_from in zip(points_from, weights_from):
        vectors = points_to - point_from[None, :]
        distances_squared = np.einsum("ij,ij->i", vectors, vectors)
        valid = distances_squared > eps
        if not np.any(valid):
            continue

        vectors = vectors[valid]
        distances_squared = distances_squared[valid]
        distances = np.sqrt(distances_squared)
        directions = vectors / distances[:, None]

        cos_from = directions @ normal_from
        cos_to = (-directions) @ normal_to
        kernel = np.maximum(cos_from, 0.0) * np.maximum(
            cos_to, 0.0) / distances_squared
        integral += weight_from * np.sum(weights_to[valid] * kernel)

    return float(integral / (np.pi * area_from))


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


def find_room(ROOM_DIMS: List, ROOM_START: List, point: ArrayLike) -> int:
    """
    Find which room a point is in.
    Args:
        ROOM_DIMS (List): list of K room dimensions, in order
        ROOM_START (List): list of K room start coordinates
        point (ArrayLike): 3D point in space
    Returns:
        int: index of room where point lies

    """
    in_rooms = [
        point_in_room(ROOM_START[k], ROOM_DIMS[k], point)
        for k in range(len(ROOM_DIMS))
    ]
    point_in_which_room = in_rooms.index(True) if True in in_rooms else None
    return point_in_which_room


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


def spherical_polygon_halfspaces(poly_pts: NDArray,
                                 viewpoint: NDArray) -> NDArray:
    """
    Project a convex planar polygon onto the unit sphere centered at `viewpoint`.
    Returns inward-pointing half-space normals (through the origin of that local
    frame) — one per edge — such that a direction x is inside the polygon's
    cone iff normal dot x >= 0 for all of them.
    """
    dirs = poly_pts - viewpoint[None, :]
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    n = len(dirs)
    normals = np.array(
        [np.cross(dirs[i], dirs[(i + 1) % n]) for i in range(n)])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    centroid = dirs.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    if np.mean(normals @ centroid) < 0:  # fix winding if needed
        dirs = dirs[::-1]
        normals = np.array(
            [np.cross(dirs[i], dirs[(i + 1) % n]) for i in range(n)])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals


def clip_spherical_polygon(poly: NDArray,
                           normal: NDArray,
                           eps: float = 1e-12) -> NDArray:
    """
    Sutherland–Hodgman clip of a convex spherical polygon (ordered unit
    vectors) against the half-space {x : normal dot x >= 0}.
    """
    if len(poly) == 0:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        da, db = normal @ a, normal @ b
        a_in, b_in = da >= -eps, db >= -eps
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = da / (da - db)
            x = a + t * (b - a)
            x /= np.linalg.norm(x)
            out.append(x)
    return np.array(out)


def spherical_polygon_intersection_solid_angle(apertures: List[Aperture],
                                               viewpoint: ArrayLike) -> float:
    """
    Solid angle (steradians) of directions from `viewpoint` along which a
    straight ray threads through EVERY aperture polygon in `apertures`. This
    is done by finding the intersection of spherical polygons from viewpoint to
    each aperture.

    Requires: each aperture convex, and every room
    between consecutive apertures on the path to be convex (true for cuboid rooms),
    so 'inside all cones' (cone from viewpoint to all apertures - aka solid angle
    from viewpoint to all apertures) implies an actually unobstructed straight path.
    Order of `apertures` doesn't matter — it's a genuine set intersection.
    """
    viewpoint = np.asarray(viewpoint, dtype=np.float64)
    dirs0 = apertures[0].poly_points - viewpoint[None, :]
    poly = dirs0 / np.linalg.norm(dirs0, axis=1, keepdims=True)

    for aperture in apertures[1:]:
        half_space_normals = spherical_polygon_halfspaces(
            aperture.poly_points, viewpoint)
        for normal in half_space_normals:
            poly = clip_spherical_polygon(poly, normal)
            if len(poly) == 0:
                return 0.0  # apertures share no common line-of-sight direction
    if len(poly) < 3:
        return 0.0
    return sum(
        solid_angle_triangle(np.stack([poly[0], poly[i], poly[i + 1]]),
                             np.zeros(3)) for i in range(1,
                                                         len(poly) - 1))


def get_aperture_hit_order(start_room_idx: int, target_room_idx: int,
                           ROOM_DIMS: List, ROOM_START: List,
                           apertures: List[Aperture]):
    """
    Return the order in which apertures are hit when sound propagates from source to receiver
    Args:
        start_room_idx: the room in which the origin point lies
        target_room_idx: the room which is the target room (we need to loop through all rooms)
        ROOM_DIMS (List): list of K room dimensions, in order
        ROOM_START (List): list of K room start coordinates
        apertures (List[NDArray]): list of all apertures in the geometry as polygons
    Returns:
        List[Aperture]: list of aperture polygons in the order that they are hit
    """

    if start_room_idx == target_room_idx:
        return None

    # Build adjacency matrix. Each room is a node in the graph.
    num_rooms = len(ROOM_DIMS)

    adjacency = {idx: [] for idx in range(num_rooms)}
    for aperture in apertures:
        room_a, room_b = (aperture.room_a, aperture.room_b)
        adjacency[room_a].append((room_b, aperture))
        adjacency[room_b].append((room_a, aperture))

    queue = deque([(start_room_idx, [])])
    visited = {start_room_idx}
    while queue:
        # room_idx is an integer, path consists of an ordered list of aperture nodes
        room_idx, path = queue.popleft()
        for next_room_idx, aperture in adjacency[room_idx]:
            if next_room_idx in visited:
                continue
            next_path = path + [aperture]
            if next_room_idx == target_room_idx:
                return next_path
            visited.add(next_room_idx)
            queue.append((next_room_idx, next_path))
    return None


def _next_room_through_aperture(current_room_idx: int,
                                aperture: Aperture) -> int:
    """Return the room reached after crossing an aperture from current_room_idx."""
    if aperture.room_a == current_room_idx:
        return aperture.room_b
    if aperture.room_b == current_room_idx:
        return aperture.room_a
    raise ValueError("aperture path is not connected to current room")


def _chained_aperture_weight(
    aperture_hit_order: List[Aperture],
    point: ArrayLike,
    start_room_idx: int,
    ROOM_DIMS: List,
    apertures: List[Aperture],
) -> float:
    """
    Estimate late-field coupling through an aperture chain.

    A single-aperture path keeps the geometric solid-angle weight. Multi-hop
    paths should not require one straight ray through every aperture; after the
    first aperture, energy is treated as diffuse in each intermediate room.
    This ensures even without a line of sight, a point to room weight is never zero.
    """
    if len(aperture_hit_order) == 1:
        return spherical_polygon_intersection_solid_angle(
            aperture_hit_order, point) / (4 * np.pi)

    first_aperture = aperture_hit_order[0]
    # true geometric weight of first aperture
    weight = solid_angle_quad(first_aperture.poly_points, point) / (4 * np.pi)
    current_room_idx = _next_room_through_aperture(start_room_idx,
                                                   first_aperture)

    for aperture in aperture_hit_order[1:]:
        # total area of all the apertures connected to the current room
        connected_area = sum(candidate.area for candidate in apertures
                             if current_room_idx in (candidate.room_a,
                                                     candidate.room_b))
        if connected_area <= 0:
            connected_area = get_room_surface_area(ROOM_DIMS[current_room_idx])
        # current aperture area / total aperture area
        weight *= aperture.area / connected_area
        current_room_idx = _next_room_through_aperture(current_room_idx,
                                                       aperture)

    return float(weight)


def get_point_to_room_weights(
    point: ArrayLike,
    ROOM_DIMS: List,
    ROOM_START: List,
    apertures: List[Aperture],
    return_delays: bool = False,
    speed_sound: float = 343,
    fs: float = 44100,
) -> Union[Tuple[ArrayLike, ArrayLike], ArrayLike]:
    """
    Get the weights from point to each room in the geometry.
    Args:
        point (Arraylike): 3D point in space
        ROOM_START (List): starting coords for each room in geometry
        ROOM_DIMS (List): dimensions of each room in geometry
        apertures (List[Aperture]): list of apertures in the geometry
        return_delays (bool): if true, also return the propagation delay from
                              point to each room
        fs (float): sampling frequency (needed if return_delays=True)
    Returns:
        Tuple, ArrayLike: either a tuple containing the weights and the delays, 
               or just the weights
    """

    start_room_idx = find_room(ROOM_DIMS, ROOM_START, point)
    num_rooms = len(ROOM_DIMS)
    weights = np.zeros(num_rooms, dtype=np.float32)
    delays = np.zeros(num_rooms, dtype=np.int32)

    # loop over all rooms in topology
    for i in range(num_rooms):
        # for the current target, find the list of apertures that are hit in order
        cur_target_room_idx = i
        if start_room_idx == cur_target_room_idx:
            continue

        aperture_hit_order = get_aperture_hit_order(start_room_idx,
                                                    cur_target_room_idx,
                                                    ROOM_DIMS, ROOM_START,
                                                    apertures)

        if aperture_hit_order is None:
            continue

        # weights[cur_target_room_idx] = _chained_aperture_weight(
        #     aperture_hit_order,
        #     point,
        #     start_room_idx,
        #     ROOM_DIMS,
        #     apertures,
        # )

        weights[
            cur_target_room_idx] = spherical_polygon_intersection_solid_angle(
                aperture_hit_order, point) / (4 * np.pi)

        distance_traversed = 0.0
        cur_point = point.copy()
        for aperture in aperture_hit_order:
            distance_traversed += np.linalg.norm(aperture.centroid - cur_point)
            cur_point = aperture.centroid
        delays[cur_target_room_idx] = int(
            np.round(distance_traversed / speed_sound * fs))

    weights[start_room_idx] = 1 - np.sum(weights)

    if return_delays:
        return weights, delays
    else:
        return weights
