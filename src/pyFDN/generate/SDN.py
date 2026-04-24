#!/usr/bin/env python3
"""
Minimal SDN (Scattering Delay Network) coefficient generator.

Produces only the filter/network parameters for use in an external FDN (e.g. pyFDN):
delay lengths, permutation/routing, scattering matrices, wall filters and attenuations.
No simulation, interpolation, or time-domain processing.
"""

import math

import numpy as np

try:
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:
    go = None
    _HAS_PLOTLY = False

# ---------------------------------------------------------------------------
# Minimal geometry (no dependency on Geometry.py)
# ---------------------------------------------------------------------------


def _point_reflect_plane(px, py, pz, a, b, c, d):
    """Reflect point (px,py,pz) in plane ax+by+cz+d=0."""
    k = -(a * px + b * py + c * pz + d) / (a * a + b * b + c * c)
    return (
        2 * (a * k + px) - px,
        2 * (b * k + py) - py,
        2 * (c * k + pz) - pz,
    )


def _line_plane_intersection(x1, y1, z1, x2, y2, z2, a, b, c, d):
    """Line (x1,y1,z1)->(x2,y2,z2) with plane ax+by+cz+d=0."""
    l, m, n = x2 - x1, y2 - y1, z2 - z1
    den = a * l + b * m + c * n
    if abs(den) < 1e-12:
        return None
    k = -(a * x1 + b * y1 + c * z1 + d) / den
    return (k * l + x1, k * m + y1, k * n + z1)


def _dist(p, q):
    """Euclidean distance between 3-tuples p and q."""
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2)


def _cuboid_walls(Lx, Ly, Lz):  # TODO: fix naming of the walls, see also visualization
    """
    Return list of 6 walls as plane coeffs (a,b,c,d) for ax+by+cz+d=0.
    Order and planes must match Geometry.Cuboid.setWallPosition():
    - floor:  (0,0,0), (Lx,0,0), (0,0,Lz)  -> y=0
    - ceiling: (0,Ly,0), (0,Ly,Lz), (Lx,Ly,0) -> y=Ly
    - left:   x=0,  right: x=Lx
    - front:  (0,0,Lz), (0,Ly,Lz), (Lx,Ly,Lz) -> z=Lz
    - back:   (0,0,0), (0,Ly,0), (Lx,Ly,0) -> z=0
    """
    return [
        (0, 1, 0, 0),  # floor:  y=0
        (0, 1, 0, -Ly),  # ceiling: y=Ly  (y - Ly = 0)
        (1, 0, 0, 0),  # left:   x=0
        (-1, 0, 0, Lx),  # right:  x=Lx
        (0, 0, -1, Lz),  # front:  z=Lz
        (0, 0, 1, 0),  # back:   z=0
    ]


def _wall_node_position(source_pos, mic_pos, wall):
    """
    First-order image source: reflect source in wall plane; node = intersection
    of line (reflected_source, mic) with wall plane.
    wall = (a, b, c, d) for plane ax+by+cz+d=0.
    """
    a, b, c, d = wall
    sx, sy, sz = source_pos
    mx, my, mz = mic_pos
    rx, ry, rz = _point_reflect_plane(sx, sy, sz, a, b, c, d)
    pt = _line_plane_intersection(rx, ry, rz, mx, my, mz, a, b, c, d)
    if pt is None:
        # fallback: use midpoint
        pt = ((rx + mx) * 0.5, (ry + my) * 0.5, (rz + mz) * 0.5)
    return (round(pt[0], 5), round(pt[1], 5), round(pt[2], 5))


def _isotropic_scattering_matrix(N):
    """Isotropic scattering matrix: S[i,j] = 2/N - delta_ij."""
    S = np.full((N, N), 2.0 / N, dtype=np.float64)
    np.fill_diagonal(S, 2.0 / N - 1.0)
    return S


# ---------------------------------------------------------------------------
# SDN coefficient generator
# ---------------------------------------------------------------------------


class SDN:
    """
    Minimal SDN: from room and source/receiver, compute only network parameters
    (delay lengths, routing, scattering matrices, wall filters) for use in an FDN.
    """

    N_WALLS = 6
    DEFAULT_C = 343.0

    def __init__(
        self,
        room_size,
        source_pos,
        receiver_pos,
        Fs=44100,
        c=None,
        wall_filters=None,
    ):
        """
        Parameters
        ----------
        room_size : tuple (Lx, Ly, Lz)
            Cuboid room dimensions in metres.
        source_pos : tuple (x, y, z)
            Source position in metres.
        receiver_pos : tuple (x, y, z)
            Receiver (microphone) position in metres.
        Fs : int or float
            Sampling frequency in Hz.
        c : float, optional
            Speed of sound in m/s (default 343).
        wall_filters : list of 6 lists of 5 SOS arrays, optional
            Per-wall, per-port filters. Each filter must be an SOS array of shape
            (n_sections, 6) as returned by ``scipy.signal`` design functions such as
            ``butter(..., output='sos')`` or ``cheby1(..., output='sos')``.
            Must have exactly 6 walls, each with exactly 5 port filters. If None
            (default), identity (pass-through) filters are used for all walls and ports.
        """
        self.room_size = tuple(room_size)
        self.source_pos = tuple(source_pos)
        self.receiver_pos = tuple(receiver_pos)
        self.Fs = float(Fs)
        self.c = float(c if c is not None else self.DEFAULT_C)

        self.wall_filters = wall_filters

        self._walls = _cuboid_walls(*self.room_size)
        self._node_positions = None
        self._result = None

    def compute(self):
        """
        Compute all SDN parameters. Call once after construction.

        Returns
        -------
        result : dict
            - delay_lengths : np.ndarray shape (6,6), float, seconds
                delay_lengths[i,j] = delay from node i to node j. Diagonal is NaN; only i != j valid.
            - delay_lengths_flat : np.ndarray shape (30,), seconds
                Wall-to-wall delays in order (0,1),(0,2),...,(5,4).
            - routing : list of (from_node, to_node)
                routing[k] = (i, j) means delay line k goes from node i to node j.
            - permutation_matrix : np.ndarray shape (30,30)
                Connectivity: permutation_matrix[k_in, k_out]=1 if output of delay k_out feeds delay k_in.
            - scattering_matrices : list of 6 arrays shape (5,5)
                Isotropic scattering matrix for each wall node.
            - wall_filters_sos : np.ndarray shape (n_sections, 6, 30), SOS coefficients in delay order for FLAMO.
            - source_to_wall_delays : np.ndarray shape (6,), seconds
            - source_to_wall_gains : np.ndarray shape (6,), 1/r gain
            - wall_to_receiver_delays : np.ndarray shape (6,), seconds
            - wall_to_receiver_gains : np.ndarray shape (6,), 1/(1 + d_node_mic/d_source_node)
            - direct_path_delay : float, seconds
            - direct_path_gain : float
            - node_positions : list of 6 (x, y, z) tuples, wall node positions in metres.
            - Fs : float, sampling frequency in Hz.
            - c : float, speed of sound in m/s.

            Input routing (FDN): 6 gains -> 6 delays -> 6-to-30 matrix (0.5 in matrix).

            - input_delays : np.ndarray shape (6,), seconds
            - input_gains : np.ndarray shape (6,), gain per node (source_to_wall_gains, 1/r).
            - input_matrix : np.ndarray shape (30, 6), input_matrix[k, j] = 0.5 if delay k leaves node j else 0.
              Use: x = input_gains * input; x = delay(x, input_delays); injection = input_matrix @ x.

            Output routing (FDN): 30-to-6 matrix -> 6 delays -> 6 gains -> sum.

            - output_matrix : np.ndarray shape (6, 30), output_matrix[j, k] = (2/5) if delay k leaves node j else 0.
            - output_gains : np.ndarray shape (6,), wall_to_receiver_gains.
            - output_delays : np.ndarray shape (6,), seconds
              Use: y = output_matrix @ state_30; y = delay(y, output_delays); output_reflected = sum(output_gains * y).
            - output_node_to_delay_indices : list of 6 lists; output_node_to_delay_indices[j] = delay indices leaving node j.
        """
        Lx, Ly, Lz = self.room_size
        walls = _cuboid_walls(Lx, Ly, Lz)
        nodes = [
            _wall_node_position(self.source_pos, self.receiver_pos, w) for w in walls
        ]
        self._node_positions = nodes

        sp = self.source_pos
        rp = self.receiver_pos
        _eps = 1e-9  # avoid division by zero in gains when nodes coincide
        # Minimum distance so delay is at least 1 sample (avoids delay-free loops when two wall nodes coincide, e.g. at room edges)
        _d_min_one_sample = self.c / self.Fs

        # Wall-to-wall delays (6*5 = 30)
        delay_lengths = np.zeros((self.N_WALLS, self.N_WALLS), dtype=float)
        delay_lengths.fill(np.nan)
        for i in range(self.N_WALLS):
            for j in range(self.N_WALLS):
                if i == j:
                    continue
                d = max(_dist(nodes[i], nodes[j]), _d_min_one_sample)
                delay_lengths[i, j] = round(self.Fs * d / self.c)

        # Order of delay lines: (0,1),(0,2),...,(0,5),(1,0),(1,2),...,(5,4)
        routing = [
            (i, j) for i in range(self.N_WALLS) for j in range(self.N_WALLS) if i != j
        ]
        # All delays in seconds for FLAMO (no conversion in delay_module)
        delay_lengths_s = delay_lengths.astype(float) / self.Fs
        delay_lengths_flat = np.array(
            [delay_lengths_s[i, j] for (i, j) in routing], dtype=float
        )

        # Scattering: 5x5 isotropic per node
        scattering_matrices = [
            _isotropic_scattering_matrix(5) for _ in range(self.N_WALLS)
        ]

        # Wall filters: list of 6 lists of 5; each element is either (b, a) or SOS array (n_sec, 6)
        # (handled below when building wall_filters_sos)

        # Source to wall (delays in seconds)
        source_to_wall_delays = np.array(
            [_dist(sp, nodes[i]) / self.c for i in range(self.N_WALLS)], dtype=float
        )
        source_to_wall_gains = np.array(
            [
                (self.c / self.Fs) / max(_dist(sp, nodes[i]), _eps)
                for i in range(self.N_WALLS)
            ],
            dtype=float,
        )

        # Wall to receiver (delays in seconds; gain 1/(1 + d_node_mic/d_source_node) as in Simulation)
        dist_source_node = np.array(
            [_dist(sp, nodes[i]) for i in range(self.N_WALLS)], dtype=float
        )
        dist_node_mic = np.array(
            [_dist(nodes[i], rp) for i in range(self.N_WALLS)], dtype=float
        )
        wall_to_receiver_delays = np.array(
            [dist_node_mic[i] / self.c for i in range(self.N_WALLS)], dtype=float
        )
        wall_to_receiver_gains = 1.0 / (
            1.0 + dist_node_mic / np.maximum(dist_source_node, _eps)
        )

        # Direct path (delay in seconds)
        direct_path_delay = _dist(sp, rp) / self.c
        direct_path_gain = (self.c / self.Fs) / max(_dist(sp, rp), _eps)

        # Permutation matrix P: P[k_in, k_out]=1 if delay k_out feeds into delay k_in (connectivity only)
        permutation_matrix = _build_fdn_permutation_from_routing(routing)

        # ---- Input routing for FDN: 6 delays -> 6 gains -> 6-to-30 matrix (0.5 in matrix) ----
        input_delays = source_to_wall_delays  # (6,) seconds
        input_gains = np.asarray(
            source_to_wall_gains, dtype=float
        )  # (6,) gain per node (1/r, no 0.5 here)
        input_matrix = np.zeros(
            (30, 6), dtype=float
        )  # 6-to-30: input_matrix[k, j] = 0.5 if delay k leaves node j
        for k, (j, _) in enumerate(routing):  # delay k leaves node j
            input_matrix[k, j] = 0.5
        # Flow: input -> 6 delays -> 6 gains -> (30,6) @ (6,) = 30 injection

        # ---- Output routing for FDN: 30-to-6 matrix -> 6 gains -> 6 delays -> sum ----
        output_matrix = np.zeros(
            (6, 30), dtype=float
        )  # 30-to-6: output_matrix[j, k] = (2/5) if delay k leaves node j
        for k, (j, _) in enumerate(routing):
            output_matrix[j, k] = 2.0 / 5.0
        output_gains = np.asarray(wall_to_receiver_gains, dtype=float)  # (6,)
        output_delays = np.asarray(wall_to_receiver_delays, dtype=float)  # (6,) seconds
        # Flow: 30 state -> (6,30) @ (30,) = 6 -> 6 gains -> 6 delays -> sum

        # Which delay indices leave each node (for per-node aggregation if preferred)
        node_to_delay_indices: list[list[int]] = [[] for _ in range(self.N_WALLS)]
        for k, (j, _) in enumerate(routing):
            node_to_delay_indices[j].append(k)
        for j in range(self.N_WALLS):
            node_to_delay_indices[j].sort()

        # Wall filters in delay order, SOS format (n_sections, 6, 30) for FLAMO
        identity_sos = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        if self.wall_filters is None:
            sos_list = [identity_sos for _ in routing]
        else:
            if len(self.wall_filters) != self.N_WALLS:
                raise ValueError(
                    f"wall_filters must have {self.N_WALLS} walls, got {len(self.wall_filters)}"
                )
            for w_idx, wall_ports in enumerate(self.wall_filters):
                if len(wall_ports) != 5:
                    raise ValueError(
                        f"wall_filters[{w_idx}] must have 5 ports, got {len(wall_ports)}"
                    )
            sos_list = []
            for k in range(len(routing)):
                j_dep, _ = routing[k]
                port = node_to_delay_indices[j_dep].index(k)
                sos = np.asarray(self.wall_filters[j_dep][port], dtype=np.float64)
                if sos.ndim != 2 or sos.shape[1] != 6:
                    raise ValueError(
                        f"wall_filters[{j_dep}][{port}] must be an SOS array of shape "
                        f"(n_sections, 6), got shape {sos.shape}"
                    )
                sos_list.append(sos)
        max_sections = max(s.shape[0] for s in sos_list)
        wall_filters_sos = np.zeros((max_sections, 6, len(routing)), dtype=np.float64)
        for ch in range(len(routing)):
            nsec = sos_list[ch].shape[0]
            wall_filters_sos[:nsec, :, ch] = sos_list[ch]
            for s in range(nsec, max_sections):
                wall_filters_sos[s, :, ch] = [1, 0, 0, 1, 0, 0]

        self._result = {
            "delay_lengths": delay_lengths_s,
            "delay_lengths_flat": delay_lengths_flat,
            "routing": routing,
            "permutation_matrix": permutation_matrix,
            "scattering_matrices": scattering_matrices,
            "wall_filters_sos": wall_filters_sos,
            "source_to_wall_delays": source_to_wall_delays,
            "source_to_wall_gains": source_to_wall_gains,
            "wall_to_receiver_delays": wall_to_receiver_delays,
            "wall_to_receiver_gains": wall_to_receiver_gains,
            "direct_path_delay": float(direct_path_delay),
            "direct_path_gain": float(direct_path_gain),
            "node_positions": nodes,
            "Fs": self.Fs,
            "c": self.c,
            # Input routing (FDN): 6 delays -> 6 gains -> 6-to-30 matrix (0.5 in matrix)
            "input_delays": input_delays,
            "input_gains": input_gains,
            "input_matrix": input_matrix,
            # Output routing (FDN): 30-to-6 matrix -> 6 gains -> 6 delays -> sum
            "output_matrix": output_matrix,
            "output_gains": output_gains,
            "output_delays": output_delays,
            "output_node_to_delay_indices": node_to_delay_indices,
        }
        return self._result

    def visualize(self, show=True, room_alpha=0.08, room_edge_color="gray"):
        """
        Plot the 3D room with source, receiver, and wall node positions (Plotly).
        Call compute() first (or it will be called for you).

        Parameters
        ----------
        show : bool
            If True, call fig.show() at the end.
        room_alpha : float
            Transparency of room faces (0=invisible, 1=opaque).
        room_edge_color : str
            Color of room wireframe edges (e.g. "gray", "rgb(128,128,128)").

        Returns
        -------
        fig : plotly.graph_objects.Figure
        """
        if not _HAS_PLOTLY:
            raise ImportError("visualize() requires plotly (pip install plotly)")
        if self._result is None:
            self.compute()
        r = self._result
        assert r is not None
        Lx, Ly, Lz = self.room_size
        nodes = r["node_positions"]
        sp = self.source_pos
        rp = self.receiver_pos
        wall_labels = ("floor", "ceiling", "left", "right", "front", "back")

        corners = np.array(
            [
                [0, 0, 0],
                [Lx, 0, 0],
                [Lx, Ly, 0],
                [0, Ly, 0],
                [0, 0, Lz],
                [Lx, 0, Lz],
                [Lx, Ly, Lz],
                [0, Ly, Lz],
            ]
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        xe, ye, ze = [], [], []
        for i, j in edges:
            xe.extend([corners[i, 0], corners[j, 0], None])
            ye.extend([corners[i, 1], corners[j, 1], None])
            ze.extend([corners[i, 2], corners[j, 2], None])

        # Room mesh faces (12 triangles for 6 quads)
        tri_i = [0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 4, 4]
        tri_j = [1, 2, 3, 7, 3, 7, 2, 6, 1, 2, 5, 6]
        tri_k = [2, 3, 7, 6, 7, 4, 6, 5, 2, 3, 6, 7]

        node_colors = [
            "rgb(31, 119, 180)",
            "rgb(255, 127, 14)",
            "rgb(44, 160, 44)",
            "rgb(214, 39, 40)",
            "rgb(148, 103, 189)",
            "rgb(140, 86, 75)",
        ]
        data = [
            go.Scatter3d(
                x=xe,
                y=ye,
                z=ze,
                mode="lines",
                line={"color": room_edge_color, "width": 2},
                name="Room",
            ),
            go.Mesh3d(
                x=corners[:, 0],
                y=corners[:, 1],
                z=corners[:, 2],
                i=tri_i,
                j=tri_j,
                k=tri_k,
                opacity=room_alpha,
                color="lightgray",
                name="Room",
            ),
            go.Scatter3d(
                x=[sp[0]],
                y=[sp[1]],
                z=[sp[2]],
                mode="markers+text",
                text=["Source"],
                textposition="top center",
                marker={"size": 10, "color": "red", "symbol": "circle"},
                name="Source",
            ),
            go.Scatter3d(
                x=[rp[0]],
                y=[rp[1]],
                z=[rp[2]],
                mode="markers+text",
                text=["Receiver"],
                textposition="top center",
                marker={"size": 10, "color": "lime", "symbol": "square"},
                name="Receiver",
            ),
            go.Scatter3d(
                x=[p[0] for p in nodes],
                y=[p[1] for p in nodes],
                z=[p[2] for p in nodes],
                mode="markers+text",
                text=wall_labels,
                textposition="top center",
                marker={"size": 8, "color": node_colors, "symbol": "diamond"},
                name="Walls",
            ),
        ]
        fig = go.Figure(
            data=data,
            layout=go.Layout(
                title="SDN room",
                scene={
                    "xaxis": {"title": "x (m)", "range": [0, Lx]},
                    "yaxis": {"title": "y (m)", "range": [0, Ly]},
                    "zaxis": {"title": "z (m)", "range": [0, Lz]},
                    "aspectmode": "data",
                },
                template="plotly_white",
                height=500,
            ),
        )
        if show:
            fig.show()
        return fig

    @property
    def result(self):
        """Return last result from compute(); None if compute() not called yet."""
        return self._result

    def sdn_to_flamo(self, nfft=2**17, device=None):
        """
        Build a runnable FLAMO model from this SDN's parameters.

        Calls compute() if not already done.

        Parameters
        ----------
        nfft : int
            FFT size for FLAMO.
        device : torch device or None
            Device for FLAMO modules.

        Returns
        -------
        model : flamo.processor.system.Shell
        result : dict
            The SDN compute result.
        """
        if self._result is None:
            self.compute()
        return _result_to_flamo(self._result, nfft, device)


# ---------------------------------------------------------------------------
# Permutation matrix helper (used by SDN.build_fdn_permutation_from_routing)
# ---------------------------------------------------------------------------


def _build_fdn_permutation_from_routing(routing):
    """
    From SDN routing list of (from_node, to_node), build the 30x30 gather
    matrix T_gather that maps delay-line outputs into node-ordered inputs
    (connectivity only). In sdn_to_flamo the feedback matrix is
    S_block @ T_gather (scattering applied after gathering).
    """
    N = len(routing)
    n_nodes = 6
    arrivals_at: list[list[int]] = [[] for _ in range(n_nodes)]
    for k, (_i, j) in enumerate(routing):
        arrivals_at[j].append(k)
    for n in range(n_nodes):
        arrivals_at[n].sort()
    departures_from: list[list[int]] = [[] for _ in range(n_nodes)]
    for k, (i, _j) in enumerate(routing):
        departures_from[i].append(k)
    for n in range(n_nodes):
        departures_from[n].sort()

    # T_gather: (30, 30), grouped[t] = state[arrivals_at[t//5][t%5]] -> one 1 per row
    T_gather = np.zeros((N, N))
    for t in range(N):
        node, slot = divmod(t, 5)
        T_gather[t, arrivals_at[node][slot]] = 1.0

    return T_gather


# ---------------------------------------------------------------------------
# SDN result to FLAMO (used by SDN.sdn_to_flamo)
# ---------------------------------------------------------------------------


def _result_to_flamo(r, nfft, device):
    """Build FLAMO model from SDN result dict."""
    from collections import OrderedDict

    import torch
    from flamo.processor import dsp, system

    from ..auxiliary.flamo import delay_module, gain_module, sos_filter_module

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 30
    n_nodes = 6
    S_block = np.zeros((N, N))
    for node in range(n_nodes):
        S_block[5 * node : 5 * node + 5, 5 * node : 5 * node + 5] = r[
            "scattering_matrices"
        ][node]
    feedback_matrix = S_block @ r["permutation_matrix"]
    Fs = float(r["Fs"])
    delays = delay_module(
        np.asarray(r["delay_lengths_flat"], dtype=np.float64),
        nfft,
        Fs=Fs,
        device=device,
    )

    mixing_matrix = gain_module(
        feedback_matrix,
        nfft,
        device=device,
    )

    # Wall filters in SOS format (from SDN result), in feedback after mixing
    wall_filters_30 = sos_filter_module(
        r["wall_filters_sos"],
        nfft,
        device=device,
    )

    feedback = system.Series(
        OrderedDict(
            {
                "delays": delays,
                "mixing_matrix": mixing_matrix,
            }
        )
    )
    feedback_loop = system.Recursion(fF=wall_filters_30, fB=feedback)

    # ---- Input path: gain (1→6) -> delays (6) -> routing (6→30) ----
    input_gain_6 = gain_module(
        np.asarray(r["input_gains"], dtype=np.float64).reshape(-1, 1),
        nfft,
        device=device,
    )

    input_delays_6 = delay_module(
        np.asarray(r["input_delays"], dtype=np.float64),
        nfft,
        Fs=Fs,
        device=device,
    )

    input_matrix_30x6 = gain_module(
        np.asarray(r["input_matrix"], dtype=np.float64),
        nfft,
        device=device,
    )

    input_block = system.Series(
        OrderedDict(
            {
                "input_gain": input_gain_6,
                "input_delays": input_delays_6,
                "input_routing": input_matrix_30x6,
            }
        )
    )

    # ---- Output path: routing (30→6) -> delays (6) -> gain (6→1) ----
    output_matrix_6x30 = gain_module(
        np.asarray(r["output_matrix"], dtype=np.float64),
        nfft,
        device=device,
    )

    output_delays_6 = delay_module(
        np.asarray(r["output_delays"], dtype=np.float64),
        nfft,
        Fs=Fs,
        device=device,
    )

    output_gain_6to1 = gain_module(
        np.asarray(r["output_gains"], dtype=np.float64).reshape(1, -1),
        nfft,
        device=device,
    )

    output_block = system.Series(
        OrderedDict(
            {
                "output_routing": output_matrix_6x30,
                "output_delays": output_delays_6,
                "output_gain": output_gain_6to1,
            }
        )
    )

    fdn = system.Series(
        OrderedDict(
            {
                "input_block": input_block,
                "feedback_loop": feedback_loop,
                "output_block": output_block,
            }
        )
    )

    # ---- Direct path: delay -> gain (parallel with FDN) ----
    direct_delay = delay_module(
        np.array([float(r["direct_path_delay"])], dtype=np.float64),
        nfft,
        Fs=Fs,
        device=device,
    )

    direct_gain = gain_module(
        np.asarray(r["direct_path_gain"], dtype=np.float64).reshape(1, 1),
        nfft,
        device=device,
    )

    direct_path = system.Series(
        OrderedDict({"direct_delay": direct_delay, "direct_gain": direct_gain})
    )

    core = system.Parallel(brA=direct_path, brB=fdn, sum_output=True)

    input_layer = dsp.FFT(nfft)
    output_layer = dsp.iFFT(nfft)
    model = system.Shell(core=core, input_layer=input_layer, output_layer=output_layer)

    return model, r
