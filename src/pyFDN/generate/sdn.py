"""Scattering delay network (SDN) of a shoebox room.

An SDN puts one scattering node on each of the six walls, at the first-order
reflection point of the source. The 30 wall-to-wall paths are delay lines;
every node scatters its five incoming waves isotropically and filters what it
sends on with its wall's absorption. The source feeds the nodes, and the
receiver listens to them, through delays of their own, next to a delayed
direct path.

That makes an SDN more general than a :class:`~pyFDN.DSS`: the receiver taps
the nodes rather than the delay-line outputs, and the source and receiver
paths carry their own delays. :class:`SDN` therefore keeps its own fields --
named as in a DSS where they mean the same thing -- and renders through
:func:`sdn_to_flamo`.

Original SDN code: Enzo de Sena, Orchisama Das. Python translation: Sebastian
J. Schlecht, 2026.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

N_WALLS = 6
WALL_NAMES = ("floor", "ceiling", "left", "right", "front", "back")


Plane = tuple[float, float, float, float]


def _cuboid_walls(Lx: float, Ly: float, Lz: float) -> list[Plane]:
    """The six wall planes ``(a, b, c, d)``, ``a x + b y + c z + d = 0``.

    In the order of :data:`WALL_NAMES`, for a right-handed, z-up room: floor
    z = 0, ceiling z = Lz, left x = 0, right x = Lx, front y = Ly, back y = 0.
    """
    return [
        (0, 0, 1, 0),
        (0, 0, 1, -Lz),
        (1, 0, 0, 0),
        (-1, 0, 0, Lx),
        (0, -1, 0, Ly),
        (0, 1, 0, 0),
    ]


def _wall_node_position(
    source: np.ndarray, receiver: np.ndarray, wall: Plane
) -> np.ndarray:
    """Where the first-order reflection off ``wall`` meets it.

    The line from the source's mirror image to the receiver, intersected with
    the wall (the midpoint if the line runs parallel to it).
    """
    normal, offset = np.asarray(wall[:3], dtype=float), float(wall[3])
    image = source - 2 * (normal @ source + offset) / (normal @ normal) * normal
    direction = receiver - image
    den = float(normal @ direction)
    if abs(den) < 1e-12:
        return np.round((image + receiver) / 2, 5)
    return np.round(image - (normal @ image + offset) / den * direction, 5)


def _isotropic_scattering_matrix(N: int) -> np.ndarray:
    """Isotropic scattering matrix: ``S[i, j] = 2 / N - delta_ij``."""
    return np.full((N, N), 2.0 / N) - np.eye(N)


class SDN:
    """Scattering delay network of a shoebox room.

    Signal flow, with every delay in samples::

        x -> input_delays (one per wall) -> B -> [ wall_filters -> delays -> A ]
          -> C -> output_delays -> sum -> y,   plus  D * x delayed by direct_delay

    where ``[ ... ]`` is the feedback loop at the wall nodes: the nodes'
    output, after the wall filters, is what ``C`` picks up.

    Parameters
    ----------
    room_size : array-like, (3,)
        Room dimensions ``(Lx, Ly, Lz)`` in metres; z points up.
    source_pos, receiver_pos : array-like, (3,)
        Positions in metres.
    fs : float
        Sampling rate in Hz.
    c : float
        Speed of sound in m/s.
    wall_filters : array, optional
        SOS bank of the wall absorption: ``(sections, 6, 6)``, one filter per
        wall in the order of :data:`WALL_NAMES`, or ``(sections, 6, 30)``, one
        per delay line in the order of :attr:`routing`. ``None`` leaves the
        walls lossless.

    Attributes
    ----------
    delays : ndarray, (30,)
        Wall-to-wall delays in samples, in the order of :attr:`routing`.
    A : ndarray, (30, 30)
        Scattering matrix: isotropic 5 x 5 scattering per node, after the
        delay lines are gathered by the node they arrive at.
    B : ndarray, (30, 6)
        Source-to-node injection, the source gains folded in.
    C : ndarray, (6, 30)
        Node-to-receiver pickup, the receiver gains folded in.
    D : ndarray, (1, 1)
        Direct-path gain.
    wall_filters : ndarray, (sections, 6, 30)
        Wall absorption per delay line, as an SOS bank.
    input_delays, output_delays : ndarray, (6,)
        Source-to-wall and wall-to-receiver delays in samples.
    direct_delay : int
        Source-to-receiver delay in samples.
    routing : tuple of (int, int)
        ``routing[k] = (i, j)``: delay line ``k`` runs from wall ``i`` to wall ``j``.
    node_positions : ndarray, (6, 3)
        Scattering-node positions in metres.
    """

    def __init__(
        self,
        room_size: ArrayLike,
        source_pos: ArrayLike,
        receiver_pos: ArrayLike,
        *,
        fs: float = 48_000.0,
        c: float = 343.0,
        wall_filters: ArrayLike | None = None,
    ) -> None:
        self.room_size = np.asarray(room_size, dtype=float).reshape(3)
        self.source_pos = np.asarray(source_pos, dtype=float).reshape(3)
        self.receiver_pos = np.asarray(receiver_pos, dtype=float).reshape(3)
        self.fs = float(fs)
        self.c = float(c)

        source, receiver = self.source_pos, self.receiver_pos
        nodes = np.array(
            [
                _wall_node_position(source, receiver, wall)
                for wall in _cuboid_walls(*self.room_size)
            ]
        )
        self.node_positions = nodes
        self.routing = tuple(
            (i, j) for i in range(N_WALLS) for j in range(N_WALLS) if i != j
        )

        def samples(distance: float) -> int:
            return round(self.fs * distance / self.c)

        # at least one sample, so two nodes that coincide (at a room edge)
        # cannot close a delay-free loop
        one_sample = self.c / self.fs
        self.delays = np.array(
            [
                samples(max(_distance(nodes[i], nodes[j]), one_sample))
                for i, j in self.routing
            ]
        )

        scattering = _isotropic_scattering_matrix(N_WALLS - 1)
        self.A = np.kron(np.eye(N_WALLS), scattering) @ _gather_by_arrival(self.routing)

        eps = 1e-9  # distance floor for the 1/r gains
        to_wall = np.array([_distance(source, node) for node in nodes])
        to_receiver = np.array([_distance(node, receiver) for node in nodes])
        source_gains = one_sample / np.maximum(to_wall, eps)
        receiver_gains = 1.0 / (1.0 + to_receiver / np.maximum(to_wall, eps))
        leaves = np.array([i for i, _ in self.routing])
        self.B = 0.5 * (leaves[:, None] == np.arange(N_WALLS)) * source_gains
        self.C = (2.0 / (N_WALLS - 1)) * (
            leaves[None, :] == np.arange(N_WALLS)[:, None]
        )
        self.C = receiver_gains[:, None] * self.C
        self.input_delays = np.array([samples(d) for d in to_wall])
        self.output_delays = np.array([samples(d) for d in to_receiver])

        direct = _distance(source, receiver)
        self.D = np.array([[one_sample / max(direct, eps)]])
        self.direct_delay = samples(direct)

        if wall_filters is None:
            wall_filters = np.zeros((1, 6, N_WALLS))
            wall_filters[0, [0, 3]] = 1.0  # b0 = a0 = 1: pass-through
        bank = np.asarray(wall_filters, dtype=float)
        if bank.ndim != 3 or bank.shape[1] != 6 or bank.shape[2] not in (6, 30):
            raise ValueError(
                "wall_filters must be an SOS bank of shape (sections, 6, 6) or "
                f"(sections, 6, 30), got {bank.shape}"
            )
        self.wall_filters = bank[:, :, leaves] if bank.shape[2] == N_WALLS else bank


def sdn_to_flamo(
    sdn: SDN, *, nfft: int = 2**16, device: Any = None, dtype: Any = None
) -> Any:
    """Build a FLAMO ``Shell`` that renders ``sdn``; see :class:`SDN` for the flow.

    Use :func:`pyFDN.flamo_time_response` for a NumPy impulse response.
    """
    from flamo.processor import dsp, system

    from pyFDN.auxiliary.flamo import (
        default_device,
        delay_module,
        gain_module,
        sos_filter_module,
    )

    device = default_device(device)
    common = {"device": device, "dtype": dtype}

    def delays(samples: np.ndarray) -> Any:
        return delay_module(samples, nfft, fs=sdn.fs, **common)

    def gains(values: np.ndarray) -> Any:
        return gain_module(np.asarray(values, dtype=float), nfft, **common)

    def series(**modules: Any) -> Any:
        return system.Series(OrderedDict(modules))

    loop = system.Recursion(
        fF=sos_filter_module(sdn.wall_filters, nfft, **common),
        fB=series(delays=delays(sdn.delays), mixing_matrix=gains(sdn.A)),
    )
    network = series(
        input_block=series(
            input_gain=gains(np.ones((N_WALLS, 1))),
            input_delays=delays(sdn.input_delays),
            input_routing=gains(sdn.B),
        ),
        feedback_loop=loop,
        output_block=series(
            output_routing=gains(sdn.C),
            output_delays=delays(sdn.output_delays),
            output_gain=gains(np.ones((1, N_WALLS))),
        ),
    )
    direct = series(
        direct_delay=delays(np.array([sdn.direct_delay])), direct_gain=gains(sdn.D)
    )
    return system.Shell(
        core=system.Parallel(brA=direct, brB=network, sum_output=True),
        input_layer=dsp.FFT(nfft),
        output_layer=dsp.iFFT(nfft),
    )


def plot_sdn(sdn: SDN, *, room_alpha: float = 0.08) -> Any:
    """Draw the room, the source, the receiver and the six scattering nodes.

    Returns the Matplotlib figure; its 3-D axes is ``fig.axes[0]``.
    """
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    Lx, Ly, Lz = sdn.room_size
    corners = np.array(
        [[x, y, z] for z in (0, Lz) for x, y in ((0, 0), (Lx, 0), (Lx, Ly), (0, Ly))]
    )
    bottom, top = [0, 1, 2, 3], [4, 5, 6, 7]
    sides = [[i, (i + 1) % 4, (i + 1) % 4 + 4, i + 4] for i in range(4)]

    fig = Figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    for k, face in enumerate([bottom, top, *sides]):
        loop = corners[[*face, face[0]]]
        ax.plot(*loop.T, color="gray", linewidth=1.5, label="Room" if k == 0 else None)
    ax.add_collection3d(
        Poly3DCollection(
            [corners[f] for f in [bottom, top, *sides]],
            facecolor="lightgray",
            alpha=room_alpha,
        )
    )
    ax.scatter(*sdn.source_pos, s=80, color="red", marker="o", label="Source")
    ax.text(*sdn.source_pos, "Source", ha="center", va="bottom")
    ax.scatter(*sdn.receiver_pos, s=80, color="lime", marker="s", label="Receiver")
    ax.text(*sdn.receiver_pos, "Receiver", ha="center", va="bottom")
    colors = [f"C{i}" for i in range(N_WALLS)]
    ax.scatter(*sdn.node_positions.T, s=50, c=colors, marker="D", label="Walls")
    for position, name in zip(sdn.node_positions, WALL_NAMES, strict=True):
        ax.text(*position, name, ha="center", va="bottom")
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m, up)")
    ax.set_box_aspect((Lx, Ly, Lz))
    ax.set_title("SDN room")
    ax.legend(loc="upper left")
    return fig


def _distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))


def _gather_by_arrival(routing: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Permutation that groups the delay lines by the node they arrive at."""
    arrivals = sorted(range(len(routing)), key=lambda k: (routing[k][1], k))
    gather = np.zeros((len(routing), len(routing)))
    gather[np.arange(len(routing)), arrivals] = 1.0
    return gather


__all__ = ["SDN", "WALL_NAMES", "plot_sdn", "sdn_to_flamo"]
