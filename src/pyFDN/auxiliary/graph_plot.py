"""Signal-flow diagrams of pyFDN systems, independent of any backend.

:func:`plot_system_graph` draws a ``pyfdn-system`` document, a
:class:`~pyFDN.system.GraphNode` tree or an :class:`~pyFDN.FDNBuild` directly
from its description; no FLAMO model or ``td`` graph is built.
:func:`pyFDN.plot_flamo_graph` uses the same renderer for a live FLAMO model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyFDN.build import FDNBuild
    from pyFDN.system import FDNSystemDoc, GraphNode


# ---------------------------------------------------------------------------
# Matplotlib renderer
# ---------------------------------------------------------------------------

# Layout constants (abstract units; figure is scaled at the end)
_LEAF_H = 0.9  # leaf box height
_GAP_X = 0.6  # horizontal wire length between series elements
_GAP_Y = 0.5  # vertical gap between parallel branches / fF and fB
_NODE_R = 0.13  # radius of sum (+) and pickoff nodes
_PAD = 0.3  # padding inside recursion container
_STUB = 0.35  # split/merge wire stub length in Parallel

_WIRE = "0.25"
_LOOP_EDGE = "#b05a4a"

# Leaf box colors by module category: (facecolor, edgecolor)
_BLOCK_COLORS = {
    "transform": ("#efefef", "#7a7a7a"),  # FFT / iFFT
    "gain": ("#dbe9f6", "#3d6d9e"),  # gains
    "matrix": ("#e8e1f2", "#7b5ea7"),  # mixing / feedback matrices
    "delay": ("#fdf1d6", "#c08a2d"),  # delays
    "filter": ("#e1efe3", "#4f8a5e"),  # filters / EQs
    "other": ("#f5f5f5", "#8c8c8c"),
}


def _block_category(mod_type: str) -> str:
    t = mod_type.lower()
    if "fft" in t:
        return "transform"
    if "delay" in t:
        return "delay"
    if any(k in t for k in ("filter", "biquad", "svf", "geq", "peq", "eq", "sos")):
        return "filter"
    if any(k in t for k in ("matrix", "scattering", "householder", "hadamard")):
        return "matrix"
    if "gain" in t:
        return "gain"
    return "other"


def _leaf_type(node: dict[str, Any]) -> str:
    """The leaf's type name: a system ``type`` or the FLAMO class name."""
    if node.get("kind"):
        return str(node["kind"])
    mod = node.get("module")
    return type(mod).__name__ if mod is not None else "?"


def _leaf_label(node: dict[str, Any]) -> str:
    mod_type = _leaf_type(node)
    name = node.get("name", "")
    if name and name != mod_type and not name.isdigit():
        return f"{name}\n{mod_type}"
    return mod_type


def _measure(node: dict[str, Any]) -> tuple[float, float, float]:
    """
    Bottom-up size pass. Returns (w, h, ay) where ay is the distance from the
    top of the bounding box to the signal-flow axis. Cached in node["_size"].
    """
    ntype = node.get("type", "Leaf")

    if ntype == "Shell":
        parts = []
        if node.get("input_layer") is not None:
            parts.append(node["input_layer"])
        parts.extend(node.get("children") or [])
        if node.get("output_layer") is not None:
            parts.append(node["output_layer"])
        node["_parts"] = parts
        sizes = [_measure(ch) for ch in parts]
        w = sum(s[0] for s in sizes) + _GAP_X * max(len(parts) - 1, 0)
        ay = max((s[2] for s in sizes), default=_LEAF_H / 2)
        below = max((s[1] - s[2] for s in sizes), default=_LEAF_H / 2)
        size = (w, ay + below, ay)

    elif ntype == "Series":
        sizes = [_measure(ch) for ch in node.get("children") or []]
        w = sum(s[0] for s in sizes) + _GAP_X * max(len(sizes) - 1, 0)
        ay = max((s[2] for s in sizes), default=_LEAF_H / 2)
        below = max((s[1] - s[2] for s in sizes), default=_LEAF_H / 2)
        size = (w, ay + below, ay)

    elif ntype == "Parallel":
        sizes = [_measure(ch) for ch in node.get("children") or []]
        w = max((s[0] for s in sizes), default=1.0) + 2 * (_STUB + _NODE_R * 2)
        h = sum(s[1] for s in sizes) + _GAP_Y * max(len(sizes) - 1, 0)
        size = (w, h, h / 2)

    elif ntype == "Recursion":
        wf, hf, ayf = _measure(node["fF"]) if node.get("fF") else (1.0, _LEAF_H, 0.45)
        wb, hb, _ = _measure(node["fB"]) if node.get("fB") else (1.0, _LEAF_H, 0.45)
        inner_w = max(wf, wb)
        w = _PAD + 2 * _NODE_R + _GAP_X + inner_w + _GAP_X + 2 * _NODE_R + _PAD
        h = _PAD + hf + _GAP_Y + hb + _PAD
        size = (w, h, _PAD + ayf)

    else:  # Leaf
        label = _leaf_label(node)
        wmax = max(len(line) for line in label.split("\n"))
        size = (max(1.1, 0.115 * wmax + 0.35), _LEAF_H, _LEAF_H / 2)

    node["_size"] = size
    return size


def plot_node_tree(
    root: dict[str, Any],
    *,
    ax: Any = None,
    scale: float = 0.85,
    fontsize: float = 9.0,
) -> Any:
    """Draw a node tree as a signal-flow diagram with matplotlib.

    ``root`` is a tree of plain dicts as :func:`pyFDN.flamo_model_to_nodes`
    returns: ``type`` is ``"Shell"``, ``"Series"``, ``"Parallel"``,
    ``"Recursion"`` or ``"Leaf"``; containers carry ``children`` (a recursion
    ``fF`` and ``fB``); a leaf is labelled by its ``name`` and by ``kind``, or
    the class name of its ``module`` when there is no ``kind``. A ``Parallel``
    with ``sum_output`` false stacks its outputs instead of summing them.

    Signal flows left to right; only the feedback path of a Recursion flows
    right to left, drawn below the forward path with a loop back to a sum
    node at the forward path's input.

    Parameters
    ----------
    root : dict
        Root of the node tree.
    ax : matplotlib Axes, optional
        Draw into this axes; otherwise a new figure sized to the layout
        is created.
    scale : float
        Inches per layout unit when creating a new figure.
    fontsize : float
        Base font size for leaf labels.

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

    total_w, total_h, root_ay = _measure(root)

    if ax is None:
        fig_w = (total_w + 2.0) * scale
        fig_h = (total_h + 0.8) * scale
        _, ax = plt.subplots(figsize=(fig_w, fig_h))

    Point = tuple[float, float]
    Ports = tuple[Point, Point]

    # y grows downward in layout; flip the axis at the end.
    def route(
        pts: list[Point], *, arrow: bool = True, color: str = _WIRE, ls: str = "-"
    ) -> None:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if len(pts) > 2:
            ax.plot(xs[:-1], ys[:-1], color=color, lw=1.2, ls=ls, zorder=1)
        if arrow:
            patch = FancyArrowPatch(
                pts[-2],
                pts[-1],
                arrowstyle="-|>",
                mutation_scale=11,
                color=color,
                lw=1.2,
                linestyle=ls,
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
            ax.add_patch(patch)
        else:
            ax.plot(xs[-2:], ys[-2:], color=color, lw=1.2, ls=ls, zorder=1)

    def sum_node(x: float, y: float, color: str = _WIRE) -> None:
        ax.add_patch(
            Circle(
                (x, y), _NODE_R, facecolor="white", edgecolor=color, lw=1.2, zorder=3
            )
        )
        ax.text(
            x,
            y,
            "+",
            ha="center",
            va="center_baseline",
            fontsize=fontsize,
            color=color,
            zorder=4,
        )

    def stack_node(x: float, y: float, color: str = _WIRE) -> None:
        """Where a non-summing Parallel stacks its branch outputs."""
        ax.add_patch(
            Rectangle(
                (x - 0.04, y - 2 * _NODE_R),
                0.08,
                4 * _NODE_R,
                facecolor=color,
                edgecolor=color,
                zorder=3,
            )
        )

    def pickoff(x: float, y: float, color: str = _WIRE) -> None:
        ax.add_patch(Circle((x, y), 0.045, facecolor=color, edgecolor=color, zorder=3))

    def xpos(x_left: float, w_total: float, off: float, w_el: float, d: int) -> float:
        """Left edge of an element at offset `off` from the input side."""
        return x_left + off if d > 0 else x_left + w_total - off - w_el

    def draw(node: dict[str, Any], x: float, y_axis: float, d: int) -> Ports:
        """Draw node with left edge x and flow axis y_axis, direction d (+1/-1).
        Returns (p_in, p_out) port coordinates."""
        ntype = node.get("type", "Leaf")
        w, h, ay = node["_size"]
        y_top = y_axis - ay

        if ntype in ("Shell", "Series"):
            children = (
                node["_parts"] if ntype == "Shell" else node.get("children") or []
            )
            if not children:
                p = ((x, y_axis), (x + w, y_axis))
                return p if d > 0 else (p[1], p[0])
            off = 0.0
            ports = []
            prev_out = None
            for ch in children:
                cw, _, _ = ch["_size"]
                cx = xpos(x, w, off, cw, d)
                p_in, p_out = draw(ch, cx, y_axis, d)
                if prev_out is not None:
                    route([prev_out, p_in])
                ports.append((p_in, p_out))
                prev_out = p_out
                off += cw + _GAP_X
            return ports[0][0], ports[-1][1]

        if ntype == "Parallel":
            stub = _STUB + 2 * _NODE_R
            x_in = x if d > 0 else x + w
            x_merge = x + w - _NODE_R if d > 0 else x + _NODE_R
            if node.get("sum_output", True):
                sum_node(x_merge, y_axis)
            else:
                stack_node(x_merge, y_axis)
            pickoff(x_in, y_axis)
            cy = y_top
            for ch in node.get("children") or []:
                cw, chh, cay = ch["_size"]
                cx = xpos(x, w, stub + (max(0.0, w - 2 * stub - cw)) / 2, cw, d)
                p_in, p_out = draw(ch, cx, cy + cay, d)
                route([(x_in, y_axis), (x_in, p_in[1]), p_in])
                route(
                    [
                        p_out,
                        (x_merge, p_out[1]),
                        (x_merge, y_axis - _NODE_R * _sign(y_axis - p_out[1])),
                    ]
                )
                cy += chh + _GAP_Y
            p = ((x_in, y_axis), (x_merge + _NODE_R * d, y_axis))
            return p

        if ntype == "Recursion":
            # dashed container
            ax.add_patch(
                FancyBboxPatch(
                    (x, y_top),
                    w,
                    h,
                    boxstyle="round,pad=0.06",
                    facecolor="none",
                    edgecolor=_LOOP_EDGE,
                    lw=1.0,
                    ls=(0, (4, 3)),
                    zorder=0,
                )
            )
            nname = node.get("name", "")
            if nname and not nname.isdigit():
                ax.text(
                    x + 0.12,
                    y_top + 0.06,
                    nname,
                    ha="left",
                    va="top",
                    fontsize=fontsize - 1.5,
                    color=_LOOP_EDGE,
                    style="italic",
                    zorder=3,
                )
            fF, fB = node.get("fF"), node.get("fB")
            if fF is None:  # placeholder; matches the _measure() default size
                fF = {
                    "type": "Leaf",
                    "name": "fF",
                    "module": None,
                    "_size": (1.0, _LEAF_H, _LEAF_H / 2),
                }
            wf, hf, _ = fF["_size"]
            wb, hb, _ = fB["_size"] if fB else (0, 0, 0)
            inner_w = max(wf, wb)
            x_sum = xpos(x, w, _PAD + _NODE_R, 0, d)
            x_pick = xpos(x, w, w - _PAD - _NODE_R, 0, d)
            sum_node(x_sum, y_axis, color=_LOOP_EDGE)
            pickoff(x_pick, y_axis)
            fx = xpos(x, w, _PAD + 2 * _NODE_R + _GAP_X + (inner_w - wf) / 2, wf, d)
            f_in, f_out = draw(fF, fx, y_axis, d)
            route([(x_sum + _NODE_R * d, y_axis), f_in])
            route([f_out, (x_pick, y_axis)], arrow=False)
            if fB is not None:
                yb_axis = y_top + _PAD + hf + _GAP_Y + fB["_size"][2]
                bx = xpos(x, w, _PAD + 2 * _NODE_R + _GAP_X + (inner_w - wb) / 2, wb, d)
                b_in, b_out = draw(fB, bx, yb_axis, -d)
                route([(x_pick, y_axis), (x_pick, b_in[1]), b_in], color=_LOOP_EDGE)
                route(
                    [b_out, (x_sum, b_out[1]), (x_sum, y_axis + _NODE_R)],
                    color=_LOOP_EDGE,
                )
            return (x_sum - _NODE_R * d, y_axis), (x_pick, y_axis)

        # Leaf
        face, edge = _BLOCK_COLORS[_block_category(_leaf_type(node))]
        ax.add_patch(
            FancyBboxPatch(
                (x, y_top),
                w,
                h,
                boxstyle="round,pad=0.02",
                facecolor=face,
                edgecolor=edge,
                lw=1.2,
                zorder=2,
            )
        )
        ax.text(
            x + w / 2,
            y_axis,
            _leaf_label(node),
            ha="center",
            va="center",
            fontsize=fontsize,
            zorder=3,
            linespacing=1.3,
        )
        p = ((x, y_axis), (x + w, y_axis))
        return p if d > 0 else (p[1], p[0])

    def _sign(v: float) -> float:
        return 1.0 if v >= 0 else -1.0

    x0 = 1.0
    p_in, p_out = draw(root, x0, root_ay, +1)
    # model input / output stubs
    route([(x0 - 0.8, p_in[1]), p_in])
    route([p_out, (p_out[0] + 0.8, p_out[1])])
    ax.text(x0 - 0.85, p_in[1], "in", ha="right", va="center", fontsize=fontsize)
    ax.text(p_out[0] + 0.85, p_out[1], "out", ha="left", va="center", fontsize=fontsize)

    ax.set_xlim(x0 - 1.6, x0 + total_w + 1.6)
    ax.set_ylim(total_h + 0.4, -0.4)  # inverted: layout y grows downward
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


_NODE_KIND = {"series": "Series", "parallel": "Parallel", "recursion": "Recursion"}


def graph_to_plot_nodes(graph: GraphNode, name: str | None = None) -> dict[str, Any]:
    """Convert a :class:`~pyFDN.system.GraphNode` tree to the dicts
    :func:`plot_node_tree` draws."""
    node: dict[str, Any] = {
        "type": _NODE_KIND.get(graph.type, "Leaf"),
        "name": graph.name or name or "",
        "children": [],
    }
    if graph.type == "recursion":
        forward, feedback = graph.children
        node["fF"] = graph_to_plot_nodes(forward, "forward")
        node["fB"] = graph_to_plot_nodes(feedback, "feedback")
    elif graph.type in _NODE_KIND:
        node["children"] = [graph_to_plot_nodes(child) for child in graph.children]
        if graph.type == "parallel":
            node["sum_output"] = graph.options.get("sum_output", True)
    else:
        node["kind"] = graph.type
    return node


def plot_system_graph(
    system: FDNSystemDoc | GraphNode | FDNBuild,
    *,
    ax: Any = None,
    scale: float = 0.85,
    fontsize: float = 9.0,
) -> Any:
    """Draw the signal flow of a system, without choosing a backend.

    Parameters
    ----------
    system : FDNSystemDoc, GraphNode or FDNBuild
        What to draw. A ``build`` document and a bare build are drawn as the
        FDN graph :func:`pyFDN.system.build_to_graph` produces.
    ax : matplotlib Axes, optional
        Draw into this axes; otherwise a new figure sized to the layout
        is created.
    scale : float
        Inches per layout unit when creating a new figure.
    fontsize : float
        Base font size for leaf labels.

    Returns
    -------
    ax : matplotlib Axes
    """
    from pyFDN.build import FDNBuild
    from pyFDN.system import FDNSystemDoc, build_to_graph

    if isinstance(system, FDNSystemDoc):
        graph = system.as_graph()
    elif isinstance(system, FDNBuild):
        graph = build_to_graph(system)
    else:
        graph = system
    root = graph_to_plot_nodes(graph)
    return plot_node_tree(root, ax=ax, scale=scale, fontsize=fontsize)


__all__ = ["graph_to_plot_nodes", "plot_node_tree", "plot_system_graph"]
