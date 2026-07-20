"""
Traverse a FLAMO model and build a node tree or visualize it as a flowchart.

Flow: left-to-right. Series and Parallel are shown as boxes with nested modules.
Recursion is shown with forward path (fF) and feedback path (fB) inside a box,
with the feedback path drawn so the loop is visible (e.g. fB below fF).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyFDN.generate.fdn_matrix_gallery import FDNBuild


# Canonical names of a standard FDN's flamo graph, written by
# :func:`pyFDN.assemble_fdn_core` and read by :func:`extract_build`.
INPUT_GAIN = "input_gain"
OUTPUT_GAIN = "output_gain"
MIXING_MATRIX = "mixing_matrix"
DELAY = "delay"
LOOP_FILTER = "filter"
OUTPUT_FILTER = "output_filter"
DIRECT = "direct_gain"
LOOP_FILTER_ALIASES = (LOOP_FILTER, "attenuation")

# Structural branch labels assigned by the traversal.
BRANCH_FDN = "fdn"
BRANCH_DIRECT = "direct"
BRANCH_FEEDFORWARD = "feedforward"
BRANCH_FEEDBACK = "feedback"


# Traversal uses type(module).__name__ and getattr; no need to import flamo here.


def _get_typename(module: Any) -> str:
    return type(module).__name__


def _series_children(module: Any) -> list[tuple[str, Any]]:
    """Return list of (name, submodule) for Series-like module."""
    # FLAMO Series is nn.Sequential; direct children are in ._modules (OrderedDict)
    modules = getattr(module, "_modules", None)
    if modules is None:
        modules = getattr(module, "modules", None)
    if modules is None:
        return []
    if hasattr(modules, "items"):
        return list(modules.items())
    if hasattr(modules, "__iter__") and not isinstance(modules, str | bytes):
        return [(str(i), m) for i, m in enumerate(modules)]
    return []


def _is_shell(module: Any) -> bool:
    return _get_typename(module) == "Shell"


def _is_series(module: Any) -> bool:
    return _get_typename(module) == "Series"


def _is_parallel(module: Any) -> bool:
    return _get_typename(module) == "Parallel"


def _is_recursion(module: Any) -> bool:
    return _get_typename(module) == "Recursion"


def flamo_model_to_nodes(
    model: Any,
    name: str = "root",
    *,
    include_shell_io: bool = False,
) -> dict[str, Any]:
    """
    Traverse a FLAMO model and build a tree of nodes (nested dicts).

    Each node has:
    - type: "Shell" | "Series" | "Parallel" | "Recursion" | "Leaf"
    - name: str (from parent's dict key or assigned)
    - module: the raw FLAMO module (for Leaf, the actual dsp module)
    - children: list of child nodes (for Series, Parallel; order preserved)
    - fF, fB: only for Recursion — nodes for forward and feedback path
    - input_layer, output_layer: only if include_shell_io and type is Shell

    Parameters
    ----------
    model : FLAMO model (Shell, Series, Parallel, Recursion, or dsp module)
    name : str
        Name for the root node.
    include_shell_io : bool
        If True, include input_layer and output_layer as children for Shell.

    Returns
    -------
    node : dict
        Root node (nested tree). Use flamo_nodes_flat() to get a list of all nodes.
    """
    node: dict[str, Any] = {
        "type": "Leaf",
        "name": name,
        "module": model,
        "children": [],
    }
    in_ch = getattr(model, "input_channels", None)
    out_ch = getattr(model, "output_channels", None)
    if in_ch is not None:
        node["input_channels"] = in_ch
    if out_ch is not None:
        node["output_channels"] = out_ch
    if _is_shell(model):
        node["type"] = "Shell"
        # FLAMO Shell stores core as __core; use get_core() if available
        core = None
        if callable(getattr(model, "get_core", None)):
            core = model.get_core()
        if core is None:
            core = getattr(model, "core", None)
        if core is None:
            core = getattr(model, "_Shell__core", None)
        if core is not None:
            node["children"] = [
                flamo_model_to_nodes(
                    core, name="core", include_shell_io=include_shell_io
                )
            ]
        # Always attach input/output layers for Shell so draw can use them as model I/O
        il = None
        if callable(getattr(model, "get_inputLayer", None)):
            il = model.get_inputLayer()
        if il is None:
            il = getattr(model, "input_layer", None) or getattr(
                model, "_Shell__input_layer", None
            )
        ol = None
        if callable(getattr(model, "get_outputLayer", None)):
            ol = model.get_outputLayer()
        if ol is None:
            ol = getattr(model, "output_layer", None) or getattr(
                model, "_Shell__output_layer", None
            )
        if il is not None:
            node.setdefault(
                "input_layer",
                flamo_model_to_nodes(
                    il, name="input_layer", include_shell_io=include_shell_io
                ),
            )
        if ol is not None:
            node.setdefault(
                "output_layer",
                flamo_model_to_nodes(
                    ol, name="output_layer", include_shell_io=include_shell_io
                ),
            )
        return node

    if _is_series(model):
        node["type"] = "Series"
        pairs = _series_children(model)
        node["children"] = [
            flamo_model_to_nodes(sub, name=nm, include_shell_io=include_shell_io)
            for nm, sub in pairs
        ]
        return node

    if _is_parallel(model):
        node["type"] = "Parallel"
        brA = getattr(model, "brA", None) or getattr(model, "branchA", None)
        brB = getattr(model, "brB", None) or getattr(model, "branchB", None)
        children = []
        if brA is not None:
            children.append(
                flamo_model_to_nodes(
                    brA, name=BRANCH_FDN, include_shell_io=include_shell_io
                )
            )
        if brB is not None:
            children.append(
                flamo_model_to_nodes(
                    brB, name=BRANCH_DIRECT, include_shell_io=include_shell_io
                )
            )
        node["children"] = children
        return node

    if _is_recursion(model):
        node["type"] = "Recursion"
        fF = getattr(model, "fF", None) or getattr(model, "feedforward", None)
        fB = getattr(model, "fB", None) or getattr(model, "feedback", None)
        node["fF"] = (
            flamo_model_to_nodes(
                fF, name=BRANCH_FEEDFORWARD, include_shell_io=include_shell_io
            )
            if fF is not None
            else None
        )
        node["fB"] = (
            flamo_model_to_nodes(
                fB, name=BRANCH_FEEDBACK, include_shell_io=include_shell_io
            )
            if fB is not None
            else None
        )
        node["children"] = []  # Recursion uses fF/fB, not children
        return node

    # Leaf (Gain, Delay, FFT, etc.)
    return node


def flamo_nodes_flat(
    root: dict[str, Any],
    path: str = "root",
) -> list[dict[str, Any]]:
    """
    Flatten the node tree into a list of nodes, each with a 'path' key.

    Parameters
    ----------
    root : dict
        Root node from flamo_model_to_nodes().
    path : str
        Path prefix for the root.

    Returns
    -------
    list of dict
        Each dict has keys from the node plus "path" (e.g. "root/core/feedback_loop/fF").
    """
    out: list[dict[str, Any]] = []
    node = {k: v for k, v in root.items() if k not in ("fF", "fB")}
    node["path"] = path
    out.append(node)

    for _i, ch in enumerate(root.get("children") or []):
        subpath = f"{path}/{ch['name']}"
        out.extend(flamo_nodes_flat(ch, path=subpath))

    if root.get("type") == "Recursion":
        for key in ("fF", "fB"):
            child = root.get(key)
            if child is not None:
                subpath = f"{path}/{child['name']}"
                out.extend(flamo_nodes_flat(child, path=subpath))
    return out


def _module_value(module: Any) -> np.ndarray:
    param = getattr(module, "param", None)
    if param is None:
        raise ValueError(f"{type(module).__name__} has no parameter tensor")
    mapper = getattr(module, "map", None)
    value = mapper(param) if callable(mapper) else param
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def _delay_samples(module: Any) -> np.ndarray:
    getter = getattr(module, "get_delays", None)
    param = getattr(module, "param", None)
    if callable(getter) and param is not None:
        value = getter()(param)
    else:
        mapper = getattr(module, "map", None)
        value = mapper(param) if callable(mapper) else param
        to_samples = getattr(module, "s2sample", None)
        if callable(to_samples):
            value = to_samples(value)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    samples = np.asarray(value.numpy() if hasattr(value, "numpy") else value)
    return np.asarray(np.round(samples), dtype=int).ravel()


def _gain_to_sos(gain: np.ndarray) -> np.ndarray:
    gain = np.asarray(gain, dtype=float).ravel()
    sos = np.zeros((1, 6, gain.size))
    sos[0, 0, :] = gain
    sos[0, 3, :] = 1.0
    return sos


def _svf_to_sos(module: Any) -> np.ndarray:
    """Bake a flamo ``parallelSVF`` into a normalized ``(n_sections, 6, N)`` SOS bank."""

    def _np(t: Any) -> np.ndarray:
        if hasattr(t, "detach"):
            t = t.detach()
        if hasattr(t, "cpu"):
            t = t.cpu()
        return np.asarray(t.numpy() if hasattr(t, "numpy") else t, dtype=float)

    f, r, m_lp, m_bp, m_hp = (_np(t) for t in module.map_param2svf(module.param))
    f, r, m_lp, m_bp, m_hp = np.broadcast_arrays(f, r, m_lp, m_bp, m_hp)
    a0 = f**2 + 2.0 * r * f + 1.0
    sos = np.empty((*f.shape, 6), dtype=float)
    sos[..., 0] = (f**2 * m_lp + f * m_bp + m_hp) / a0
    sos[..., 1] = (2.0 * f**2 * m_lp - 2.0 * m_hp) / a0
    sos[..., 2] = (f**2 * m_lp - f * m_bp + m_hp) / a0
    sos[..., 3] = 1.0
    sos[..., 4] = (2.0 * f**2 - 2.0) / a0
    sos[..., 5] = (f**2 - 2.0 * r * f + 1.0) / a0
    return np.ascontiguousarray(np.moveaxis(sos, -1, 1))


def _attenuation_to_sos(module: Any) -> np.ndarray:
    """In-loop filter module -> ``(n_sections, 6, N)`` SOS for ``FDNBuild.filters``.

    Supports ``parallelSVF`` and ``parallelSOSFilter`` (and a plain broadband
    gain); other filter types raise.
    """
    type_name = type(module).__name__
    if "SVF" in type_name:
        return _svf_to_sos(module)
    value = _module_value(module)
    if value.ndim == 1:
        return _gain_to_sos(value)
    if value.ndim == 3 and value.shape[1] == 6:
        return np.asarray(value, dtype=float)
    raise ValueError(
        f"cannot extract in-loop filter of type {type_name!r} to "
        "FDNBuild.filters; only biquad-cascade IIR filters (parallelSVF, "
        "parallelSOSFilter) are supported"
    )


def _feedback_leaf_module(leaves: list[dict[str, Any]]) -> Any:
    """The single feedback-matrix leaf module (named ``mixing_matrix``).

    Returns the **live** module (``.map``/``.param`` intact); callers that need
    gradients must apply ``.map`` themselves rather than going through
    :func:`_module_value`, which detaches.
    """
    matches = [n for n in leaves if n["name"] == MIXING_MATRIX]
    if len(matches) != 1:
        raise ValueError(
            "FLAMO graph must contain exactly one feedback matrix leaf; "
            f"found {len(matches)}"
        )
    return matches[0]["module"]


def feedback_matrix_module(model: Any) -> Any:
    """Return the live feedback-matrix module from a FLAMO FDN model.

    Apply ``module.map(module.param)`` to read the realized matrix **in-graph**
    (e.g. inside a training loss); :func:`extract_build` detaches instead.
    """
    root = flamo_model_to_nodes(model)
    leaves = [node for node in flamo_nodes_flat(root) if node["type"] == "Leaf"]
    return _feedback_leaf_module(leaves)


def extract_build(model: Any) -> FDNBuild:
    """Extract a complete :class:`~pyFDN.FDNBuild` from a named FLAMO model graph.

    The graph is a standard FDN as built by :func:`assemble_fdn_core`, with the
    canonical leaf names ``input_gain``, ``output_gain``, ``delay``,
    ``direct_gain``, ``mixing_matrix`` (the feedback matrix), and optional
    ``filter`` (-> ``filters``) / ``output_filter`` (-> ``post_eq``). The sample
    rate is read from the delay module and is
    required: a graph that does not expose ``fs`` is malformed and raises
    :class:`ValueError`.
    """
    root = flamo_model_to_nodes(model)
    leaves = [node for node in flamo_nodes_flat(root) if node["type"] == "Leaf"]

    def _named(name: str) -> list[dict[str, Any]]:
        return [node for node in leaves if node["name"] == name]

    def _one_of(*names: str) -> Any:
        matches: list[dict[str, Any]] = []
        for name in names:
            matches = _named(name)
            if matches:
                break
        if len(matches) != 1:
            raise ValueError(
                f"FLAMO graph must contain exactly one of {names!r}; "
                f"found {len(matches)}"
            )
        return matches[0]["module"]

    delay_matches = _named(DELAY)
    if len(delay_matches) != 1:
        raise ValueError(
            "FLAMO graph must contain exactly one delay leaf; "
            f"found {len(delay_matches)}"
        )
    delay_module = delay_matches[0]["module"]
    delays = _delay_samples(delay_module)

    A = np.asarray(_module_value(_feedback_leaf_module(leaves)), dtype=float)
    B = np.asarray(_module_value(_one_of(INPUT_GAIN)), dtype=float)
    C = np.asarray(_module_value(_one_of(OUTPUT_GAIN)), dtype=float)
    if A.ndim != 2:
        raise ValueError("feedback matrix must be a constant two-dimensional matrix")

    direct_matches = _named(DIRECT)
    D = (
        np.asarray(_module_value(direct_matches[0]["module"]), dtype=float)
        if len(direct_matches) == 1
        else np.zeros((C.shape[0], B.shape[1]))
    )

    attenuation_sos = None
    attenuation_matches = [
        n for n in leaves if n["name"] in LOOP_FILTER_ALIASES
    ]
    if len(attenuation_matches) == 1:
        attenuation_sos = _attenuation_to_sos(attenuation_matches[0]["module"])

    post_eq_sos = None
    post_eq_matches = _named(OUTPUT_FILTER)
    if len(post_eq_matches) == 1:
        post_eq_value = _module_value(post_eq_matches[0]["module"])
        if post_eq_value.ndim == 3 and post_eq_value.shape[1] == 6:
            post_eq_sos = np.asarray(post_eq_value, dtype=float)

    fs_value = getattr(delay_module, "fs", None)
    if fs_value is None:
        raise ValueError(
            "FLAMO delay module exposes no sample rate; cannot extract fs"
        )

    from pyFDN.generate.fdn_matrix_gallery import FDNBuild

    return FDNBuild(
        A=A,
        B=B,
        C=C,
        D=D,
        delays=delays,
        fs=float(fs_value),
        filters=attenuation_sos,
        post_eq=post_eq_sos,
    )


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


def _leaf_label(node: dict[str, Any]) -> str:
    mod = node.get("module")
    mod_type = type(mod).__name__ if mod is not None else "?"
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


def plot_flamo_graph(
    model: Any,
    *,
    name: str = "flamo",
    ax: Any = None,
    scale: float = 0.85,
    fontsize: float = 9.0,
) -> Any:
    """
    Draw the FLAMO model signal flow with matplotlib.

    Signal flows left to right; only the feedback path of a Recursion flows
    right to left, drawn below the forward path with a loop back to a sum
    node at the forward path's input.

    Parameters
    ----------
    model : FLAMO model (Shell, Series, Parallel, Recursion, or dsp module)
    name : str
        Name for the root node.
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
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

    root = flamo_model_to_nodes(model, name=name, include_shell_io=True)
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
            sum_node(x_merge, y_axis)
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
        mod = node.get("module")
        face, edge = _BLOCK_COLORS[_block_category(type(mod).__name__)]
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
