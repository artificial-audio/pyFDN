"""
Traverse a FLAMO model and build a node tree or visualize it as a flowchart.

Flow: left-to-right. Series and Parallel are shown as boxes with nested modules.
Recursion is shown with forward path (fF) and feedback path (fB) inside a box,
with the feedback path drawn so the loop is visible (e.g. fB below fF).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pyFDN.auxiliary.flamo import to_numpy

if TYPE_CHECKING:
    from pyFDN.build import FDNBuild


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
                flamo_model_to_nodes(brA, name="brA", include_shell_io=include_shell_io)
            )
        if brB is not None:
            children.append(
                flamo_model_to_nodes(brB, name="brB", include_shell_io=include_shell_io)
            )
        node["children"] = children
        return node

    if _is_recursion(model):
        node["type"] = "Recursion"
        fF = getattr(model, "fF", None) or getattr(model, "feedforward", None)
        fB = getattr(model, "fB", None) or getattr(model, "feedback", None)
        node["fF"] = (
            flamo_model_to_nodes(fF, name="fF", include_shell_io=include_shell_io)
            if fF is not None
            else None
        )
        node["fB"] = (
            flamo_model_to_nodes(fB, name="fB", include_shell_io=include_shell_io)
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
                subpath = f"{path}/{key}"
                out.extend(flamo_nodes_flat(child, path=subpath))
    return out


def _module_value(module: Any) -> np.ndarray:
    param = getattr(module, "param", None)
    if param is None:
        raise ValueError(f"{type(module).__name__} has no parameter tensor")
    mapper = getattr(module, "map", None)
    value = mapper(param) if callable(mapper) else param
    return to_numpy(value)


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
    samples = to_numpy(value)
    return np.asarray(np.round(samples), dtype=int).ravel()


def _gain_to_sos(gain: np.ndarray) -> np.ndarray:
    gain = np.asarray(gain, dtype=float).ravel()
    sos = np.zeros((1, 6, gain.size))
    sos[0, 0, :] = gain
    sos[0, 3, :] = 1.0
    return sos


def _feedback_leaf_module(leaves: list[dict[str, Any]]) -> Any:
    """The single feedback-matrix leaf module from a flat leaf list.

    Accepts a ``mixing_matrix`` leaf or the standard recursion feedback leaf
    ``fB`` (preferring the one at ``.../feedback_loop/fB`` so a Parallel direct
    branch does not confuse it). Returns the **live** module (``.map``/``.param``
    intact). Callers that need gradients must apply ``.map`` themselves rather
    than going through :func:`_module_value`, which detaches.
    """
    matches = [n for n in leaves if n["name"] == "mixing_matrix"]
    if not matches:
        matches = [
            n
            for n in leaves
            if n["name"] == "fB" and n["path"].endswith("/feedback_loop/fB")
        ] or [n for n in leaves if n["name"] == "fB"]
    if len(matches) != 1:
        raise ValueError(
            "FLAMO graph must contain exactly one feedback matrix leaf; "
            f"found {len(matches)}"
        )
    return matches[0]["module"]


def feedback_matrix_module(model: Any) -> Any:
    """Return the live feedback-matrix module from a FLAMO FDN model.

    Works for both a plain ``Series`` core and a ``Parallel`` core (an FDN summed
    with a direct path). The module returned is the one on the recursion's
    feedback branch; apply ``module.map(module.param)`` to read the realized
    matrix **in-graph** -- e.g. inside a training loss, where the detaching
    extraction path :func:`extract_build` would break gradients.
    """
    root = flamo_model_to_nodes(model)
    leaves = [node for node in flamo_nodes_flat(root) if node["type"] == "Leaf"]
    return _feedback_leaf_module(leaves)


def extract_build(model: Any) -> FDNBuild:
    """Extract a complete :class:`~pyFDN.FDNBuild` from a named FLAMO model graph.

    The graph must be one :func:`pyFDN.assemble_fdn_core` would build: leaves
    named ``input_gain`` and ``output_gain``, plus either ``mixing_matrix`` or
    the standard recursion feedback leaf ``fB``. The delay can be named
    ``delay`` or be the graph's only delay module. The sample rate is read from
    the delay module and is required: a graph that does not expose ``fs`` is
    malformed and raises :class:`ValueError`.

    All three filter hooks -- ``post_delay``, ``post_matrix``, ``post_output``
    -- are read when present. Because a build is *baked*, a hook must hold
    something that bakes: a per-channel gain or an SOS bank. A hook holding
    anything else -- a nested allpass core, a cascade of several modules -- is
    refused rather than dropped, since a build missing it would render
    differently from the model it came from.
    """
    root = flamo_model_to_nodes(model)
    nodes = flamo_nodes_flat(root)
    leaves = [node for node in nodes if node["type"] == "Leaf"]

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

    def _hook_nodes(name: str) -> list[dict[str, Any]]:
        """Every node belonging to hook ``name``.

        A hook given several modules is a ``Series`` whose leaves
        :func:`pyFDN.hook_module` names ``{name}_0``, ``{name}_1``, ... -- and
        flamo's ``Series`` flattens a nested one, so the container's own name is
        gone by the time the graph is walked. Matching the prefix too is what
        keeps such a hook visible instead of silently reading as empty.
        """
        prefix = f"{name}_"
        return [
            node
            for node in nodes
            if node["name"] == name or node["name"].startswith(prefix)
        ]

    def _hook_sos(name: str) -> np.ndarray | None:
        """The SOS bank in hook ``name``, or ``None`` if the hook is empty."""
        matches = _hook_nodes(name)
        if not matches:
            return None
        node = matches[0]
        if len(matches) > 1 or node["name"] != name or node["type"] != "Leaf":
            raise ValueError(
                f"the {name!r} hook holds {len(matches)} module(s) that do not "
                "bake down to one SOS bank, so an FDNBuild cannot carry it and "
                "would render differently from this model; render the model "
                "directly, or take the hook apart yourself"
            )
        value = _module_value(node["module"])
        if value.ndim == 1:
            return _gain_to_sos(value)
        if value.ndim == 3 and value.shape[1] == 6:
            return np.asarray(value, dtype=float)
        raise ValueError(
            f"the {name!r} hook holds a module whose value has shape "
            f"{tuple(value.shape)}; an FDNBuild can carry a per-channel gain or "
            "an (n_sections, 6, n_channels) SOS bank"
        )

    delay_matches = _named("delay")
    if not delay_matches:
        delay_matches = [
            node for node in leaves if "delay" in type(node["module"]).__name__.lower()
        ]
    if len(delay_matches) != 1:
        raise ValueError(
            "FLAMO graph must contain exactly one delay leaf; "
            f"found {len(delay_matches)}"
        )
    delay_module = delay_matches[0]["module"]
    delays = _delay_samples(delay_module)

    A = np.asarray(_module_value(_feedback_leaf_module(leaves)), dtype=float)
    B = np.asarray(_module_value(_one_of("input_gain")), dtype=float)
    C = np.asarray(_module_value(_one_of("output_gain")), dtype=float)
    if A.ndim != 2:
        raise ValueError("feedback matrix must be a constant two-dimensional matrix")

    direct_matches = _named("direct_gain")
    if not direct_matches:
        direct_matches = [
            node
            for node in _named("brB")
            if node["path"] in {"root/brB", "root/core/brB"}
        ]
    D = (
        np.asarray(_module_value(direct_matches[0]["module"]), dtype=float)
        if len(direct_matches) == 1
        else np.zeros((C.shape[0], B.shape[1]))
    )

    post_delay_sos = _hook_sos("post_delay")
    post_matrix_sos = _hook_sos("post_matrix")
    post_output_sos = _hook_sos("post_output")

    fs_value = getattr(delay_module, "fs", None)
    if fs_value is None:
        raise ValueError("FLAMO delay module exposes no sample rate; cannot extract fs")

    from pyFDN.build import FDNBuild

    return FDNBuild(
        A=A,
        B=B,
        C=C,
        D=D,
        delays=delays,
        fs=float(fs_value),
        post_delay=post_delay_sos,
        post_matrix=post_matrix_sos,
        post_output=post_output_sos,
    )


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
    node at the forward path's input. To draw a saved ``pyfdn-system``
    document without building a FLAMO model, use
    :func:`pyFDN.auxiliary.graph_plot.plot_system_graph`.

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
    from pyFDN.auxiliary.graph_plot import plot_node_tree

    root = flamo_model_to_nodes(model, name=name, include_shell_io=True)
    return plot_node_tree(root, ax=ax, scale=scale, fontsize=fontsize)
