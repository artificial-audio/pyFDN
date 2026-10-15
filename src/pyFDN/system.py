"""``pyfdn-system``: one JSON document for every pyFDN backend.

A system document describes a feedback delay network well enough for any
backend to render it, and well enough to re-generate or re-train it. It has two
tiers, and a document carries exactly one of them:

* ``build`` -- the existing ``pyfdn-fdn-build`` v2 object, unchanged, for a
  vanilla FDN. It is what should be written whenever it is possible.
* ``graph`` -- a tree of :class:`GraphNode` for everything the flat form cannot
  express: time-varying mixing, nested structures, cascaded FIR matrices.

:func:`build_to_graph` lowers a build to a graph; :func:`graph_to_build` lifts a
graph back and refuses, by name, any node that does not bake into an
:class:`~pyFDN.FDNBuild`.

This module is pure data: it imports no backend. Rendering a graph is the job
of the backend converters (``graph_to_td``, ``graph_to_flamo``), which each
raise on a node type they cannot build rather than drop it.

See ``docs/design/system-format.md`` for the rationale.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .build import FDNBuild, fdn_build_from_dict, fdn_build_to_dict

SYSTEM_FORMAT = "pyfdn-system"
SYSTEM_VERSION = 1

ParamKind = Literal["array", "int", "float", "bool", "str", "json"]
"""How a ``params`` entry is stored: an ``array`` is a finite float array; a
``json`` value is copied as is (e.g. a kernel type that is a string or a list).
"""

CONTAINER_TYPES = ("series", "parallel", "recursion")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeType:
    """How to read, validate and count the channels of one leaf ``type``.

    ``params`` maps each parameter to its kind; ``required`` lists the ones a
    node must carry. ``channels`` returns ``(in_channels, out_channels)`` from
    the converted parameters and raises ``ValueError`` on a malformed shape.
    """

    params: Mapping[str, ParamKind]
    required: tuple[str, ...]
    channels: Callable[[Mapping[str, Any]], tuple[int, int]]


def _square(n: int) -> tuple[int, int]:
    return n, n


def _gain_channels(p: Mapping[str, Any]) -> tuple[int, int]:
    matrix = p["matrix"]
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-D, shape (out, in)")
    return matrix.shape[1], matrix.shape[0]


def _matrix_channels(p: Mapping[str, Any]) -> tuple[int, int]:
    in_ch, out_ch = _gain_channels(p)
    if in_ch != out_ch:
        raise ValueError("matrix must be square")
    return in_ch, out_ch


def _matrix_fir_channels(p: Mapping[str, Any]) -> tuple[int, int]:
    coeffs = p["coeffs"]
    if coeffs.ndim != 3:
        raise ValueError("coeffs must be 3-D, shape (out, in, taps)")
    return coeffs.shape[1], coeffs.shape[0]


def _delay_channels(p: Mapping[str, Any]) -> tuple[int, int]:
    samples = p["samples"]
    if samples.ndim != 1 or np.any(samples < 0):
        raise ValueError("samples must be a 1-D array of non-negative delays")
    return _square(samples.size)


def _sos_channels(p: Mapping[str, Any]) -> tuple[int, int]:
    sos = p["sos"]
    if sos.ndim != 3 or sos.shape[1] != 6:
        raise ValueError("sos must have shape (sections, 6, channels)")
    return _square(sos.shape[2])


def _kronecker_channels(p: Mapping[str, Any]) -> tuple[int, int]:
    return _square(1 << np.asarray(p["angles"]).size)


def _channels_param(p: Mapping[str, Any]) -> tuple[int, int]:
    if p["channels"] <= 0:
        raise ValueError("channels must be positive")
    return _square(p["channels"])


def _n_param(p: Mapping[str, Any]) -> tuple[int, int]:
    if p["N"] <= 0 or p["N"] % 2:
        raise ValueError("N must be a positive even integer")
    return _square(p["N"])


NODE_TYPES: dict[str, NodeType] = {
    "identity": NodeType({"channels": "int"}, ("channels",), _channels_param),
    "gain": NodeType({"matrix": "array"}, ("matrix",), _gain_channels),
    "matrix": NodeType({"matrix": "array"}, ("matrix",), _matrix_channels),
    "matrix_fir": NodeType({"coeffs": "array"}, ("coeffs",), _matrix_fir_channels),
    "delay": NodeType({"samples": "array"}, ("samples",), _delay_channels),
    "sos": NodeType({"sos": "array"}, ("sos",), _sos_channels),
    "kronecker_matrix": NodeType(
        {"angles": "array", "kernel_type": "json"}, ("angles",), _kronecker_channels
    ),
    "time_varying_kronecker_matrix": NodeType(
        {
            "angles": "array",
            "rate": "array",
            "depth": "array",
            "phase": "array",
            "kernel_type": "json",
            "waveform": "str",
        },
        ("angles",),
        _kronecker_channels,
    ),
    "time_varying_matrix": NodeType(
        {
            "N": "int",
            "cycles_per_second": "float",
            "amplitude": "float",
            "spread": "float",
        },
        ("N", "cycles_per_second", "amplitude", "spread"),
        _n_param,
    ),
    "dc_blocker": NodeType(
        {
            "channels": "int",
            "R": "float",
            "correct_loss": "bool",
            "env_tau_s": "float",
            "gain_tau_s": "float",
            "max_gain": "float",
        },
        ("channels",),
        _channels_param,
    ),
    "pitch_shift": NodeType(
        {
            "channels": "int",
            "max_delay_samps": "int",
            "window_size": "int",
            "transpose_cents": "float",
            "active_channels": "array",
            "min_delay_samps": "int",
        },
        ("channels", "max_delay_samps", "window_size", "transpose_cents"),
        _channels_param,
    ),
}
"""The leaf vocabulary. Names are pyFDN's own, never a backend's class names."""


def register_node_type(name: str, node_type: NodeType) -> None:
    """Add a leaf type to the registry.

    The registry is open so that a new operator can be saved before every
    backend can render it. Container names and existing types cannot be
    replaced: a document on disk must keep meaning what it meant.
    """
    if name in CONTAINER_TYPES or name in NODE_TYPES:
        raise ValueError(f"node type {name!r} is already defined")
    NODE_TYPES[name] = node_type


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphNode:
    """One node of a ``graph`` document.

    A container (``series``, ``parallel``, ``recursion``) has ``children`` and
    no ``params``; a leaf has ``params`` and no ``children``. A ``recursion``
    has exactly two children, ``(forward, feedback)``.

    ``params`` holds the effective values a renderer needs. ``design`` holds how
    they were made (generator type, seed, raw trainable weights, target RT);
    renderers ignore it. Any node may carry ``design``, including a container:
    a cascaded paraunitary matrix is a ``series`` of ``matrix`` and ``delay``
    stages whose ``design`` records the cascade.

    ``options`` holds container settings: ``sum_output`` for ``parallel``,
    ``delay_position`` for ``recursion``.
    """

    type: str
    name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    design: dict[str, Any] = field(default_factory=dict)
    children: tuple[GraphNode, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def channels(self) -> tuple[int, int]:
        """``(in_channels, out_channels)``, derived from the parameters."""
        return node_channels(self)

    def walk(self, path: str = "") -> Iterator[tuple[str, GraphNode]]:
        """Yield ``(path, node)`` for this node and every descendant."""
        here = f"{path}/{self.name or self.type}"
        yield here, self
        for child in self.children:
            yield from child.walk(here)

    def find(self, name: str) -> GraphNode | None:
        """Return the first node named ``name``, depth first."""
        return next((node for _, node in self.walk() if node.name == name), None)


def node_channels(node: GraphNode) -> tuple[int, int]:
    """``(in_channels, out_channels)`` of a node, checking that children fit."""
    if node.type == "series":
        if not node.children:
            raise ValueError("series needs at least one child")
        counts = [node_channels(child) for child in node.children]
        for i, ((_, out_a), (in_b, _)) in enumerate(
            zip(counts, counts[1:], strict=False)
        ):
            if out_a != in_b:
                raise ValueError(
                    f"series child {i} outputs {out_a} channels but child "
                    f"{i + 1} takes {in_b}"
                )
        return counts[0][0], counts[-1][1]
    if node.type == "parallel":
        if not node.children:
            raise ValueError("parallel needs at least one child")
        counts = [node_channels(child) for child in node.children]
        if len({c[0] for c in counts}) != 1:
            raise ValueError("parallel children must take the same input count")
        if node.options.get("sum_output", True):
            if len({c[1] for c in counts}) != 1:
                raise ValueError("summed parallel children must match output counts")
            return counts[0]
        return counts[0][0], sum(c[1] for c in counts)
    if node.type == "recursion":
        if len(node.children) != 2:
            raise ValueError("recursion needs exactly (forward, feedback)")
        (f_in, f_out), (b_in, b_out) = map(node_channels, node.children)
        if f_out != b_in or b_out != f_in:
            raise ValueError(
                f"recursion forward is {f_in}->{f_out} but feedback is {b_in}->{b_out}"
            )
        return f_in, f_out
    spec = NODE_TYPES.get(node.type)
    if spec is None:
        raise ValueError(f"unknown node type {node.type!r}")
    return spec.channels(node.params)


# ---------------------------------------------------------------------------
# JSON conversion
# ---------------------------------------------------------------------------


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(v) for v in value]
    return value


def _read_param(value: Any, kind: ParamKind, where: str) -> Any:
    if kind == "array":
        array = np.asarray(value, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError(f"{where} must contain only finite values")
        return array
    if kind == "int":
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{where} must be an integer")
        return int(value)
    if kind == "float":
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{where} must be finite")
        return number
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{where} must be true or false")
        return value
    if kind == "str":
        if not isinstance(value, str):
            raise ValueError(f"{where} must be a string")
        return value
    return value


def graph_to_dict(node: GraphNode) -> dict[str, Any]:
    """Convert a :class:`GraphNode` tree to JSON-compatible dictionaries.

    Every node is written with its ``channels`` as ``[in, out]``. They are
    derivable, and are checked on read, so a hand-edited document that no
    longer fits together is reported at the node that broke it.
    """
    in_ch, out_ch = node_channels(node)
    data: dict[str, Any] = {"type": node.type}
    if node.name is not None:
        data["name"] = node.name
    data["channels"] = [in_ch, out_ch]
    data.update(_json_value(node.options))
    if node.type == "recursion":
        forward, feedback = node.children
        data["forward"] = graph_to_dict(forward)
        data["feedback"] = graph_to_dict(feedback)
    elif node.type in CONTAINER_TYPES:
        data["children"] = [graph_to_dict(child) for child in node.children]
    else:
        data["params"] = _json_value(node.params)
    if node.design:
        data["design"] = _json_value(node.design)
    return data


_OPTIONS = {"parallel": ("sum_output",), "recursion": ("delay_position",)}
_NODE_KEYS = {"type", "name", "channels", "params", "design", "children"}


def graph_from_dict(data: Mapping[str, Any], *, path: str = "") -> GraphNode:
    """Construct and validate a :class:`GraphNode` tree from parsed JSON.

    Errors name the path of the offending node, e.g.
    ``/root/brA/feedback_loop/forward``.
    """
    if not isinstance(data, Mapping):
        raise ValueError(f"{path or '/'}: node must be an object")
    node_type = data.get("type")
    name = data.get("name")
    where = f"{path}/{name or node_type}"
    if not isinstance(node_type, str):
        raise ValueError(f"{where}: node needs a string 'type'")
    if name is not None and not isinstance(name, str):
        raise ValueError(f"{where}: 'name' must be a string")
    design = data.get("design", {})
    if not isinstance(design, Mapping):
        raise ValueError(f"{where}: 'design' must be an object")

    options = {key: data[key] for key in _OPTIONS.get(node_type, ()) if key in data}
    allowed = _NODE_KEYS | set(_OPTIONS.get(node_type, ()))
    if node_type == "recursion":
        # fF / fB are FLAMO's names, accepted from adac's IR but never written.
        forward = data.get("forward", data.get("fF"))
        feedback = data.get("feedback", data.get("fB"))
        if forward is None or feedback is None:
            raise ValueError(f"{where}: recursion needs 'forward' and 'feedback'")
        children: tuple[GraphNode, ...] = (
            graph_from_dict(forward, path=where),
            graph_from_dict(feedback, path=where),
        )
        params: dict[str, Any] = {}
        allowed |= {"forward", "feedback", "fF", "fB"}
    elif node_type in CONTAINER_TYPES:
        raw_children = data.get("children")
        if not isinstance(raw_children, list):
            raise ValueError(f"{where}: {node_type} needs a 'children' list")
        children = tuple(graph_from_dict(child, path=where) for child in raw_children)
        params = {}
    else:
        spec = NODE_TYPES.get(node_type)
        if spec is None:
            raise ValueError(f"{where}: unknown node type {node_type!r}")
        raw_params = data.get("params", {})
        if not isinstance(raw_params, Mapping):
            raise ValueError(f"{where}: 'params' must be an object")
        unknown = set(raw_params) - set(spec.params)
        missing = set(spec.required) - set(raw_params)
        if unknown or missing:
            detail = [f"unknown {sorted(unknown)}"] if unknown else []
            detail += [f"missing {sorted(missing)}"] if missing else []
            raise ValueError(f"{where}: {node_type} params: {'; '.join(detail)}")
        params = {
            key: _read_param(value, spec.params[key], f"{where}.params.{key}")
            for key, value in raw_params.items()
        }
        children = ()

    extra = set(data) - allowed
    if extra:
        raise ValueError(f"{where}: unexpected fields {sorted(extra)}")

    node = GraphNode(node_type, name, params, dict(design), children, options)
    try:
        channels = node_channels(node)
    except ValueError as err:
        raise ValueError(f"{where}: {err}") from None
    stored = data.get("channels")
    if stored is not None and list(stored) != list(channels):
        raise ValueError(
            f"{where}: stored channels {list(stored)} disagree with the "
            f"parameters, which give {list(channels)}"
        )
    return node


# ---------------------------------------------------------------------------
# build <-> graph
# ---------------------------------------------------------------------------


_DESIGN_NODE = {
    "feedback_matrix": "mixing_matrix",
    "input_matrix": "input_gain",
    "output_matrix": "output_gain",
    "direct_matrix": "direct_gain",
    "delays": "delay",
}


def build_to_graph(
    build: FDNBuild, *, design: Mapping[str, Mapping[str, Any]] | None = None
) -> GraphNode:
    """Lower an :class:`~pyFDN.FDNBuild` to the canonical FDN graph.

    The node names are the ones :func:`pyFDN.auxiliary.flamo.assemble_fdn_core`
    produces, and they are normative: :func:`graph_to_build` recognises a
    vanilla FDN by them. ``design`` takes the per-component dictionaries of an
    :class:`~pyFDN.FDNPreset` and attaches each to its node.
    """
    # Preset design components are keyed by FDNBuild field, nodes by name.
    by_node = {
        _DESIGN_NODE.get(key, key): value for key, value in (design or {}).items()
    }

    def leaf(node_type: str, name: str, **params: Any) -> GraphNode:
        return GraphNode(node_type, name, params, dict(by_node.get(name, {})))

    def with_hook(first: GraphNode, hook: np.ndarray | None, name: str) -> GraphNode:
        if hook is None:
            return first
        return GraphNode(
            "series", children=(first, leaf("sos", name, sos=np.asarray(hook)))
        )

    forward = with_hook(
        leaf("delay", "delay", samples=np.asarray(build.delays, dtype=float)),
        build.post_delay,
        "post_delay",
    )
    feedback = with_hook(
        leaf("matrix", "mixing_matrix", matrix=np.asarray(build.A)),
        build.post_matrix,
        "post_matrix",
    )
    wet: list[GraphNode] = [
        leaf("gain", "input_gain", matrix=np.asarray(build.B)),
        GraphNode("recursion", "feedback_loop", children=(forward, feedback)),
        leaf("gain", "output_gain", matrix=np.asarray(build.C)),
    ]
    if build.post_output is not None:
        wet.append(leaf("sos", "post_output", sos=np.asarray(build.post_output)))
    return GraphNode(
        "parallel",
        "root",
        children=(
            GraphNode("series", "brA", children=tuple(wet)),
            leaf("gain", "direct_gain", matrix=np.asarray(build.D)),
        ),
        options={"sum_output": True},
    )


def _single_leaf(node: GraphNode, want: str, role: str) -> np.ndarray:
    if node.type != want:
        raise ValueError(
            f"{role} is a {node.type!r} node; an FDNBuild needs a single {want!r}"
        )
    return next(iter(node.params.values()))


def _split_hook(
    path: GraphNode, first_name: str, first_type: str, hook: str
) -> tuple[np.ndarray, np.ndarray | None]:
    if path.type != "series":
        return _single_leaf(path, first_type, first_name), None
    by_name = {child.name: child for child in path.children}
    extra = [
        c.name or c.type for c in path.children if c.name not in (first_name, hook)
    ]
    if extra or first_name not in by_name or len(path.children) > 2:
        raise ValueError(
            f"{first_name} path has nodes an FDNBuild cannot hold: "
            f"{', '.join(map(str, extra)) or 'unexpected layout'}"
        )
    first = _single_leaf(by_name[first_name], first_type, first_name)
    return first, (
        _single_leaf(by_name[hook], "sos", hook) if hook in by_name else None
    )


def graph_to_build(graph: GraphNode, fs: float) -> FDNBuild:
    """Lift a graph back to an :class:`~pyFDN.FDNBuild`, or say why it cannot.

    Accepts exactly the layout :func:`build_to_graph` writes. Anything else --
    a time-varying mixing stage, a nested structure in a hook, an FIR matrix --
    raises ``ValueError`` naming the node that does not bake. Nothing is
    silently dropped.
    """
    if graph.type != "parallel" or len(graph.children) != 2:
        raise ValueError("an FDNBuild graph is parallel(brA, direct_gain)")
    wet, direct = graph.children
    D = _single_leaf(direct, "gain", "direct_gain")
    if wet.type != "series":
        raise ValueError("brA must be a series")
    names = [child.name for child in wet.children]
    if names[:3] != ["input_gain", "feedback_loop", "output_gain"] or names[3:] not in (
        [],
        ["post_output"],
    ):
        raise ValueError(
            "brA must be input_gain, feedback_loop, output_gain[, post_output]; "
            f"got {names}"
        )
    B = _single_leaf(wet.children[0], "gain", "input_gain")
    loop = wet.children[1]
    if loop.type != "recursion":
        raise ValueError("feedback_loop must be a recursion")
    C = _single_leaf(wet.children[2], "gain", "output_gain")
    post_output = (
        _single_leaf(wet.children[3], "sos", "post_output")
        if len(wet.children) == 4
        else None
    )
    forward, feedback = loop.children
    delays, post_delay = _split_hook(forward, "delay", "delay", "post_delay")
    A, post_matrix = _split_hook(feedback, "mixing_matrix", "matrix", "post_matrix")
    if not np.array_equal(delays, np.rint(delays)):
        raise ValueError("an FDNBuild needs integer delays")
    build = FDNBuild(
        A, B, C, D, delays.astype(np.int64), float(fs),
        post_delay=post_delay, post_matrix=post_matrix, post_output=post_output,
    )  # fmt: skip
    fdn_build_from_dict(fdn_build_to_dict(build))  # shape checks, one place
    return build


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


@dataclass
class FDNSystemDoc:
    """A ``pyfdn-system`` document: metadata plus exactly one of build or graph.

    ``render`` holds backend evaluation settings -- ``nfft`` and
    ``alias_decay_db`` for FLAMO, ``block_size`` for ``td`` -- which change how a
    backend approximates the system, never what the system is.

    ``design`` is the per-component design record of a ``build`` document, in
    the vocabulary :class:`~pyFDN.FDNPreset` validates. A ``graph`` document
    keeps its design on the nodes instead.
    """

    sample_rate: float
    build: FDNBuild | None = None
    graph: GraphNode | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    render: dict[str, Any] = field(default_factory=dict)
    design: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.build is None) == (self.graph is None):
            raise ValueError("a system document carries exactly one of build, graph")
        if self.graph is not None and self.design:
            raise ValueError("a graph document keeps its design on the nodes")
        if self.build is not None and float(self.build.fs) != float(self.sample_rate):
            raise ValueError("build.sample_rate disagrees with the document's")

    def as_graph(self) -> GraphNode:
        """The system as a graph, lowering a build when necessary."""
        if self.graph is not None:
            return self.graph
        assert self.build is not None
        return build_to_graph(self.build, design=self.design)

    def as_build(self) -> FDNBuild:
        """The system as a build; raises if the graph is not a vanilla FDN."""
        if self.build is not None:
            return self.build
        assert self.graph is not None
        return graph_to_build(self.graph, self.sample_rate)


RENDER_FIELDS: dict[str, ParamKind] = {
    "nfft": "int",
    "alias_decay_db": "float",
    "block_size": "int",
}


def system_to_dict(doc: FDNSystemDoc) -> dict[str, Any]:
    """Convert an :class:`FDNSystemDoc` to its JSON-compatible form."""
    data: dict[str, Any] = {"format": SYSTEM_FORMAT, "version": SYSTEM_VERSION}
    if doc.metadata:
        data["metadata"] = _json_value(doc.metadata)
    data["sample_rate"] = float(doc.sample_rate)
    if doc.render:
        data["render"] = _json_value(doc.render)
    if doc.build is not None:
        data["build"] = fdn_build_to_dict(doc.build)
        if doc.design:
            data["design"] = _json_value(doc.design)
    else:
        assert doc.graph is not None
        data["graph"] = graph_to_dict(doc.graph)
    system_from_dict(data)
    return data


def system_from_dict(data: Mapping[str, Any]) -> FDNSystemDoc:
    """Construct and validate an :class:`FDNSystemDoc` from parsed JSON."""
    if data.get("format") != SYSTEM_FORMAT:
        raise ValueError(f"Expected format '{SYSTEM_FORMAT}'")
    if data.get("version") != SYSTEM_VERSION:
        raise ValueError(f"Unsupported pyfdn-system version: {data.get('version')!r}")
    extra = set(data) - {
        "format", "version", "metadata", "sample_rate", "render", "build", "graph",
        "design",
    }  # fmt: skip
    if extra:
        raise ValueError(f"unexpected top-level fields {sorted(extra)}")
    sample_rate = _read_param(data.get("sample_rate"), "float", "sample_rate")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object")
    tags = metadata.get("tags")
    if tags is not None and not (
        isinstance(tags, list) and all(isinstance(t, str) for t in tags)
    ):
        raise ValueError("metadata.tags must be a list of strings")
    render_in = data.get("render", {})
    if not isinstance(render_in, Mapping) or set(render_in) - set(RENDER_FIELDS):
        raise ValueError(f"render may only hold {', '.join(RENDER_FIELDS)}")
    render = {
        key: _read_param(value, RENDER_FIELDS[key], f"render.{key}")
        for key, value in render_in.items()
    }
    build = graph = None
    if "build" in data:
        build = fdn_build_from_dict(data["build"])
    if "graph" in data:
        graph = graph_from_dict(data["graph"])
    design = data.get("design", {})
    if design:
        from .preset import _design_from_dict

        design = _design_from_dict(design)
    return FDNSystemDoc(
        sample_rate=sample_rate,
        build=build,
        graph=graph,
        metadata=dict(metadata),
        render=render,
        design=dict(design),
    )


def save_fdn_system(path: str | PathLike[str], doc: FDNSystemDoc) -> None:
    """Write a system document as indented, human-readable JSON.

    ``metadata.created`` and ``metadata.pyfdn_version`` are added unless the
    document already carries them.
    """
    from . import __version__

    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata = {"created": created, "pyfdn_version": __version__, **doc.metadata}
    Path(path).write_text(
        json.dumps(
            system_to_dict(replace(doc, metadata=metadata)), indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def load_fdn_system(path: str | PathLike[str]) -> FDNSystemDoc:
    """Load a system document from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("pyfdn-system JSON must contain an object")
    return system_from_dict(data)


__all__ = [
    "CONTAINER_TYPES",
    "FDNSystemDoc",
    "GraphNode",
    "NODE_TYPES",
    "NodeType",
    "RENDER_FIELDS",
    "SYSTEM_FORMAT",
    "SYSTEM_VERSION",
    "build_to_graph",
    "graph_from_dict",
    "graph_to_build",
    "graph_to_dict",
    "load_fdn_system",
    "node_channels",
    "register_node_type",
    "save_fdn_system",
    "system_from_dict",
    "system_to_dict",
]
