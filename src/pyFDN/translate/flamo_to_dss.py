"""Read the delay state-space (DSS) system and baked filters back out of a FLAMO model."""

from __future__ import annotations

from typing import Any

import numpy as np

from pyFDN.auxiliary.flamo_graph import (
    _delay_samples,
    _feedback_leaf_module,
    _gain_to_sos,
    _module_value,
    flamo_model_to_nodes,
    flamo_nodes_flat,
)
from pyFDN.dss import DSS


def flamo_to_dss(model: Any) -> tuple[DSS, dict[str, np.ndarray | None]]:
    """Extract the :class:`~pyFDN.DSS` and baked filters from a named FLAMO model.

    Returns ``(dss, filters)``, where ``filters`` maps ``post_delay``,
    ``post_matrix`` and ``post_output`` to an SOS bank or ``None``, ready for
    ``pyFDN.dss_to_td(*dss, **filters)``.

    The graph must be one :func:`pyFDN.assemble_fdn_core` would build: leaves
    named ``input_gain`` and ``output_gain``, plus either ``mixing_matrix`` or
    the standard recursion feedback leaf ``fB``. The delay can be named
    ``delay`` or be the graph's only delay module.

    A hook must hold something that bakes: a per-channel gain or an SOS bank. A
    hook holding anything else -- a nested allpass core, a cascade of several
    modules -- is refused rather than dropped, since the extracted system would
    render differently from the model it came from. Save such a model whole
    with :func:`pyFDN.save_fdn` instead.
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
                "bake down to one SOS bank, so extracting it would render "
                "differently from this model; render or save the model "
                "directly, or take the hook apart yourself"
            )
        value = _module_value(node["module"])
        if value.ndim == 1:
            return _gain_to_sos(value)
        if value.ndim == 3 and value.shape[1] == 6:
            return np.asarray(value, dtype=float)
        raise ValueError(
            f"the {name!r} hook holds a module whose value has shape "
            f"{tuple(value.shape)}; only a per-channel gain or an "
            "(n_sections, 6, n_channels) SOS bank can be extracted"
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

    filters = {
        name: _hook_sos(name) for name in ("post_delay", "post_matrix", "post_output")
    }
    return DSS(delays, A, B, C, D), filters


__all__ = ["flamo_to_dss"]
