"""Name a single trainable parameter of a model, so a penalty can point at it.

A cost on model parameters has to say *which* parameter. :func:`param` resolves
a name against a model's FLAMO graph once, at the line you write it, and returns
a :class:`ParamRef` whose :meth:`~ParamRef.value` is the live, differentiable
tensor. :func:`params` lists what a model offers -- the answer to "what can I put
a cost on?" for a non-standard FDN, where guessing the feedback matrix from the
graph structure would be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

# Semantic aliases for FLAMO's structural leaf names. A ``Recursion``'s feedback
# branch is called "fB" and a ``Parallel``'s second branch "brB"; those are
# positions in the graph, not what the parameter *is*. Both spellings resolve.
_ALIASES: dict[str, tuple[str, ...]] = {
    "feedback": ("mixing_matrix", "fB"),
    "delay": ("delay", "fF"),
    "direct": ("direct_gain", "brB"),
    # The three filter hooks, under the names assemble_fdn_core gives them.
    # "absorption" and "post_eq" are alternative spellings of the first and the
    # last, naming what the filter does rather than where it sits.
    "post_delay": ("post_delay", "attenuation", "filter"),
    "post_matrix": ("post_matrix",),
    "post_output": ("post_output", "output_filter"),
    "absorption": ("post_delay", "attenuation", "filter"),
    "post_eq": ("post_output", "output_filter"),
}


@dataclass(frozen=True)
class ParamRef:
    """A reference to one parameter of one model.

    Attributes
    ----------
    name : str
        The name it was resolved under.
    module : flamo module
        The module holding the parameter. Bound at construction, so the ref keeps
        pointing at the same parameter no matter what a loss does with it.
    """

    name: str
    module: Any

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the *mapped* value (what a penalty actually sees)."""
        return tuple(self.value().shape)

    @property
    def trainable(self) -> bool:
        return bool(self.module.param.requires_grad)

    def value(self) -> torch.Tensor:
        """The parameter's mapped value, still attached to the autograd graph.

        FLAMO stores a raw parameter and a ``map`` onto the value the system
        uses -- e.g. a skew-symmetric matrix mapped onto SO(N). A penalty wants
        the mapped value (the actual feedback matrix), not the raw parameter.
        """
        return self.module.map(self.module.param)

    def raw(self) -> torch.Tensor:
        """The parameter *before* the map -- what the optimizer actually steps.

        Usually the mapped :meth:`value` is what you want. The pre-image is,
        when the map is the point: the RT in seconds behind an absorption
        filter (:class:`~pyFDN.DecayFilter`), where the mapped value is the SOS
        bank designed from it.
        """
        return self.module.param

    def __repr__(self) -> str:
        state = "trainable" if self.trainable else "frozen"
        return f"ParamRef({self.name!r}, {tuple(self.shape)}, {state})"


def params(model: Any) -> list[ParamRef]:
    """Every parameter of ``model``, in graph order.

    Use it to see what a model exposes before writing a penalty::

        >>> for p in pyFDN.params(model):  # doctest: +SKIP
        ...     print(p)
        ParamRef('input_gain', (8, 1), trainable)
        ParamRef('fF', (8,), frozen)
        ParamRef('fB', (8, 8), trainable)
        ParamRef('output_gain', (1, 8), trainable)
    """
    from pyFDN.auxiliary.flamo_graph import flamo_model_to_nodes, flamo_nodes_flat

    return [
        ParamRef(name=node["name"], module=node["module"])
        for node in flamo_nodes_flat(flamo_model_to_nodes(model))
        if node["type"] == "Leaf" and getattr(node["module"], "param", None) is not None
    ]


def param(model: Any, name: str | None = None) -> ParamRef:
    """Reference the parameter called ``name`` in ``model``.

    Parameters
    ----------
    model : flamo Shell, or a flamo module
        The model to resolve against. Pass a module directly (with ``name``
        omitted) to reference it without any lookup -- the escape hatch for a
        graph whose leaf names you do not control.
    name : str, optional
        A leaf name, or one of the semantic aliases ``"feedback"`` (the feedback
        matrix, FLAMO's ``fB``), ``"delay"`` (``fF``), ``"direct"`` (``brB``),
        and the three filter hooks ``"post_delay"`` (the in-loop filter, i.e.
        the decay), ``"post_matrix"`` and ``"post_output"`` (the output EQ).
        ``"absorption"`` and ``"post_eq"`` also resolve to the first and last
        of those.

    Raises
    ------
    ValueError
        If the name matches no parameter, or more than one. The message lists
        what the model does offer.
    """
    if name is None:
        if getattr(model, "param", None) is None:
            raise ValueError(
                "param(module) needs a module with a .param; pass "
                "param(model, name) to look a name up in a model instead."
            )
        return ParamRef(name=type(model).__name__, module=model)

    available = params(model)
    candidates = _ALIASES.get(name, (name,))
    for candidate in candidates:
        matches = [p for p in available if p.name == candidate]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"{name!r} matches {len(matches)} parameters named "
                f"{candidate!r}; pass the module itself with param(module)."
            )
    raise ValueError(
        f"no parameter named {name!r} in this model; available: "
        f"{[p.name for p in available]} (aliases: {sorted(_ALIASES)})"
    )
