"""Compile a FLAMO model graph into a time-domain operator tree, and run it.

:func:`compile_flamo_graph` walks the same node tree as
:func:`pyFDN.flamo_model_to_nodes` and maps each node to a
:class:`~pyFDN.td.operators.TimeOperator`: ``Series`` -> :class:`Series`,
``Parallel`` -> :class:`Parallel`, ``Recursion`` -> :class:`Recursion`, and each
leaf module to the matching wrapper. The Shell's FFT/iFFT I/O layers carry no
time-domain meaning and are dropped (the core is compiled directly).

:func:`process` is the one-call entry point: compile, then stream the signal
through the tree.

Supported leaves (v1, FDN-essential): ``Gain``/``Matrix`` (constant gains and
feedback matrix), ``parallelDelay``, ``parallelSOSFilter`` (in-loop absorption),
and ``Filter`` (FIR matrix). Other leaf types raise ``NotImplementedError``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyFDN.auxiliary.flamo_graph import (
    _delay_samples,
    _module_value,
    flamo_model_to_nodes,
)
from pyFDN.td.operators import (
    Delay,
    Gain,
    Identity,
    MatrixFIR,
    Parallel,
    Recursion,
    Series,
    SOSBank,
    TimeOperator,
)


def compile_flamo_graph(model: Any) -> TimeOperator:
    """Compile a FLAMO model (Shell/Series/Parallel/Recursion/leaf) to operators.

    Parameters
    ----------
    model
        A FLAMO model, e.g. the output of :func:`pyFDN.dss_to_flamo` /
        :func:`pyFDN.build_to_flamo`.

    Returns
    -------
    TimeOperator
        The root operator; call ``.process(signal)`` or use :func:`process`.
    """
    root = flamo_model_to_nodes(model, include_shell_io=False)
    return _compile_node(root)


def _compile_node(node: dict[str, Any]) -> TimeOperator:
    ntype = node.get("type", "Leaf")

    if ntype == "Shell":
        children = node.get("children") or []
        if len(children) != 1:
            raise ValueError("Shell must wrap exactly one core module")
        # FFT/iFFT input/output layers are identity in the time domain -> dropped.
        return _compile_node(children[0])

    if ntype == "Series":
        children = node.get("children") or []
        return Series([_compile_node(c) for c in children])

    if ntype == "Parallel":
        children = node.get("children") or []
        # dss_to_flamo builds the direct path with sum_output=True.
        return Parallel([_compile_node(c) for c in children], sum_output=True)

    if ntype == "Recursion":
        fF = node.get("fF")
        fB = node.get("fB")
        if fF is None or fB is None:
            raise ValueError("Recursion must have both fF and fB paths")
        return Recursion(_compile_node(fF), _compile_node(fB))

    return _compile_leaf(node)


def _compile_leaf(node: dict[str, Any]) -> TimeOperator:
    module = node.get("module")
    if module is None:
        raise ValueError(f"leaf node {node.get('name')!r} has no module")
    type_name = type(module).__name__
    low = type_name.lower()

    if "delay" in low:
        return Delay(_delay_samples(module))

    if "sos" in low:
        return SOSBank(np.asarray(_module_value(module), dtype=float))

    # FFT/iFFT/Transform layers may appear if a graph nests an I/O layer; in the
    # time domain they are pass-throughs.
    if type_name in {"FFT", "iFFT"} or "transform" in low or "fft" in low:
        channels = node.get("input_channels") or node.get("output_channels") or 1
        return Identity(int(channels))

    if type_name == "Filter" or ("filter" in low and "sos" not in low):
        value = np.asarray(_module_value(module), dtype=float)
        if value.ndim == 3:
            # FLAMO stores (n_taps, n_out, n_in); FIRMatrixFilter wants
            # (n_out, n_in, n_taps).
            return MatrixFIR(np.transpose(value, (1, 2, 0)))
        raise NotImplementedError(
            f"Filter leaf with value ndim {value.ndim} is not supported yet"
        )

    if "gain" in low or type_name == "Matrix":
        return Gain(np.asarray(_module_value(module), dtype=float))

    raise NotImplementedError(
        f"td compiler does not support leaf module {type_name!r} yet"
    )


def process(model: Any, signal: np.ndarray, *, squeeze: bool = True) -> np.ndarray:
    """Render a signal through a FLAMO model in the time domain.

    Compiles ``model`` to a :class:`~pyFDN.td.operators.TimeOperator` tree and
    streams ``signal`` through it. No torch, no FFT -- a pure NumPy/SciPy block
    recursion that lines up sample-for-sample with the FLAMO frequency-domain
    render (see :class:`~pyFDN.td.operators.Recursion`).

    Parameters
    ----------
    model
        A FLAMO model (e.g. from :func:`pyFDN.dss_to_flamo`).
    signal
        Input of shape ``(num_samples,)`` or ``(num_samples, num_inputs)``.
    squeeze
        Squeeze singleton output channels (default ``True``).

    Returns
    -------
    np.ndarray
        Output of shape ``(num_samples, num_outputs)``, squeezed by default.
    """
    op = compile_flamo_graph(model)
    x = np.asarray(signal, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    out = op.process(x)
    return out.squeeze() if squeeze else out
