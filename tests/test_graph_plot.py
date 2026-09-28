"""Tests for backend-independent signal-flow diagrams (``graph_plot``)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from pyFDN.auxiliary.graph_plot import (  # noqa: E402
    graph_to_plot_nodes,
    plot_system_graph,
)
from pyFDN.build import FDNBuild  # noqa: E402
from pyFDN.system import FDNSystemDoc, GraphNode, build_to_graph  # noqa: E402


def _build() -> FDNBuild:
    N = 4
    sos = np.zeros((1, 6, N))
    sos[0, 0] = 0.9
    sos[0, 3] = 1.0
    return FDNBuild(
        A=np.eye(N),
        B=np.ones((N, 1)),
        C=np.ones((1, N)),
        D=np.zeros((1, 1)),
        delays=np.array([101, 131, 173, 211]),
        fs=48000.0,
        post_delay=sos,
    )


def _texts(ax) -> set[str]:
    return {t.get_text() for t in ax.texts}


@pytest.mark.parametrize("as_doc", [False, True])
def test_plot_system_graph_draws_a_build_without_a_backend(as_doc):
    build = _build()
    system = FDNSystemDoc(sample_rate=build.fs, build=build) if as_doc else build
    ax = plot_system_graph(system)
    labels = _texts(ax)
    assert {"input_gain\ngain", "delay", "post_delay\nsos"} <= labels
    assert "mixing_matrix\nmatrix" in labels
    assert "feedback_loop" in labels
    plt.close(ax.figure)


def test_plot_system_graph_draws_types_no_backend_renders():
    tv = GraphNode(
        "time_varying_matrix",
        "post_matrix",
        {"N": 4, "cycles_per_second": 1.2, "amplitude": 0.1, "spread": 0.2},
    )
    ax = plot_system_graph(GraphNode("series", children=(tv,)))
    assert "post_matrix\ntime_varying_matrix" in _texts(ax)
    plt.close(ax.figure)


def test_graph_to_plot_nodes_keeps_structure():
    root = graph_to_plot_nodes(build_to_graph(_build()))
    assert root["type"] == "Parallel" and root["sum_output"] is True
    wet, direct = root["children"]
    assert direct["kind"] == "gain"
    loop = wet["children"][1]
    assert loop["type"] == "Recursion"
    assert [c["name"] for c in loop["fF"]["children"]] == ["delay", "post_delay"]
    assert loop["fB"]["kind"] == "matrix"


def test_non_summing_parallel_is_drawn_as_a_stack():
    branch = GraphNode("gain", "g", {"matrix": np.ones((1, 1))})
    summed = plot_system_graph(GraphNode("parallel", children=(branch, branch)))
    stacked = plot_system_graph(
        GraphNode("parallel", children=(branch, branch), options={"sum_output": False})
    )
    assert "+" in _texts(summed)
    assert "+" not in _texts(stacked)
    plt.close("all")


def test_plot_flamo_graph_uses_the_same_renderer():
    pytest.importorskip("flamo")
    from pyFDN import build_to_flamo, plot_flamo_graph

    ax = plot_flamo_graph(build_to_flamo(_build(), nfft=256))
    assert "feedback_loop" in _texts(ax)
    plt.close(ax.figure)
