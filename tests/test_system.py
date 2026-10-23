"""Tests for the ``pyfdn-system`` document format (``pyFDN.system``)."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

import pyFDN
from pyFDN.build import FDNBuild, fdn_build_to_dict
from pyFDN.system import (
    FDNSystemDoc,
    GraphNode,
    build_to_graph,
    graph_from_dict,
    graph_to_build,
    graph_to_dict,
    load_fdn_system,
    save_fdn_system,
    system_from_dict,
    system_to_dict,
)


def _build(hooks: bool = True) -> FDNBuild:
    rng = np.random.default_rng(0)
    N = 4
    sos = np.zeros((1, 6, N))
    sos[0, 0] = 0.9
    sos[0, 3] = 1.0
    return FDNBuild(
        A=np.linalg.qr(rng.standard_normal((N, N)))[0],
        B=np.ones((N, 1)),
        C=np.ones((2, N)),
        D=np.zeros((2, 1)),
        delays=np.array([101, 131, 173, 211]),
        fs=48000.0,
        post_delay=sos if hooks else None,
        post_matrix=sos if hooks else None,
        post_output=sos[:, :, :2] if hooks else None,
    )


def _assert_builds_equal(a: FDNBuild, b: FDNBuild) -> None:
    assert fdn_build_to_dict(a) == fdn_build_to_dict(b)


@pytest.mark.parametrize("hooks", [False, True])
def test_build_graph_build_round_trip(hooks):
    build = _build(hooks)
    graph = build_to_graph(build)
    assert graph.channels == (1, 2)
    _assert_builds_equal(graph_to_build(graph, build.fs), build)


def test_graph_json_round_trip_is_stable():
    data = graph_to_dict(build_to_graph(_build()))
    again = graph_to_dict(graph_from_dict(json.loads(json.dumps(data))))
    assert again == data
    assert data["name"] == "root"
    assert data["children"][0]["children"][1]["forward"]["children"][1]["name"] == (
        "post_delay"
    )


@pytest.mark.parametrize("name", pyFDN.available_fdn_presets())
def test_packaged_presets_round_trip_through_graph(name):
    preset = pyFDN.get_fdn_preset(name)
    graph = build_to_graph(preset.build, design=preset.design)
    lifted = graph_to_build(graph_from_dict(graph_to_dict(graph)), preset.build.fs)
    _assert_builds_equal(lifted, preset.build)


def test_preset_design_lands_on_nodes():
    graph = build_to_graph(_build(), design={"feedback_matrix": {"type": "orthogonal"}})
    node = graph.find("mixing_matrix")
    assert node is not None and node.design == {"type": "orthogonal"}


def test_graph_to_build_refuses_time_varying_mixing_by_name():
    graph = build_to_graph(_build(hooks=False))
    wet, direct = graph.children
    loop = wet.children[1]
    tv = GraphNode(
        "time_varying_matrix",
        "post_matrix",
        {"N": 4, "cycles_per_second": 1.2, "amplitude": 0.1, "spread": 0.2},
    )
    feedback = GraphNode("series", children=(loop.children[1], tv))
    loop = GraphNode(
        "recursion", "feedback_loop", children=(loop.children[0], feedback)
    )
    wet = GraphNode("series", "brA", children=(wet.children[0], loop, wet.children[2]))
    graph = GraphNode("parallel", "root", children=(wet, direct), options=graph.options)

    assert graph.channels == (1, 2)
    with pytest.raises(ValueError, match="time_varying_matrix"):
        graph_to_build(graph, 48000.0)


def test_cascaded_matrix_is_a_series_of_stages():
    # A two-stage paraunitary cascade, stored as cascaded rather than as FIR.
    rng = np.random.default_rng(1)
    stage = lambda: GraphNode(  # noqa: E731
        "matrix", params={"matrix": np.linalg.qr(rng.standard_normal((4, 4)))[0]}
    )
    shift = GraphNode("delay", params={"samples": np.array([0.0, 1, 2, 3])})
    cascade = GraphNode(
        "series",
        "mixing_matrix",
        design={"type": "cascaded_paraunitary", "stages": 2, "seed": 1},
        children=(stage(), shift, stage()),
    )
    data = graph_to_dict(cascade)
    assert data["design"]["type"] == "cascaded_paraunitary"
    assert graph_from_dict(data).channels == (4, 4)


def test_stored_channels_are_checked():
    data = graph_to_dict(build_to_graph(_build()))
    data["children"][1]["channels"] = [1, 3]
    with pytest.raises(ValueError, match="/root/direct_gain.*disagree"):
        graph_from_dict(data)


def test_mismatched_series_names_the_node():
    data = graph_to_dict(build_to_graph(_build()))
    data["children"][0]["children"][0]["params"]["matrix"] = [[1.0]] * 3
    with pytest.raises(ValueError, match="/root/brA"):
        graph_from_dict(data)


def test_unknown_type_is_refused_by_name():
    with pytest.raises(ValueError, match="parallelSOSFilter"):
        graph_from_dict({"type": "parallelSOSFilter", "params": {}})


def test_flamo_recursion_aliases_are_read_but_not_written():
    data = graph_to_dict(build_to_graph(_build(hooks=False)))
    loop = data["children"][0]["children"][1]
    loop["fF"] = loop.pop("forward")
    loop["fB"] = loop.pop("feedback")
    written = graph_to_dict(graph_from_dict(data))
    assert "forward" in written["children"][0]["children"][1]
    assert "fF" not in json.dumps(written)


def test_document_build_tier_embeds_build_unchanged(tmp_path):
    build = _build()
    doc = FDNSystemDoc(
        sample_rate=build.fs,
        build=build,
        metadata={"name": "static-N4", "tags": ["test"]},
        render={"nfft": 16384, "block_size": 64},
        design={"feedback_matrix": {"type": "orthogonal"}},
    )
    data = system_to_dict(doc)
    assert data["build"] == fdn_build_to_dict(build)

    path = tmp_path / "static.json"
    save_fdn_system(path, doc)
    loaded = load_fdn_system(path)
    _assert_builds_equal(loaded.as_build(), build)
    assert loaded.render == {"nfft": 16384, "block_size": 64}
    assert loaded.as_graph().find("mixing_matrix").design == {"type": "orthogonal"}


def test_document_graph_tier_round_trips():
    build = _build()
    doc = FDNSystemDoc(sample_rate=build.fs, graph=build_to_graph(build))
    loaded = system_from_dict(json.loads(json.dumps(system_to_dict(doc))))
    _assert_builds_equal(loaded.as_build(), build)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.update(format="adac"), "format"),
        (lambda d: d.update(version=2), "version"),
        (lambda d: d.update(graph={"type": "identity", "params": {"channels": 1}}),
         "exactly one"),
        (lambda d: d.update(render={"nfft": 1024, "hop": 2}), "render"),
        (lambda d: d.update(metadata={"tags": "reverb"}), "tags"),
    ],
)  # fmt: skip
def test_document_validation(mutate, match):
    build = _build(hooks=False)
    data = system_to_dict(FDNSystemDoc(sample_rate=build.fs, build=build))
    data = copy.deepcopy(data)
    mutate(data)
    with pytest.raises(ValueError, match=match):
        system_from_dict(data)


def test_save_stamps_created_and_pyfdn_version(tmp_path):
    build = _build(hooks=False)
    doc = FDNSystemDoc(sample_rate=build.fs, build=build, metadata={"name": "N4"})
    path = tmp_path / "n4.json"
    save_fdn_system(path, doc)

    metadata = load_fdn_system(path).metadata
    assert metadata["name"] == "N4"
    assert metadata["pyfdn_version"] == pyFDN.__version__
    assert metadata["created"].endswith("Z")
    assert doc.metadata == {"name": "N4"}

    # Re-saving keeps when, and by which pyFDN, the document was first made.
    stamped = {"created": "2026-09-27T14:03:00Z", "pyfdn_version": "0.4.1"}
    save_fdn_system(
        path, FDNSystemDoc(sample_rate=build.fs, build=build, metadata=stamped)
    )
    assert load_fdn_system(path).metadata == stamped


def test_non_finite_params_are_refused():
    node = {"type": "gain", "params": {"matrix": [[float("nan")]]}}
    with pytest.raises(ValueError, match="finite"):
        graph_from_dict(node)
