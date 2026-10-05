"""Tests for controlled FDN preset documents."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

import pyFDN


def _build(*, post_delay=None, post_matrix=None, post_output=None):
    return pyFDN.FDNBuild(
        A=np.array([[0.0, -1.0], [1.0, 0.0]]),
        B=np.ones((2, 1)) / np.sqrt(2.0),
        C=np.ones((1, 2)) / np.sqrt(2.0),
        D=np.zeros((1, 1)),
        delays=np.array([13, 17]),
        fs=48_000.0,
        post_delay=post_delay,
        post_matrix=post_matrix,
        post_output=post_output,
    )


def test_fdn_preset_json_round_trip(tmp_path) -> None:
    preset = pyFDN.FDNPreset(
        build=_build(
            post_delay=np.ones((1, 6, 2)),
            post_matrix=np.ones((1, 6, 2)),
            post_output=np.ones((11, 6, 1)),
        ),
        metadata={
            "name": "test-room",
            "description": "A test preset",
            "authors": ["A. Author", "B. Author"],
            "license": "CC0-1.0",
            "source": "https://example.com/test-room",
            "tags": ["room", "test"],
        },
        design={
            "delays": {
                "type": "geometric",
                "range": (10, 20),
                "coprime": True,
                "sort": False,
            },
            "feedback_matrix": {"type": "orthogonal"},
            "input_matrix": {"type": "normalised"},
            "output_matrix": {"type": "ones"},
            "post_delay": {
                "type": "first_order_shelf",
                "rt": 1.2,
                "rt_nyquist": 0.8,
                "rt_crossover": 4_000,
            },
            "post_matrix": {
                "type": "one_pole",
                "gain_db": 0.0,
                "gain_db_nyquist": -1.0,
            },
            "post_output": {
                "type": "graphic_eq",
                "gain_db": np.arange(10.0),
            },
        },
    )
    path = tmp_path / "test-room.json"

    pyFDN.save_fdn_preset(path, preset)
    encoded = json.loads(path.read_text(encoding="utf-8"))
    assert set(encoded) == {"metadata", "design", "build"}
    assert encoded["build"]["sample_rate"] == 48_000.0
    assert encoded["design"]["delays"] == {
        "type": "geometric",
        "range": [10, 20],
        "coprime": True,
        "sort": False,
    }
    assert encoded["design"]["post_delay"]["type"] == "first_order_shelf"
    assert encoded["design"]["post_matrix"]["type"] == "one_pole"
    assert "hooks" not in encoded["design"]
    assert "construction" not in json.dumps(encoded)

    restored = pyFDN.load_fdn_preset(path)
    assert restored.metadata == preset.metadata
    assert restored.design["delays"] == {
        "type": "geometric",
        "range": [10, 20],
        "coprime": True,
        "sort": False,
    }
    assert restored.design["feedback_matrix"] == {"type": "orthogonal"}
    assert restored.design["post_delay"]["rt"] == 1.2
    assert restored.design["post_delay"]["rt_nyquist"] == 0.8
    for name in (
        "A",
        "B",
        "C",
        "D",
        "delays",
        "post_delay",
        "post_matrix",
        "post_output",
    ):
        np.testing.assert_array_equal(
            getattr(restored.build, name), getattr(preset.build, name)
        )
    assert restored.build.fs == preset.build.fs


def test_unknown_design_information_is_omitted() -> None:
    preset = pyFDN.FDNPreset(
        build=_build(),
        metadata={"name": "unknown-origin"},
        design={"delays": {"coprime": False}},
    )

    encoded = pyFDN.fdn_preset_to_dict(preset)
    assert encoded["design"] == {"delays": {"coprime": False}}


def test_metadata_is_open_but_tags_remain_easy_to_filter() -> None:
    preset = pyFDN.FDNPreset(
        build=_build(),
        metadata={
            "name": "tagged",
            "tags": ["room", "short"],
            "catalog_note": {"reviewed": True},
        },
    )

    assert {"room", "short"} <= set(preset.metadata.get("tags", []))
    assert preset.metadata["catalog_note"] == {"reviewed": True}


def test_preset_rejects_unknown_vocabulary_and_inconsistent_hooks() -> None:
    data = pyFDN.fdn_preset_to_dict(
        pyFDN.FDNPreset(build=_build(), metadata={"name": "strict"})
    )
    data["design"]["feedback_matrix"] = {"type": "random_orthogonal"}
    with pytest.raises(ValueError, match="design.feedback_matrix.type"):
        pyFDN.fdn_preset_from_dict(data)


def test_packaged_build_is_available_as_a_preset() -> None:
    preset = pyFDN.get_fdn_preset("colorless_N4_d1")
    assert preset.metadata["name"] == "colorless_N4_d1"
    assert preset.metadata["description"] == "Optimized colorless FDN build."
    assert preset.metadata["license"] == "MIT"
    assert "optimized" in preset.metadata["tags"]
    assert preset.design == {"feedback_matrix": {"type": "orthogonal"}}
    assert preset.build.fs == 48_000.0


def test_gallery_design_round_trips_into_trainable_filter_targets() -> None:
    torch = pytest.importorskip("torch")
    build, design = pyFDN.fdn_build_gallery(
        2,
        rt=1.2,
        rt_nyquist=0.8,
        rt_crossover=4_000,
        output_gain_db=0.0,
        output_gain_db_nyquist=-3.0,
        output_crossover=6_000,
        rng=4,
        return_design=True,
    )
    preset = pyFDN.fdn_preset_from_dict(
        pyFDN.fdn_preset_to_dict(
            pyFDN.FDNPreset(build=build, design=design, metadata={"name": "gallery"})
        )
    )

    model = pyFDN.trainable_from_preset(
        preset,
        matrix="random",
        nfft=256,
        dtype=torch.float64,
    )

    np.testing.assert_allclose(
        pyFDN.param(model, "post_delay").raw().detach().cpu().numpy(),
        [1.2, 0.8],
    )
    np.testing.assert_allclose(
        pyFDN.param(model, "post_output").raw().detach().cpu().numpy(),
        [[0.0], [-3.0]],
    )


def test_trainable_from_preset_recovers_meaningful_filter_targets() -> None:
    torch = pytest.importorskip("torch")
    nfft = 256
    base = _build()
    decay = pyFDN.AttenuationFilter(
        1.2,
        base.delays,
        base.fs,
        rt_nyquist=0.8,
        design="first_order_shelf",
        rt_crossover=4_000,
        nfft=nfft,
        dtype=torch.float64,
        requires_grad=False,
    )
    output = pyFDN.OutputEQ(
        0.0,
        base.C.shape[0],
        base.fs,
        gain_db_nyquist=-3.0,
        design="first_order_shelf",
        crossover=6_000,
        nfft=nfft,
        dtype=torch.float64,
        requires_grad=False,
    )
    matrix = pyFDN.OutputEQ(
        0.0,
        base.A.shape[0],
        base.fs,
        gain_db_nyquist=-1.0,
        design="one_pole",
        nfft=nfft,
        dtype=torch.float64,
        requires_grad=False,
    )
    build = replace(
        base,
        post_delay=decay.map(decay.param).detach().cpu().numpy(),
        post_matrix=matrix.map(matrix.param).detach().cpu().numpy(),
        post_output=output.map(output.param).detach().cpu().numpy(),
    )
    preset = pyFDN.FDNPreset(
        build=build,
        metadata={"name": "trainable"},
        design={
            "post_delay": {
                "type": "first_order_shelf",
                "rt": 1.2,
                "rt_nyquist": 0.8,
                "rt_crossover": 4_000,
            },
            "post_matrix": {
                "type": "one_pole",
                "gain_db": 0.0,
                "gain_db_nyquist": -1.0,
            },
            "post_output": {
                "type": "first_order_shelf",
                "gain_db": 0.0,
                "gain_db_nyquist": -3.0,
                "crossover": 6_000,
            },
        },
    )

    model = pyFDN.trainable_from_preset(
        preset,
        trainable_hooks=("post_delay", "post_matrix", "post_output"),
        nfft=nfft,
        dtype=torch.float64,
    )

    decay_ref = pyFDN.param(model, "post_delay")
    matrix_ref = pyFDN.param(model, "post_matrix")
    output_ref = pyFDN.param(model, "post_output")
    assert decay_ref.trainable
    assert matrix_ref.trainable
    assert output_ref.trainable
    assert decay_ref.module.design == "first_order_shelf"
    assert output_ref.module.crossover == 6_000
    assert matrix_ref.module.design == "one_pole"
    np.testing.assert_allclose(decay_ref.raw().detach().cpu().numpy(), [1.2, 0.8])
    np.testing.assert_allclose(
        pyFDN.extract_build(model).post_output, build.post_output, rtol=1e-12
    )


def test_trainable_from_preset_rejects_a_target_that_changes_the_build() -> None:
    torch = pytest.importorskip("torch")
    build = _build(post_delay=np.ones((1, 6, 2)))
    preset = pyFDN.FDNPreset(
        build=build,
        metadata={"name": "mismatch"},
        design={
            "post_delay": {
                "type": "first_order_shelf",
                "rt": 1.2,
                "rt_nyquist": 0.8,
            }
        },
    )

    with pytest.raises(ValueError, match="does not reproduce build.post_delay"):
        pyFDN.trainable_from_preset(preset, nfft=256, dtype=torch.float64)
