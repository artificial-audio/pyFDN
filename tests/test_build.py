"""Tests for FDNBuild and its readable exchange format."""

from __future__ import annotations

import json

import numpy as np
import pytest

import pyFDN


def test_fdn_build_json_round_trip(tmp_path) -> None:
    build = pyFDN.FDNBuild(
        A=np.array([[0.0, 1.0], [1.0, 0.0]]),
        B=np.array([[1.0], [0.5]]),
        C=np.array([[0.25, 0.75]]),
        D=np.zeros((1, 1)),
        delays=np.array([101, 149]),
        fs=48_000.0,
        post_delay=np.ones((1, 6, 2)),
        post_matrix=np.full((1, 6, 2), 0.5),
        post_output=np.ones((1, 6, 1)),
    )
    path = tmp_path / "build.json"

    pyFDN.save_fdn_build(path, build, metadata={"description": "Test build"})
    encoded = json.loads(path.read_text(encoding="utf-8"))
    assert encoded["format"] == "pyfdn-fdn-build"
    assert encoded["version"] == 2
    assert encoded["metadata"]["description"] == "Test build"

    restored = pyFDN.load_fdn_build(path)
    for field in (
        "A",
        "B",
        "C",
        "D",
        "delays",
        "post_delay",
        "post_matrix",
        "post_output",
    ):
        np.testing.assert_array_equal(getattr(restored, field), getattr(build, field))
    assert restored.fs == build.fs


def test_fdn_build_sample_rate_override() -> None:
    build = pyFDN.load_fdn_preset("colorless_N4_d1")
    data = pyFDN.fdn_build_to_dict(build)
    restored = pyFDN.fdn_build_from_dict(data, fs=96_000)
    assert restored.fs == 96_000


def test_fdn_build_rejects_unknown_version() -> None:
    build = pyFDN.load_fdn_preset("colorless_N4_d1")
    data = pyFDN.fdn_build_to_dict(build)
    data["version"] = 999
    with pytest.raises(ValueError, match="Unsupported FDN build version"):
        pyFDN.fdn_build_from_dict(data)
