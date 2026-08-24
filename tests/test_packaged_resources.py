"""Tests for resources that must work from an installed pyFDN wheel."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import pyFDN
from pyFDN.auxiliary import audio as audio_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_packaged_audio_manifest_and_files_agree() -> None:
    names = pyFDN.available_audio()
    assert names == (
        "Kleiderschrank_Taylor_DI",
        "P1_Set1_8",
        "P20_Set1_8",
        "Sommerhit2016_Taylor_DI",
        "drums",
        "s3_r4_o",
        "synth_dry",
    )
    for name in names:
        signal, fs = pyFDN.load_audio(name)
        metadata = pyFDN.audio_metadata(name)
        assert signal.ndim == 1
        assert signal.size > 0
        assert np.isfinite(signal).all()
        assert fs > 0
        assert metadata["license"]
        assert metadata["source_url"]


def test_load_audio_accepts_extension_and_resamples() -> None:
    original, original_fs = pyFDN.load_audio("synth_dry.wav")
    target_fs = original_fs // 2
    resampled, returned_fs = pyFDN.load_audio("synth_dry", fs=target_fs)
    assert returned_fs == target_fs
    assert len(resampled) == round(len(original) * target_fs / original_fs)


def test_load_audio_passes_string_path_to_soundfile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def read(path: str, *, dtype: str) -> tuple[np.ndarray, int]:
        assert isinstance(path, str)
        assert dtype == "float64"
        return np.ones(4), 48_000

    monkeypatch.setattr(audio_module.sf, "read", read)
    signal, fs = pyFDN.load_audio("synth_dry")

    np.testing.assert_array_equal(signal, np.ones(4))
    assert fs == 48_000


def test_unknown_audio_lists_choices() -> None:
    with pytest.raises(ValueError, match="Available samples"):
        pyFDN.load_audio("not-a-sample")


def test_colorless_presets_are_packaged_as_fdn_builds() -> None:
    names = pyFDN.available_fdn_presets()
    assert len(names) == 16
    assert "colorless_N4_d1" in names
    assert "colorless_init_N16_d2" in names

    build = pyFDN.load_fdn_preset("colorless_N4_d1")
    assert build.A.shape == (4, 4)
    assert build.B.shape == (4, 1)
    assert build.C.shape == (1, 4)
    assert build.D.shape == (1, 1)
    assert build.delays.shape == (4,)
    np.testing.assert_allclose(build.A.T @ build.A, np.eye(4), atol=1e-12)

    with_extension = pyFDN.load_fdn_preset("colorless_N4_d1.json", fs=44_100)
    np.testing.assert_array_equal(with_extension.A, build.A)
    assert with_extension.fs == 44_100


def test_every_example_citation_resolves_from_packaged_bibliography() -> None:
    paper_ids: set[str] = set()
    for example in (PROJECT_ROOT / "examples").rglob("*.py"):
        paper_ids.update(
            re.findall(r'paper_link\(["\']([^"\']+)["\']\)', example.read_text(encoding="utf-8"))
        )

    assert paper_ids
    for paper_id in paper_ids:
        reference = pyFDN.paper_reference(paper_id)
        link = pyFDN.paper_link(paper_id)
        assert reference["title"] in link
        assert reference.get("url", "#") in link
