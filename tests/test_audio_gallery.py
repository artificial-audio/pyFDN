"""Tests for the generated audio gallery."""

from pathlib import Path

from docs.audio_gallery import OUTPUT_FILE, discover_audio_files, render_gallery


def test_gallery_is_up_to_date() -> None:
    expected = render_gallery(discover_audio_files())
    assert OUTPUT_FILE.read_text(encoding="utf-8") == expected, (
        "docs/audio_gallery.rst is stale; run python3 docs/audio_gallery.py"
    )


def test_gallery_contains_every_example_once() -> None:
    gallery = OUTPUT_FILE.read_text(encoding="utf-8")
    for example in discover_audio_files():
        link = example.audio_url
        assert gallery.count(link) == 1, f"Gallery does not contain {example.path} once"