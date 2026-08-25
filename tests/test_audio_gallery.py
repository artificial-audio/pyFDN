"""Tests for the generated audio gallery."""

from pathlib import Path

from docs.audio_gallery import (
    discover_audio_files,
    generate_audio_gallery,
    render_gallery,
)


def test_gallery_generator_writes_rendered_content(tmp_path: Path) -> None:
    output_file = tmp_path / "audio_gallery.rst"
    expected = render_gallery(discover_audio_files())

    assert generate_audio_gallery(output_file) == expected
    assert output_file.read_text(encoding="utf-8") == expected


def test_gallery_contains_every_example_once() -> None:
    gallery = render_gallery(discover_audio_files())
    for example in discover_audio_files():
        link = example.audio_url
        assert gallery.count(link) == 1, f"Gallery does not contain {example.path} once"
