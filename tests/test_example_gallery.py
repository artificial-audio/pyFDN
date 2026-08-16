"""Tests for the generated examples gallery."""

from pathlib import Path

from docs.example_gallery import discover_examples, generate_gallery, render_gallery

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_gallery_generator_writes_rendered_content(tmp_path: Path) -> None:
    output_file = tmp_path / "examples_gallery.rst"
    expected = render_gallery(discover_examples())

    assert generate_gallery(output_file) == expected
    assert output_file.read_text(encoding="utf-8") == expected


def test_gallery_contains_every_example_once() -> None:
    gallery = render_gallery(discover_examples())
    for example in discover_examples():
        link = f"_static/marimo/notebooks/{example.output_name}.html"
        assert gallery.count(link) == 1, f"Gallery does not contain {example.path} once"


def test_every_example_has_explicit_gallery_metadata() -> None:
    for path in (PROJECT_ROOT / "examples").rglob("example_*.py"):
        header = path.read_text(encoding="utf-8").splitlines()[:20]
        for key in ("category", "description"):
            assert any(line.startswith(f"# gallery_{key}: ") for line in header), (
                f"{path.relative_to(PROJECT_ROOT)} has no gallery {key} tag"
            )


def test_gallery_descriptions_are_complete_sentences() -> None:
    for example in discover_examples():
        assert example.description != "Open the rendered marimo notebook."
        assert example.description.endswith((".", "!", "?")), example.path
