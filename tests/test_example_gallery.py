"""Tests for the generated examples gallery."""

from pathlib import Path

from docs.example_gallery import OUTPUT_FILE, discover_examples, render_gallery
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_gallery_is_up_to_date() -> None:
    expected = render_gallery(discover_examples())
    assert OUTPUT_FILE.read_text(encoding="utf-8") == expected, (
        "docs/examples_gallery.rst is stale; run python3 docs/example_gallery.py"
    )


def test_gallery_contains_every_example_once() -> None:
    gallery = OUTPUT_FILE.read_text(encoding="utf-8")
    for example in discover_examples():
        link = f"_static/marimo/notebooks/{example.output_name}.html"
        assert gallery.count(link) == 1, f"Gallery does not contain {example.path} once"


def test_every_example_has_an_explicit_category() -> None:
    for path in (PROJECT_ROOT / "examples").rglob("example_*.py"):
        header = path.read_text(encoding="utf-8").splitlines()[:20]
        assert any(line.startswith("# gallery_category: ") for line in header), (
            f"{path.relative_to(PROJECT_ROOT)} has no gallery category tag"
        )
