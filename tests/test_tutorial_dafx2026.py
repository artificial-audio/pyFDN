"""Guard the DAFx 2026 tutorial deck against API drift.

``docs/tutorials/dafx2026/slides.qmd`` renders without executing any Python (see
that directory's README), which makes the deck durable but also means a code
listing can go stale without anything failing. These tests close the obvious
half of that gap: every ``pyFDN.<name>`` the deck shows must still exist, and
every figure it references must still be on disk.

They deliberately do **not** check call signatures or argument names — that would
mean parsing the listings as code, and the listings are elided for slides.

Once the tutorial has been given, the deck becomes a historical record and
should stop constraining the API. At that point delete this module (see the
"Freezing after the conference" section of the tutorial README).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pyFDN

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent / "docs" / "tutorials" / "dafx2026"
)
FIGURE_DIR = TUTORIAL_DIR / "figures" / "out"
QMD_FILES = ("slides.qmd", "index.qmd")

# `pyFDN.thing` in a code listing or in prose. Attribute access only: the
# trailing `[A-Za-z_]` requirement means prose like ``pyFDN.<thing>`` and shell
# lines like ``pyfdn[examples]`` are not picked up.
_ATTRIBUTE = re.compile(r"\bpyFDN\.([A-Za-z_][A-Za-z0-9_]*)")

# Markdown images and reveal.js background images, e.g. ![](figures/out/ir.svg)
# and {background-image="figures/out/diagram.svg"}.
_FIGURE = re.compile(r"(?:!\[[^\]]*\]\(|background-image=\")(figures/out/[^)\"]+)")

# Notebooks the deck sends people to, e.g. `example_vanilla_FDN` — and the
# wildcard form `example_allpass_FDN_*`, which stands for a family of them.
_NOTEBOOK = re.compile(r"`(example_[A-Za-z0-9_]*\*?)`")

# Notebooks the deck references that are not on this branch yet. Each entry is a
# promise: the deck is wrong until the notebook lands.
#   - example_fdn_to_faust: arrives with the adac / FLAMO_to_FAUST work
#     (branch FLAMO_to_FAUST-example). Delete this entry once it is merged.
PENDING_NOTEBOOKS = {"example_fdn_to_faust"}


def _sources() -> dict[str, str]:
    return {
        name: (TUTORIAL_DIR / name).read_text(encoding="utf-8") for name in QMD_FILES
    }


def test_tutorial_sources_exist() -> None:
    for name in QMD_FILES:
        assert (TUTORIAL_DIR / name).is_file(), f"missing {name}"


def test_every_referenced_name_is_public_api() -> None:
    """A rename in pyFDN must not leave the deck teaching a function that is gone."""
    missing: list[str] = []
    for source_name, text in _sources().items():
        for line_number, line in enumerate(text.splitlines(), start=1):
            for attribute in _ATTRIBUTE.findall(line):
                if not hasattr(pyFDN, attribute):
                    missing.append(
                        f"{source_name}:{line_number}: pyFDN.{attribute} — {line.strip()}"
                    )

    assert not missing, (
        "the tutorial references names that no longer exist:\n" + "\n".join(missing)
    )


def test_every_referenced_figure_exists() -> None:
    missing: list[str] = []
    for source_name, text in _sources().items():
        for reference in _FIGURE.findall(text):
            if not (TUTORIAL_DIR / reference).is_file():
                missing.append(f"{source_name}: {reference}")

    assert not missing, (
        "the tutorial references figures that are not on disk:\n" + "\n".join(missing)
    )


def test_every_referenced_notebook_exists() -> None:
    """The deck sends the room to these notebooks; a rename must not silently strand them."""
    examples_dir = TUTORIAL_DIR.parent.parent.parent / "examples"
    on_disk = {path.stem for path in examples_dir.rglob("example_*.py")}

    missing: list[str] = []
    for source_name, text in _sources().items():
        for reference in sorted(set(_NOTEBOOK.findall(text))):
            if reference.endswith("*"):
                found = any(name.startswith(reference[:-1]) for name in on_disk)
            else:
                found = reference in on_disk
            if not found and reference not in PENDING_NOTEBOOKS:
                missing.append(f"{source_name}: {reference}")

    assert not missing, (
        "the tutorial links to notebooks that do not exist:\n" + "\n".join(missing)
    )


def test_pending_notebooks_are_still_pending() -> None:
    """Once a pending notebook lands, drop it from PENDING_NOTEBOOKS so it is really checked."""
    examples_dir = TUTORIAL_DIR.parent.parent.parent / "examples"
    on_disk = {path.stem for path in examples_dir.rglob("example_*.py")}
    landed = sorted(PENDING_NOTEBOOKS & on_disk)
    assert not landed, (
        f"these notebooks now exist — remove them from PENDING_NOTEBOOKS: {', '.join(landed)}"
    )


def test_no_orphan_figures() -> None:
    """Every generated figure is used, so `make figures` output stays meaningful."""
    referenced = {
        reference for text in _sources().values() for reference in _FIGURE.findall(text)
    }
    orphans = sorted(
        path.name
        for path in FIGURE_DIR.glob("*.svg")
        if f"figures/out/{path.name}" not in referenced
    )
    assert not orphans, f"figures generated but never shown: {', '.join(orphans)}"


def test_figure_script_covers_every_figure() -> None:
    """`make_figures.py` must be able to regenerate every checked-in figure."""
    script = (TUTORIAL_DIR / "figures" / "make_figures.py").read_text(encoding="utf-8")
    defined = set(re.findall(r"^def fig_([a-z_]+)\(", script, flags=re.MULTILINE))
    on_disk = {path.stem for path in FIGURE_DIR.glob("*.svg")}

    unbuildable = sorted(on_disk - defined)
    assert not unbuildable, (
        f"figures with no fig_* function to rebuild them: {', '.join(unbuildable)}"
    )


PUBLISH_DIR = TUTORIAL_DIR.parent.parent / "_static" / "tutorials" / "dafx2026"
REPO_ROOT = TUTORIAL_DIR.parent.parent.parent


def test_rendered_output_is_published() -> None:
    """The committed HTML is what the website serves — Sphinx never runs Quarto."""
    missing = [
        str((PUBLISH_DIR / f"{Path(name).stem}.html").relative_to(REPO_ROOT))
        for name in sorted(QMD_FILES)
        if not (PUBLISH_DIR / f"{Path(name).stem}.html").is_file()
    ]
    assert not missing, (
        f"not published: {', '.join(missing)} — "
        "run `make publish` in docs/tutorials/dafx2026/"
    )


def test_published_output_is_not_stale() -> None:
    """Editing a slide without republishing must fail here, not ship quietly.

    Compares the digest recorded by ``make publish`` against the current sources.
    Content-based rather than mtime-based, because a git checkout does not
    preserve modification times.
    """
    spec = importlib.util.spec_from_file_location(
        "_tutorial_sources_digest", TUTORIAL_DIR / "sources_digest.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    recorded_file = PUBLISH_DIR / "sources.sha256"
    assert recorded_file.is_file(), (
        f"{recorded_file.relative_to(REPO_ROOT)} is missing — "
        "run `make publish` in docs/tutorials/dafx2026/"
    )

    recorded = recorded_file.read_text(encoding="utf-8").strip()
    current = module.digest()
    assert recorded == current, (
        "the published tutorial is stale: the sources have changed since the last "
        "`make publish`. Run `make publish` in docs/tutorials/dafx2026/ and commit "
        f"the result.\n  recorded {recorded[:16]}…\n  current  {current[:16]}…"
    )
