"""Content digest of everything the published tutorial is built from.

The rendered deck is committed under ``docs/_static/tutorials/dafx2026/`` because
the Sphinx build only copies ``docs/_static`` — it never runs Quarto. That makes
one failure mode possible: editing a slide and forgetting ``make publish``, so
the website keeps serving the old deck.

``make publish`` writes this digest next to the rendered output as
``sources.sha256``; ``tests/test_tutorial_dafx2026.py`` recomputes it and fails
when the two disagree. So a stale publish breaks CI instead of shipping quietly.

Usage::

    python docs/tutorials/dafx2026/sources_digest.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Everything Quarto reads to produce the deck. Figures are included by content,
# so regenerating them also invalidates the digest.
INPUT_GLOBS = (
    "_quarto.yml",
    "index.qmd",
    "slides.qmd",
    "theme/*.scss",
    "assets/*",
    "figures/out/*.svg",
)


def input_files() -> list[Path]:
    """Every source file, deduplicated and in a stable order."""
    found: set[Path] = set()
    for pattern in INPUT_GLOBS:
        found.update(path for path in HERE.glob(pattern) if path.is_file())
    return sorted(found)


def digest() -> str:
    """SHA-256 over the source paths and their contents."""
    running = hashlib.sha256()
    for path in input_files():
        running.update(path.relative_to(HERE).as_posix().encode())
        running.update(b"\0")
        running.update(path.read_bytes())
        running.update(b"\0")
    return running.hexdigest()


if __name__ == "__main__":
    print(digest())
