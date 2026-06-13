"""Smoke-test that every example script runs to completion."""

import runpy
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.io as pio
import pytest
from pytest import MonkeyPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"


def _example_scripts() -> list[Path]:
    return sorted(EXAMPLES_DIR.rglob("example_*.py"))


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _headless_rendering(monkeypatch: MonkeyPatch) -> Iterator[None]:
    """Neutralise display side-effects so examples run headless.

    Only *rendering* is suppressed — the underlying objects are still built, so
    the example logic is exercised:
      - matplotlib draws to the Agg buffer (no window);
      - plotly's ``fig.show()`` becomes a no-op, so plotly never reaches the
        notebook mime stack (which would otherwise need IPython + nbformat).
    """
    plt.switch_backend("Agg")
    monkeypatch.setattr(pio, "show", lambda *args, **kwargs: None)
    yield
    plt.close("all")


def test_examples_dir_is_populated() -> None:
    assert _example_scripts(), f"No example scripts found under {EXAMPLES_DIR}"


@pytest.mark.parametrize(  # type: ignore[misc]
    "script",
    _example_scripts(),
    ids=lambda p: str(p.relative_to(EXAMPLES_DIR)),
)
def test_example_runs(script: Path) -> None:
    runpy.run_path(str(script), run_name="__main__")
