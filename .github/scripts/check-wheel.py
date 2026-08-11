"""Fail CI when the wheel leaks build-time docs or exceeds its resource budget."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

MAX_WHEEL_BYTES = 6_000_000
COLORLESS_PRESET_ROOT = "pyFDN/resources/presets/colorless_FDN"
COLORLESS_PRESETS = {
    f"{COLORLESS_PRESET_ROOT}/colorless_{initial}N{size}_d{delay}.json"
    for initial in ("", "init_")
    for size in (4, 6, 8, 16)
    for delay in (1, 2)
}
REQUIRED_SUFFIXES = {
    "pyFDN/resources/audio/metadata.json",
    "pyFDN/resources/audio/drums/drums.wav",
    "pyFDN/resources/audio/general/synth_dry.wav",
    "pyFDN/resources/audio/rir/s3_r4_o.wav",
    "pyFDN/resources/audio/speech/P1_Set1_8.wav",
    "pyFDN/resources/audio/speech/P20_Set1_8.wav",
    "pyFDN/resources/audio/strings/Kleiderschrank_Taylor_DI.wav",
    "pyFDN/resources/audio/strings/Sommerhit2016_Taylor_DI.wav",
    "pyFDN/resources/licenses/diff-fdn-colorless-MIT.txt",
    "pyFDN/resources/references.bib",
} | COLORLESS_PRESETS


def main(wheel_name: str) -> None:
    wheel = Path(wheel_name)
    assert wheel.stat().st_size <= MAX_WHEEL_BYTES, (
        f"wheel is {wheel.stat().st_size:,} bytes; budget is {MAX_WHEEL_BYTES:,}"
    )
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    leaked_docs = sorted(name for name in names if name.startswith("docs/"))
    assert not leaked_docs, f"documentation files leaked into wheel: {leaked_docs[:5]}"
    obsolete_mat_presets = sorted(
        name
        for name in names
        if name.startswith(f"{COLORLESS_PRESET_ROOT}/") and name.endswith(".mat")
    )
    assert not obsolete_mat_presets, (
        f"obsolete colorless-FDN MAT presets leaked into wheel: {obsolete_mat_presets}"
    )
    missing = sorted(REQUIRED_SUFFIXES - names)
    assert not missing, f"packaged resources missing from wheel: {missing}"


if __name__ == "__main__":
    main(sys.argv[1])
