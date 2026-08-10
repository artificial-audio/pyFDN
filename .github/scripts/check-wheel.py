"""Fail CI when the wheel leaks build-time docs or exceeds its resource budget."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

MAX_WHEEL_BYTES = 6_000_000
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
    "pyFDN/resources/presets/colorless_FDN/param_N4_d1.mat",
    "pyFDN/resources/references.bib",
}


def main(wheel_name: str) -> None:
    wheel = Path(wheel_name)
    assert wheel.stat().st_size <= MAX_WHEEL_BYTES, (
        f"wheel is {wheel.stat().st_size:,} bytes; budget is {MAX_WHEEL_BYTES:,}"
    )
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    leaked_docs = sorted(name for name in names if name.startswith("docs/"))
    assert not leaked_docs, f"documentation files leaked into wheel: {leaked_docs[:5]}"
    missing = sorted(REQUIRED_SUFFIXES - names)
    assert not missing, f"packaged resources missing from wheel: {missing}"


if __name__ == "__main__":
    main(sys.argv[1])
