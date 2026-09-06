"""Baked feedback-delay-network builds and their JSON representation.

An :class:`FDNBuild` is a delay state-space (DSS) system -- ``A``, ``B``, ``C``,
``D``, and ``delays`` -- plus the sample rate and three optional baked filter
hooks. A build is designed around a DSS system, not the other way around: use
the raw DSS arrays (and ``dss_to_*`` functions) for pure state-space math, and
an :class:`FDNBuild` (and ``build_to_*`` functions) once ``fs`` or the filter
hooks matter, e.g. for rendering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

FDN_BUILD_FORMAT = "pyfdn-fdn-build"
FDN_BUILD_VERSION = 2


@dataclass(frozen=True)
class FDNBuild:
    """Complete, renderable parameters of a vanilla FDN.

    ``A``, ``B``, ``C``, ``D``, and ``delays`` are a delay state-space (DSS)
    system; ``fs`` and the three optional filter hooks turn that system into a
    complete, renderable build. Every field is a plain NumPy value consumed by
    :func:`pyFDN.process_fdn` and :func:`pyFDN.build_to_impz`. A build does not
    remember how its numbers were designed; that optional information belongs
    to :class:`pyFDN.FDNPreset`.

    The three optional SOS banks correspond directly to pyFDN's filter hooks:

    * ``post_delay`` has shape ``(sections, 6, N)`` and sets the decay inside
      the loop after the delays.
    * ``post_matrix`` has shape ``(sections, 6, N)`` and filters the feedback
      path after the feedback matrix.
    * ``post_output`` has shape ``(sections, 6, outputs)`` and filters the wet
      signal outside the recursion.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    delays: np.ndarray
    fs: float
    post_delay: np.ndarray | None = None
    post_matrix: np.ndarray | None = None
    post_output: np.ndarray | None = None


def _optional_array(value: Any, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def fdn_build_to_dict(
    build: FDNBuild, *, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Convert an :class:`FDNBuild` to its JSON-compatible format."""
    data: dict[str, Any] = {
        "format": FDN_BUILD_FORMAT,
        "version": FDN_BUILD_VERSION,
        "feedback_matrix": np.asarray(build.A).tolist(),
        "input_matrix": np.asarray(build.B).tolist(),
        "output_matrix": np.asarray(build.C).tolist(),
        "direct_matrix": np.asarray(build.D).tolist(),
        "delays": np.asarray(build.delays).tolist(),
        "sample_rate": float(build.fs),
        "post_delay": (
            None if build.post_delay is None else np.asarray(build.post_delay).tolist()
        ),
        "post_matrix": (
            None
            if build.post_matrix is None
            else np.asarray(build.post_matrix).tolist()
        ),
        "post_output": (
            None
            if build.post_output is None
            else np.asarray(build.post_output).tolist()
        ),
    }
    if metadata:
        data["metadata"] = dict(metadata)
    fdn_build_from_dict(data)
    return data


def fdn_build_from_dict(data: dict[str, Any], *, fs: float | None = None) -> FDNBuild:
    """Construct an :class:`FDNBuild` from its JSON-compatible dictionary.

    ``fs`` is an optional override retained for loading standalone legacy build
    files. Preset loading always uses the sample rate stored in the build.
    """
    if data.get("format") != FDN_BUILD_FORMAT:
        raise ValueError(f"Expected format '{FDN_BUILD_FORMAT}'")
    version = data.get("version")
    if version != FDN_BUILD_VERSION:
        raise ValueError(f"Unsupported FDN build version: {version!r}")

    required = (
        "feedback_matrix",
        "input_matrix",
        "output_matrix",
        "direct_matrix",
        "delays",
        "sample_rate",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"FDN build is missing required fields: {', '.join(missing)}")

    A = _optional_array(data["feedback_matrix"], "feedback_matrix")
    B = _optional_array(data["input_matrix"], "input_matrix")
    C = _optional_array(data["output_matrix"], "output_matrix")
    D = _optional_array(data["direct_matrix"], "direct_matrix")
    assert A is not None and B is not None and C is not None and D is not None

    delays_float = _optional_array(data["delays"], "delays")
    assert delays_float is not None
    delays_float = delays_float.ravel()
    if not np.array_equal(delays_float, np.rint(delays_float)):
        raise ValueError("delays must contain integers")
    delays = delays_float.astype(np.int64)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("feedback_matrix must be square")
    num_delays = A.shape[0]
    if B.ndim != 2 or B.shape[0] != num_delays:
        raise ValueError("input_matrix must have one row per delay line")
    if C.ndim != 2 or C.shape[1] != num_delays:
        raise ValueError("output_matrix must have one column per delay line")
    if D.shape != (C.shape[0], B.shape[1]):
        raise ValueError("direct_matrix shape must match the output and input counts")
    if delays.size != num_delays or np.any(delays <= 0):
        raise ValueError("delays must contain one positive integer per delay line")

    sample_rate = float(data["sample_rate"] if fs is None else fs)
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be positive and finite")

    post_delay = _optional_array(data.get("post_delay"), "post_delay")
    post_matrix = _optional_array(data.get("post_matrix"), "post_matrix")
    post_output = _optional_array(data.get("post_output"), "post_output")
    for name, hook in (("post_delay", post_delay), ("post_matrix", post_matrix)):
        if hook is not None and (
            hook.ndim != 3 or hook.shape[1] != 6 or hook.shape[2] != num_delays
        ):
            raise ValueError(f"{name} must have shape (sections, 6, delay lines)")
    if post_output is not None and (
        post_output.ndim != 3
        or post_output.shape[1] != 6
        or post_output.shape[2] != C.shape[0]
    ):
        raise ValueError("post_output must have shape (sections, 6, outputs)")

    return FDNBuild(
        A,
        B,
        C,
        D,
        delays,
        sample_rate,
        post_delay=post_delay,
        post_matrix=post_matrix,
        post_output=post_output,
    )


def save_fdn_build(
    path: str | PathLike[str],
    build: FDNBuild,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write an :class:`FDNBuild` as indented, human-readable JSON."""
    Path(path).write_text(
        json.dumps(
            fdn_build_to_dict(build, metadata=metadata), indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def load_fdn_build(path: str | PathLike[str], *, fs: float | None = None) -> FDNBuild:
    """Load an :class:`FDNBuild` from its JSON format."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("FDN build JSON must contain an object")
    return fdn_build_from_dict(data, fs=fs)


__all__ = [
    "FDNBuild",
    "FDN_BUILD_FORMAT",
    "FDN_BUILD_VERSION",
    "fdn_build_from_dict",
    "fdn_build_to_dict",
    "load_fdn_build",
    "save_fdn_build",
]
