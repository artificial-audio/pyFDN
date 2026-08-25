"""Readable, versioned serialization for :class:`pyFDN.FDNBuild`."""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from .generate.fdn_matrix_gallery import FDNBuild

FDN_BUILD_FORMAT = "pyfdn-fdn-build"
#: Current schema version. The three filter hooks are spelled ``post_delay``,
#: ``post_matrix`` and ``post_output``, the names they carry everywhere else in
#: pyFDN.
FDN_BUILD_VERSION = 2


def _optional_array(value: Any, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def fdn_build_to_dict(
    build: FDNBuild, *, metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Convert an :class:`FDNBuild` to the versioned JSON-compatible format."""

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
    # Validate shape, values, and schema before exposing or writing the data.
    fdn_build_from_dict(data)
    return data


def fdn_build_from_dict(
    data: Mapping[str, Any], *, fs: float | None = None
) -> FDNBuild:
    """Construct an :class:`FDNBuild` from its versioned dictionary format.

    Args:
        data: Parsed JSON-compatible build dictionary.
        fs: Optional sample-rate override in Hz.
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
    # The two in-loop hooks run on the delay lines; the output hook on the
    # output channels.
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
    metadata: Mapping[str, Any] | None = None,
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
    """Load an :class:`FDNBuild` from the versioned JSON format."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("FDN build JSON must contain an object")
    return fdn_build_from_dict(data, fs=fs)
