"""Readable preset documents for vanilla feedback delay networks.

A preset keeps the exact, renderable :class:`FDNBuild` separate from optional
metadata and JSON-like design notes. The build is authoritative; design records
only preserve choices that cannot be inferred reliably from its numbers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib.resources import files
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from .build import FDNBuild, fdn_build_from_dict, fdn_build_to_dict
from .eq.design import EQ_DESIGNS
from .generate.fdn_matrix_gallery import (
    FEEDBACK_MATRIX_TYPES,
    IO_MATRIX_TYPES,
)
from .generate.sample_delay_lengths import DELAY_DISTRIBUTIONS


@dataclass
class FDNPreset:
    """A baked vanilla FDN plus metadata and optional design information.

    ``metadata`` is an open JSON object intended for catalog and attribution
    information. When used, ``tags`` should be a list of strings so callers can
    filter presets consistently without restricting other metadata fields.

    ``design`` deliberately uses the same nested dictionaries as the JSON
    document. Unknown choices are represented by leaving out ``type`` or the
    entire component::

        FDNPreset(
            build=build,
            metadata={"name": "small-room", "tags": ["room", "short"]},
            design={
                "delays": {"type": "uniform", "coprime": True},
                "feedback_matrix": {"type": "orthogonal"},
            },
        )
    """

    build: FDNBuild
    metadata: dict[str, Any]
    design: dict[str, dict[str, Any]] = field(default_factory=dict)


_DESIGN_TYPES = {
    "delays": DELAY_DISTRIBUTIONS,
    "feedback_matrix": FEEDBACK_MATRIX_TYPES,
    "input_matrix": IO_MATRIX_TYPES,
    "output_matrix": IO_MATRIX_TYPES,
    "post_delay": EQ_DESIGNS,
    "post_matrix": EQ_DESIGNS,
    "post_output": EQ_DESIGNS,
}

_PRESET_ROOT = files("pyFDN.resources").joinpath("presets", "colorless_FDN")
_PRESET_RE = re.compile(r"colorless_(?P<initial>init_)?N(?P<size>\d+)_d(?P<delay>\d+)")


def _design_from_dict(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    design: dict[str, dict[str, Any]] = {}
    for component, fields in data.items():
        choices = _DESIGN_TYPES.get(component)
        if choices is None:
            raise ValueError(f"unknown preset design component: {component!r}")
        if not isinstance(fields, dict):
            raise ValueError(f"design.{component} must be an object")
        fields = dict(fields)
        design_type = fields.get("type")
        if design_type is not None and design_type not in choices:
            raise ValueError(
                f"design.{component}.type must be one of {', '.join(choices)}"
            )
        design[component] = fields
    return design


def _json_value(value: Any) -> Any:
    """Copy arrays and nested containers into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


def fdn_preset_to_dict(preset: FDNPreset) -> dict[str, Any]:
    """Convert an :class:`FDNPreset` to a JSON-compatible dictionary."""
    return {
        "metadata": _json_value(preset.metadata),
        "design": _json_value(_design_from_dict(preset.design)),
        "build": fdn_build_to_dict(preset.build),
    }


def fdn_preset_from_dict(data: dict[str, Any]) -> FDNPreset:
    """Construct an :class:`FDNPreset` from a parsed JSON dictionary."""
    if "metadata" not in data or "build" not in data:
        raise ValueError("preset needs metadata and build fields")
    metadata = data["metadata"]
    design = data.get("design", {})
    build = data["build"]
    if not isinstance(metadata, dict):
        raise ValueError("preset metadata must be an object")
    if not isinstance(design, dict):
        raise ValueError("preset design must be an object")
    if not isinstance(build, dict):
        raise ValueError("preset build must be an object")
    return FDNPreset(
        build=fdn_build_from_dict(build),
        metadata=dict(metadata),
        design=_design_from_dict(design),
    )


def save_fdn_preset(path: str | PathLike[str], preset: FDNPreset) -> None:
    """Write a preset as indented, human-readable JSON."""
    Path(path).write_text(
        json.dumps(fdn_preset_to_dict(preset), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_fdn_preset_file(path: str | PathLike[str]) -> FDNPreset:
    """Load an :class:`FDNPreset` from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("FDN preset JSON must contain an object")
    return fdn_preset_from_dict(data)


def available_fdn_presets() -> tuple[str, ...]:
    """Return the names accepted by :func:`load_fdn_preset`."""
    names = []
    for resource in _PRESET_ROOT.iterdir():
        match = re.fullmatch(r"colorless_(init_)?N(\d+)_d(\d+)\.json", resource.name)
        if match:
            initial = "init_" if match.group(1) else ""
            names.append(f"colorless_{initial}N{match.group(2)}_d{match.group(3)}")
    return tuple(sorted(names))


def _preset_resource(name: str) -> tuple[str, Any]:
    normalized_name = name.removesuffix(".json")
    resource = _PRESET_ROOT.joinpath(f"{normalized_name}.json")
    if not _PRESET_RE.fullmatch(normalized_name) or not resource.is_file():
        choices = ", ".join(available_fdn_presets())
        raise ValueError(f"Unknown FDN preset '{name}'. Available presets: {choices}")
    return normalized_name, resource


def get_fdn_preset(name: str) -> FDNPreset:
    """Return a packaged :class:`FDNPreset` document.

    Older build-only resources are also accepted. Their metadata is lifted
    into a preset without guessing design choices that were not saved.
    """
    normalized_name, resource = _preset_resource(name)
    data = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Packaged FDN preset JSON must contain an object")
    if "build" in data:
        return fdn_preset_from_dict(data)

    legacy_metadata = data.get("metadata", {})
    if not isinstance(legacy_metadata, dict):
        raise ValueError("Packaged FDN build metadata must contain an object")
    metadata = dict(legacy_metadata)
    metadata.setdefault("name", normalized_name)
    return FDNPreset(build=fdn_build_from_dict(data), metadata=metadata)


def load_fdn_preset(name: str) -> FDNBuild:
    """Load the exact baked build carried by a packaged preset."""
    return get_fdn_preset(name).build


__all__ = [
    "FDNPreset",
    "available_fdn_presets",
    "fdn_preset_from_dict",
    "fdn_preset_to_dict",
    "get_fdn_preset",
    "load_fdn_preset",
    "load_fdn_preset_file",
    "save_fdn_preset",
]
