"""JSON serialiser for the FDN graph IR.

Stubbed pending the step-4 rewrite. The new format is type-tagged and
recursive: every graph element serialises as ``{"type": <kind>,
"name": ..., ...}``; terminal `Node` subclasses additionally carry
``"module_type"`` (derived from the Python class name) and a
``"params"`` block with their typed fields.

See the canonical target shape in
``~/.claude/plans/can-you-give-me-elegant-popcorn.md`` and the new IR
in ``pyFDN.config.builders``.
"""
from __future__ import annotations

from typing import Any

from pyFDN.config.builders import Shell


_PENDING = (
    "to_json is awaiting the step-4 rewrite for the new Shell-rooted IR. "
    "See the plan at ~/.claude/plans/can-you-tell-me-cached-bachman.md "
    "for the format."
)


def config_to_dict(config: Shell) -> dict:
    raise NotImplementedError(_PENDING)


def config_to_json(config: Shell, indent: int = 2) -> str:
    raise NotImplementedError(_PENDING)


def config_to_json_file(config: Shell, path: str) -> None:
    raise NotImplementedError(_PENDING)
