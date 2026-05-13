"""JSON deserialiser for the FDN graph IR.

Stubbed pending the step-4 rewrite. Will dispatch on the ``"type"``
discriminator (``Shell`` / ``Series`` / ``Parallel`` / ``Recursion`` /
``Node``) and resolve terminal ``"module_type"`` strings via
``pyFDN.config.builders.get_node_class``.
"""
from __future__ import annotations

from typing import Any

from pyFDN.config.builders import Shell


_PENDING = (
    "from_json is awaiting the step-4 rewrite for the new Shell-rooted IR. "
    "See the plan at ~/.claude/plans/can-you-tell-me-cached-bachman.md "
    "for the format."
)


def dict_to_config(data: dict) -> Shell:
    raise NotImplementedError(_PENDING)


def json_to_config(json_str: str) -> Shell:
    raise NotImplementedError(_PENDING)


def json_file_to_config(path: str) -> Shell:
    raise NotImplementedError(_PENDING)
