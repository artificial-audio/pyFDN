"""Citation helpers backed by the bibliography distributed with pyFDN."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files


def _split_fields(body: str) -> list[str]:
    fields: list[str] = []
    start = 0
    depth = 0
    quoted = False
    escaped = False
    for index, character in enumerate(body):
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"' and depth == 0:
            quoted = not quoted
        elif not quoted:
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
            elif character == "," and depth == 0:
                fields.append(body[start:index])
                start = index + 1
    fields.append(body[start:])
    return fields


def _unwrap(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and (
        (value[0] == "{" and value[-1] == "}") or (value[0] == '"' and value[-1] == '"')
    ):
        value = value[1:-1]
    return " ".join(value.split())


def _parse_bibtex(text: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    cursor = 0
    while (entry_start := text.find("@", cursor)) >= 0:
        opening = text.find("{", entry_start)
        separator = text.find(",", opening)
        if opening < 0 or separator < 0:
            break

        paper_id = text[opening + 1 : separator].strip()
        depth = 1
        index = separator + 1
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth:
            raise ValueError(f"Unterminated BibTeX entry '{paper_id}'")

        body = text[separator + 1 : index - 1]
        entry: dict[str, str] = {}
        for field in _split_fields(body):
            key, equals, value = field.partition("=")
            if equals:
                entry[key.strip().lower()] = _unwrap(value)
        entries[paper_id] = entry
        cursor = index
    return entries


@lru_cache(maxsize=1)
def _papers() -> dict[str, dict[str, str]]:
    bibliography = files("pyFDN.resources").joinpath("references.bib")
    return _parse_bibtex(bibliography.read_text(encoding="utf-8"))


def paper_reference(paper_id: str) -> dict[str, str]:
    """Return a copy of the bibliography fields for ``paper_id``."""

    try:
        return dict(_papers()[paper_id])
    except KeyError as exc:
        choices = ", ".join(sorted(_papers()))
        raise KeyError(
            f"Unknown paper ID '{paper_id}'. Available IDs: {choices}"
        ) from exc


def paper_link(paper_id: str) -> str:
    """Return a Markdown citation link for a packaged bibliography entry."""

    paper = paper_reference(paper_id)
    text = ", ".join(
        value
        for value in (paper.get("author"), paper.get("title"), paper.get("year"))
        if value
    )
    url = paper.get("url", "#")
    return f"[{text}]({url})"
