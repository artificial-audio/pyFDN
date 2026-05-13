"""Validate an FDN graph (rooted at `Shell`) and infer per-edge channel counts.

Walks the graph once. At each `Module` calls `check_shape()`. At each
composite enforces the composition rule:

  * `Series`   — child outputs chain into the next child's inputs.
  * `Parallel` — branches must agree on `N_in`; with `sum_output=True`
                 they must also agree on `N_out`, otherwise outputs
                 concatenate.
  * `Recursion` — `fF` and `fB` must each be square and share the
                 same `N`. `gamma`, if set, has length N.

Returns `IOSpec(N_in, N_out)` for any element. Raises `ValidationError`
on any inconsistency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Set

from pyFDN.config.builders import (
    GraphElement,
    Module,
    Parallel,
    Recursion,
    Series,
    Shell,
    ValidationError,
)


__all__ = ["ValidationError", "IOSpec", "validate", "infer_io"]


@dataclass(frozen=True)
class IOSpec:
    """Channel-count signature: `N_in -> N_out`."""

    N_in: int
    N_out: int


# ──────────────────────────────────────────────────────────────
# Tree walk
# ──────────────────────────────────────────────────────────────


def _visit(element: GraphElement, path: str) -> IOSpec:
    here = f"{path}/{element.name}" if path else element.name

    if isinstance(element, Module):
        try:
            element.check_shape()
        except ValidationError:
            raise
        except ValueError as exc:
            raise ValidationError(f"{here}: {exc}") from exc
        return IOSpec(element.N_in, element.N_out)

    if isinstance(element, Series):
        return _visit_series(element, here)

    if isinstance(element, Parallel):
        return _visit_parallel(element, here)

    if isinstance(element, Recursion):
        return _visit_recursion(element, here)

    if isinstance(element, Shell):
        raise ValidationError(
            f"{here}: nested Shell is not allowed; Shell is root-only"
        )

    raise ValidationError(
        f"{here}: unknown graph element type {type(element).__name__!r}"
    )


def _visit_series(series: Series, here: str) -> IOSpec:
    if not series.children:
        raise ValidationError(f"{here} (Series): must have at least one child")

    _check_unique_names(series.children, here)

    first_spec = _visit(series.children[0], here)
    N_in = first_spec.N_in
    prev_out = first_spec.N_out

    for child in series.children[1:]:
        spec = _visit(child, here)
        if spec.N_in != prev_out:
            raise ValidationError(
                f"{here} (Series): channel mismatch at {child.name!r}: "
                f"previous output={prev_out}, child input={spec.N_in}"
            )
        prev_out = spec.N_out

    return IOSpec(N_in, prev_out)


def _visit_parallel(par: Parallel, here: str) -> IOSpec:
    if not par.children:
        raise ValidationError(f"{here} (Parallel): must have at least one child")

    _check_unique_names(par.children, here)

    specs = [_visit(child, here) for child in par.children]

    N_ins = {s.N_in for s in specs}
    if len(N_ins) != 1:
        raise ValidationError(
            f"{here} (Parallel): branch input channels disagree: {sorted(N_ins)}"
        )
    N_in = next(iter(N_ins))

    if par.sum_output:
        N_outs = {s.N_out for s in specs}
        if len(N_outs) != 1:
            raise ValidationError(
                f"{here} (Parallel, sum_output=True): branch output channels "
                f"disagree: {sorted(N_outs)}"
            )
        N_out = next(iter(N_outs))
    else:
        N_out = sum(s.N_out for s in specs)

    return IOSpec(N_in, N_out)


def _visit_recursion(rec: Recursion, here: str) -> IOSpec:
    fF_spec = _visit(rec.fF, here + "/fF")
    fB_spec = _visit(rec.fB, here + "/fB")

    if fF_spec.N_in != fF_spec.N_out:
        raise ValidationError(
            f"{here} (Recursion): fF must be square (N -> N), got "
            f"{fF_spec.N_in} -> {fF_spec.N_out}"
        )
    if fB_spec.N_in != fB_spec.N_out:
        raise ValidationError(
            f"{here} (Recursion): fB must be square (N -> N), got "
            f"{fB_spec.N_in} -> {fB_spec.N_out}"
        )
    if fF_spec.N_in != fB_spec.N_in:
        raise ValidationError(
            f"{here} (Recursion): fF width {fF_spec.N_in} does not match "
            f"fB width {fB_spec.N_in}"
        )

    n = fF_spec.N_in

    if rec.gamma is not None and rec.gamma.shape != (n,):
        raise ValidationError(
            f"{here} (Recursion): gamma shape {rec.gamma.shape} must be ({n},)"
        )

    return IOSpec(n, n)


def _check_unique_names(children: list, here: str) -> None:
    seen: Set[str] = set()
    for child in children:
        if child.name in seen:
            raise ValidationError(f"{here}: duplicate child name {child.name!r}")
        seen.add(child.name)


# ──────────────────────────────────────────────────────────────
# Public entry points
# ──────────────────────────────────────────────────────────────


def infer_io(element: GraphElement) -> IOSpec:
    """Walk `element` and return its inferred (N_in, N_out)."""
    return _visit(element, "")


def validate(shell: Shell) -> IOSpec:
    """Validate a `Shell` and return its outer (N_in, N_out)."""
    if not isinstance(shell, Shell):
        raise ValidationError(
            f"validate(): expected Shell root, got {type(shell).__name__!r}"
        )

    if not shell.children:
        raise ValidationError(f"Shell {shell.name!r}: must have at least one child")

    _check_unique_names(shell.children, shell.name)

    if len(shell.children) == 1:
        return _visit(shell.children[0], shell.name)

    # Multiple top-level children behave like an implicit Series.
    first = _visit(shell.children[0], shell.name)
    N_in = first.N_in
    prev_out = first.N_out
    for child in shell.children[1:]:
        spec = _visit(child, shell.name)
        if spec.N_in != prev_out:
            raise ValidationError(
                f"Shell {shell.name!r}: top-level Series channel mismatch at "
                f"{child.name!r}: previous output={prev_out}, child input={spec.N_in}"
            )
        prev_out = spec.N_out
    return IOSpec(N_in, prev_out)
