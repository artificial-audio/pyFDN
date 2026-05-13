"""Declarative FDN graph IR — dataclasses + builder helpers.

Closed structural composites: `Shell`, `Series`, `Parallel`, `Recursion`.
Open terminal hierarchy rooted at `Module`; subclasses register via
`@register_module`. JSON discriminators: `type` (structural) and
`module_type` (terminal class name).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Type, Union

import numpy as np

from pyFDN.auxiliary.acoustics import one_pole_absorption
from pyFDN.generate.construct_cascaded_paraunitary_matrix import (
    construct_cascaded_paraunitary_matrix,
)
from pyFDN.generate.construct_velvet_feedback_matrix import (
    construct_velvet_feedback_matrix,
)
from pyFDN.generate.random_orthogonal import random_orthogonal


class ValidationError(ValueError):
    """Raised when a graph fails a structural or shape check."""


# ──────────────────────────────────────────────────────────────
# Composites (closed)
# ──────────────────────────────────────────────────────────────


@dataclass
class Shell:
    """Root container — top-level metadata + the graph."""

    name: str
    fs: int = 48000
    children: List["GraphElement"] = field(default_factory=list)
    nfft: Optional[int] = None
    alias_decay_db: Optional[float] = None

    type: ClassVar[str] = "Shell"


@dataclass
class Series:
    """Serial composition — children evaluated left-to-right."""

    name: str
    children: List["GraphElement"] = field(default_factory=list)

    type: ClassVar[str] = "Series"


@dataclass
class Parallel:
    """Parallel composition. `sum_output=True` sums outputs, else concatenates."""

    name: str
    children: List["GraphElement"] = field(default_factory=list)
    sum_output: bool = True

    type: ClassVar[str] = "Parallel"


@dataclass
class Recursion:
    """Feedback loop: `lines = fF(x + fB(lines))`."""

    name: str
    fF: "GraphElement"
    fB: "GraphElement"
    gamma: Optional[np.ndarray] = None

    type: ClassVar[str] = "Recursion"

    def __post_init__(self) -> None:
        if self.gamma is not None:
            self.gamma = np.asarray(self.gamma, dtype=np.float64)


# ──────────────────────────────────────────────────────────────
# Terminal base (open set)
# ──────────────────────────────────────────────────────────────


@dataclass
class Module:
    """Base for DSP terminal modules. Subclass + `@register_module` to extend."""

    name: str
    N_in: int
    N_out: int

    type: ClassVar[str] = "Module"

    @property
    def module_type(self) -> str:
        return type(self).__name__

    def check_shape(self) -> None:
        return None


GraphElement = Union[Shell, Series, Parallel, Recursion, Module]


# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────


_MODULE_REGISTRY: Dict[str, Type[Module]] = {}


def register_module(cls: Type[Module]) -> Type[Module]:
    """Decorator: register a `Module` subclass for JSON lookup by class name."""
    if not issubclass(cls, Module):
        raise TypeError(f"register_module: {cls!r} is not a Module subclass")
    _MODULE_REGISTRY[cls.__name__] = cls
    return cls


def get_module_class(module_type: str) -> Type[Module]:
    """Look up a registered `Module` subclass by name."""
    try:
        return _MODULE_REGISTRY[module_type]
    except KeyError as exc:
        known = ", ".join(sorted(_MODULE_REGISTRY)) or "<none>"
        raise ValidationError(
            f"Unknown terminal module_type {module_type!r}. Known: {known}"
        ) from exc


def registered_modules() -> List[str]:
    """Names of all registered terminal module types."""
    return sorted(_MODULE_REGISTRY)


# ──────────────────────────────────────────────────────────────
# Concrete terminals — Gain / Matrix
# ──────────────────────────────────────────────────────────────


@register_module
@dataclass
class Gain(Module):
    """Linear gain — 2-D `gains`, shape `(N_out, N_in)`."""

    gains: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float64)
    )

    def __post_init__(self) -> None:
        self.gains = np.asarray(self.gains, dtype=np.float64)

    def check_shape(self) -> None:
        if self.gains.ndim != 2:
            raise ValidationError(
                f"Gain {self.name!r}: `gains` must be 2-D, "
                f"got {self.gains.ndim}-D shape {self.gains.shape}"
            )
        n_out, n_in = self.gains.shape
        if self.N_in != n_in or self.N_out != n_out:
            raise ValidationError(
                f"Gain {self.name!r}: `gains` shape ({n_out}, {n_in}) "
                f"must match N_in={self.N_in}, N_out={self.N_out}"
            )


MATRIX_TYPES = frozenset(
    {"Dense", "Diagonal", "Orthogonal", "Householder", "Hadamard"}
)


@register_module
@dataclass(init=False)
class Matrix(Gain):
    """Square (N x N) `Gain` with a `matrix_type` discriminator.

    Fields:
      * ``matrix_type`` ∈ ``MATRIX_TYPES``
      * ``diag``        — length-N vector when ``matrix_type == "Diagonal"``
      * ``unit_vector`` — length-N unit vector when ``matrix_type == "Householder"``

    Dense N×N data lives in inherited ``gains``. For ``"Diagonal"`` and
    ``"Householder"``, ``gains`` is auto-materialised from the generator
    if not supplied.

    Constructor resolves N from (in priority): ``N=``, ``N_in==N_out``
    pair, ``gains.shape``, ``len(diag)``, ``len(unit_vector)``.
    """

    matrix_type: str = "Dense"
    diag: Optional[np.ndarray] = None
    unit_vector: Optional[np.ndarray] = None

    def __init__(
        self,
        *,
        name: str,
        N: Optional[int] = None,
        N_in: Optional[int] = None,
        N_out: Optional[int] = None,
        gains: Optional[np.ndarray] = None,
        matrix_type: str = "Dense",
        diag: Optional[np.ndarray] = None,
        unit_vector: Optional[np.ndarray] = None,
    ) -> None:
        gains_arr = None if gains is None else np.asarray(gains, dtype=np.float64)
        diag_arr = None if diag is None else np.asarray(diag, dtype=np.float64)
        unit_vector_arr = (
            None if unit_vector is None else np.asarray(unit_vector, dtype=np.float64)
        )

        if N is not None:
            if N_in is not None or N_out is not None:
                raise ValidationError(
                    f"Matrix {name!r}: pass `N` OR `N_in`+`N_out`, not both"
                )
            resolved_n = int(N)
        elif N_in is not None and N_out is not None:
            if N_in != N_out:
                raise ValidationError(
                    f"Matrix {name!r}: must be square, N_in={N_in} != N_out={N_out}"
                )
            resolved_n = int(N_in)
        elif N_in is not None or N_out is not None:
            raise ValidationError(
                f"Matrix {name!r}: supply both `N_in` and `N_out` (or use `N=`)"
            )
        elif gains_arr is not None:
            if gains_arr.ndim != 2 or gains_arr.shape[0] != gains_arr.shape[1]:
                raise ValidationError(
                    f"Matrix {name!r}: cannot infer N from non-square gains "
                    f"of shape {gains_arr.shape}"
                )
            resolved_n = int(gains_arr.shape[0])
        elif diag_arr is not None:
            resolved_n = int(diag_arr.shape[0])
        elif unit_vector_arr is not None:
            resolved_n = int(unit_vector_arr.shape[0])
        else:
            raise ValidationError(
                f"Matrix {name!r}: cannot determine N — pass `N=` or supply "
                f"`gains` / `diag` / `unit_vector`"
            )

        if gains_arr is None:
            if matrix_type == "Diagonal" and diag_arr is not None:
                gains_arr = np.diag(diag_arr)
            elif matrix_type == "Householder" and unit_vector_arr is not None:
                gains_arr = np.eye(resolved_n) - 2.0 * np.outer(
                    unit_vector_arr, unit_vector_arr
                )

        self.name = name
        self.N_in = resolved_n
        self.N_out = resolved_n
        self.gains = (
            gains_arr
            if gains_arr is not None
            else np.zeros((resolved_n, resolved_n), dtype=np.float64)
        )
        self.matrix_type = matrix_type
        self.diag = diag_arr
        self.unit_vector = unit_vector_arr

    def check_shape(self) -> None:
        if self.gains.ndim != 2:
            raise ValidationError(
                f"Matrix {self.name!r}: `gains` must be 2-D N×N, "
                f"got {self.gains.ndim}-D shape {self.gains.shape}"
            )
        n_out, n_in = self.gains.shape
        if n_in != n_out:
            raise ValidationError(
                f"Matrix {self.name!r}: must be square, got shape ({n_out}, {n_in})"
            )
        n = n_in
        if self.N_in != n or self.N_out != n:
            raise ValidationError(
                f"Matrix {self.name!r}: `gains` shape ({n}, {n}) must match "
                f"N_in={self.N_in}, N_out={self.N_out}"
            )

        if self.matrix_type not in MATRIX_TYPES:
            raise ValidationError(
                f"Matrix {self.name!r}: matrix_type {self.matrix_type!r} not in "
                f"{sorted(MATRIX_TYPES)}"
            )

        if self.matrix_type == "Diagonal":
            if self.diag is None:
                raise ValidationError(
                    f"Matrix {self.name!r}: matrix_type='Diagonal' requires `diag`"
                )
            if self.diag.ndim != 1 or self.diag.shape[0] != n:
                raise ValidationError(
                    f"Matrix {self.name!r}: `diag` shape {self.diag.shape} must be ({n},)"
                )
        elif self.matrix_type == "Householder":
            if self.unit_vector is None:
                raise ValidationError(
                    f"Matrix {self.name!r}: matrix_type='Householder' requires `unit_vector`"
                )
            if self.unit_vector.ndim != 1 or self.unit_vector.shape[0] != n:
                raise ValidationError(
                    f"Matrix {self.name!r}: `unit_vector` shape {self.unit_vector.shape} "
                    f"must be ({n},)"
                )


# ──────────────────────────────────────────────────────────────
# Concrete terminals — Delay
# ──────────────────────────────────────────────────────────────


@register_module
@dataclass
class Delay(Module):
    """Per-channel integer delay line. `samples` shape `(N,)`."""

    samples: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=np.int64)
        if self.samples.ndim != 1:
            raise ValidationError(
                f"Delay {self.name!r}: `samples` must be 1-D, got shape {self.samples.shape}"
            )
        if np.any(self.samples < 1):
            raise ValidationError(
                f"Delay {self.name!r}: all `samples` must be >= 1"
            )

    def check_shape(self) -> None:
        n = int(self.samples.shape[0])
        if self.N_in != n or self.N_out != n:
            raise ValidationError(
                f"Delay {self.name!r}: `samples` length {n} must match "
                f"N_in={self.N_in}, N_out={self.N_out}"
            )


# ──────────────────────────────────────────────────────────────
# Concrete terminals — Filter and its subclasses
# ──────────────────────────────────────────────────────────────


@register_module
@dataclass
class Filter(Module):
    """Per-channel cascaded SOS filter.

    `sos` shape `(sections, N, 5)` with last axis `[b0, b1, b2, a1, a2]`
    (a0 normalised to 1).
    """

    sos: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0, 5), dtype=np.float64)
    )

    def __post_init__(self) -> None:
        self.sos = np.asarray(self.sos, dtype=np.float64)
        if self.sos.ndim != 3 or self.sos.shape[-1] != 5:
            raise ValidationError(
                f"Filter {self.name!r}: `sos` must have shape (sections, N, 5), "
                f"got {self.sos.shape}"
            )

    def check_shape(self) -> None:
        n = int(self.sos.shape[1])
        if self.N_in != n or self.N_out != n:
            raise ValidationError(
                f"Filter {self.name!r}: channel count {n} must match "
                f"N_in={self.N_in}, N_out={self.N_out}"
            )


@register_module
@dataclass
class Biquad(Filter):
    """Single-section biquad. Placeholder — inherits `Filter`."""


@register_module
@dataclass
class OnePole(Filter):
    """Per-channel one-pole filter. Placeholder — inherits `Filter`."""


@register_module
@dataclass
class GEQ(Filter):
    """Per-channel graphic EQ. Placeholder — inherits `Filter`."""


# ──────────────────────────────────────────────────────────────
# Builders — matrix
# ──────────────────────────────────────────────────────────────


def hadamard_matrix(N: int, *, name: str = "hadamard") -> Matrix:
    """Normalised Hadamard matrix (N must be a power of two)."""
    from scipy.linalg import hadamard

    H = hadamard(N).astype(np.float64) / math.sqrt(N)
    return Matrix(name=name, N=N, gains=H, matrix_type="Hadamard")


def householder_matrix(N: int, *, name: str = "householder") -> Matrix:
    """Random Householder reflection. `gains` materialised from `unit_vector`."""
    v = np.random.standard_normal(N)
    v /= np.linalg.norm(v)
    return Matrix(name=name, N=N, matrix_type="Householder", unit_vector=v)


def random_orthogonal_matrix(N: int, *, name: str = "orthogonal") -> Matrix:
    """Haar-distributed random orthogonal matrix."""
    return Matrix(
        name=name, N=N, gains=random_orthogonal(N), matrix_type="Orthogonal"
    )


def diagonal_matrix(diag: np.ndarray, *, name: str = "diagonal") -> Matrix:
    """Diagonal matrix from a 1-D `diag` vector. `gains = np.diag(diag)`."""
    diag = np.asarray(diag, dtype=np.float64)
    if diag.ndim != 1:
        raise ValidationError(
            f"diagonal_matrix: `diag` must be 1-D, got shape {diag.shape}"
        )
    return Matrix(name=name, matrix_type="Diagonal", diag=diag)


def velvet_feedback_matrix(
    N: int, stages: int, sparsity: float, *, name: str = "velvet"
) -> Matrix:
    """Cascaded-paraunitary (VELVET-style) feedback matrix (DC tap)."""
    forward, _ = construct_velvet_feedback_matrix(N, stages, sparsity)
    A = forward[:, :, 0].astype(np.float64)
    return Matrix(name=name, N=N, gains=A, matrix_type="Dense")


def scattering_matrix(
    N: int, sparsity: float = 1.0, *, name: str = "scattering"
) -> Matrix:
    """Scattering / paraunitary matrix (k=1 cascaded construction)."""
    forward, _ = construct_cascaded_paraunitary_matrix(
        N, k=1, sparsity=sparsity, matrix_type="random"
    )
    A = forward[:, :, 0].astype(np.float64)
    return Matrix(name=name, N=N, gains=A, matrix_type="Dense")


# ──────────────────────────────────────────────────────────────
# Builders — filter
# ──────────────────────────────────────────────────────────────


def one_pole_absorption_filter(
    rt_dc: float,
    rt_ny: float,
    delays: np.ndarray,
    fs: float,
    *,
    name: str = "absorption",
) -> OnePole:
    """One-pole absorption filters from RT targets, packed as SOS `(1, N, 5)`."""
    sos_6n = one_pole_absorption(rt_dc, rt_ny, delays, fs)
    if sos_6n.shape[0] != 6:
        raise ValidationError(
            f"one_pole_absorption returned shape {sos_6n.shape}, expected (6, N)"
        )
    n = int(sos_6n.shape[1])
    a0 = sos_6n[3, :]
    if np.any(a0 == 0):
        raise ValidationError(
            "one_pole_absorption: encountered a0=0, cannot normalise"
        )
    norm = np.stack(
        [
            sos_6n[0] / a0,
            sos_6n[1] / a0,
            sos_6n[2] / a0,
            sos_6n[4] / a0,
            sos_6n[5] / a0,
        ],
        axis=0,
    )
    sos = norm.T[np.newaxis, :, :]
    return OnePole(name=name, N_in=n, N_out=n, sos=sos)


# ──────────────────────────────────────────────────────────────
# Builders — delay
# ──────────────────────────────────────────────────────────────


def random_delays(
    N: int, min_len: int = 400, max_len: int = 1200, *, name: str = "delays"
) -> Delay:
    """Uniformly random delay lengths in `[min_len, max_len)`."""
    samples = np.random.randint(min_len, max_len, size=N).astype(np.int64)
    return Delay(name=name, N_in=N, N_out=N, samples=samples)


def prime_delays(
    N: int, base_length: int = 401, *, name: str = "delays"
) -> Delay:
    """Mutually-incrementing delay scheme: `base + 7*i`."""
    samples = np.array([base_length + i * 7 for i in range(N)], dtype=np.int64)
    return Delay(name=name, N_in=N, N_out=N, samples=samples)


# ──────────────────────────────────────────────────────────────
# Builders — I/O taps (rectangular `Gain`)
# ──────────────────────────────────────────────────────────────


def unity_input_tap(
    N_lines: int, N_in: int = 1, *, name: str = "input_tap"
) -> Gain:
    """Equal-weight input distribution `N_in -> N_lines`."""
    B = np.ones((N_lines, N_in), dtype=np.float64) / math.sqrt(N_lines)
    return Gain(name=name, N_in=N_in, N_out=N_lines, gains=B)


def unity_output_tap(
    N_lines: int, N_out: int = 1, *, name: str = "output_tap"
) -> Gain:
    """Equal-weight output mix `N_lines -> N_out`."""
    C = np.ones((N_out, N_lines), dtype=np.float64) / math.sqrt(N_lines)
    return Gain(name=name, N_in=N_lines, N_out=N_out, gains=C)


# ──────────────────────────────────────────────────────────────
# High-level builder
# ──────────────────────────────────────────────────────────────


def build_vanilla_config(
    N: int = 64,
    *,
    rt_dc: float = 2.0,
    rt_ny: float = 0.5,
    fs: float = 48000.0,
    delay_min: int = 400,
    delay_max: int = 1200,
    N_in: int = 1,
    N_out: int = 1,
    name: str = "vanilla_fdn",
) -> Shell:
    """Vanilla FDN `Shell` mirroring `generate.vanilla_FDN`."""
    delays = random_delays(N, delay_min, delay_max, name="delays")
    absorption = one_pole_absorption_filter(
        rt_dc, rt_ny, delays.samples, fs, name="absorption"
    )
    feedback = random_orthogonal_matrix(N, name="feedback")
    input_tap = unity_input_tap(N, N_in, name="input_tap")
    output_tap = unity_output_tap(N, N_out, name="output_tap")

    return Shell(
        name=name,
        fs=fs,
        children=[
            Series(
                name="full_chain",
                children=[
                    input_tap,
                    Recursion(
                        name="loop",
                        fF=Series(name="fF", children=[delays, absorption]),
                        fB=feedback,
                    ),
                    output_tap,
                ],
            )
        ],
    )
