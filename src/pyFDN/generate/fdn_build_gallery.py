"""Generate complete vanilla FDN builds and optionally record their design."""

from __future__ import annotations

from typing import Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike

from ..build import FDNBuild
from .fdn_matrix_gallery import IO_MATRIX_TYPES
from .sample_delay_lengths import DelayDistribution, sample_delay_lengths

FDNDesign = dict[str, dict[str, Any]]


def _build_rng(
    rng: np.random.Generator | int | None, default_seed: int
) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(default_seed if rng is None else rng)


def _random_orthogonal(N: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((N, N)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def _build_io_matrices(
    N: int,
    num_inputs: int,
    num_outputs: int,
    io_type: str,
    input_scale: float,
    output_scale: float,
    direct_gain: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if min(num_inputs, num_outputs) < 1:
        raise ValueError("num_inputs and num_outputs must be positive")

    io_type = "normalised" if io_type == "normalized" else io_type
    if io_type == "ones":
        B = np.ones((N, num_inputs))
        C = np.ones((num_outputs, N))
    elif io_type == "normalised":
        B = np.ones((N, num_inputs)) / np.sqrt(N)
        C = np.ones((num_outputs, N)) / np.sqrt(N)
    elif io_type == "identity":
        B = np.eye(N, num_inputs)
        C = np.eye(num_outputs, N)
    elif io_type == "random":
        B = rng.standard_normal((N, num_inputs))
        C = rng.standard_normal((num_outputs, N))
    else:
        raise ValueError(
            "io_type must be one of 'ones', 'normalised', 'identity', or 'random'"
        )

    B = input_scale * B
    C = output_scale * C
    if direct_gain is None:
        D = rng.standard_normal((num_outputs, num_inputs))
    else:
        D = np.full((num_outputs, num_inputs), direct_gain, dtype=float)
    return B, C, D, io_type


def _build_post_output(
    num_outputs: int,
    fs: float,
    gain_db: ArrayLike | None,
    gain_db_nyquist: ArrayLike | None,
    crossover: float | None,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Design a per-output first-order shelf and its provenance."""
    if gain_db is None and gain_db_nyquist is None:
        if crossover is not None:
            raise ValueError(
                "output_gain_db or output_gain_db_nyquist must be set to "
                "configure the post_output hook"
            )
        return None, None
    if gain_db is None:
        gain_db = gain_db_nyquist
    if gain_db_nyquist is None:
        gain_db_nyquist = gain_db
    assert gain_db is not None and gain_db_nyquist is not None

    def _per_output(value: ArrayLike, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=float).ravel()
        if array.size == 1:
            array = np.full(num_outputs, array.item())
        if array.size != num_outputs:
            raise ValueError(f"{name} must be scalar or length num_outputs")
        return array

    from ..eq import gain_to_first_order_shelf

    return gain_to_first_order_shelf(
        _per_output(gain_db, "output_gain_db"),
        _per_output(gain_db_nyquist, "output_gain_db_nyquist"),
        crossover,
        fs,
        return_design=True,
    )


@overload
def fdn_build_gallery(
    N: int | None = None,
    *,
    fs: float = 48_000.0,
    delays: np.ndarray | None = None,
    delay_range: tuple[int, int] = (400, 1200),
    delay_distribution: DelayDistribution = "uniform",
    coprime: bool = False,
    sort_delays: bool = False,
    num_inputs: int = 1,
    num_outputs: int = 1,
    io_type: str = "normalised",
    input_scale: float = 1.0,
    output_scale: float = 1.0,
    direct_gain: float | None = 0.0,
    rt: float | None = 2.0,
    rt_nyquist: float | None = None,
    rt_crossover: float | None = None,
    output_gain_db: ArrayLike | None = None,
    output_gain_db_nyquist: ArrayLike | None = None,
    output_crossover: float | None = None,
    rng: np.random.Generator | int | None = None,
    return_design: Literal[False] = False,
) -> FDNBuild: ...


@overload
def fdn_build_gallery(
    N: int | None = None,
    *,
    fs: float = 48_000.0,
    delays: np.ndarray | None = None,
    delay_range: tuple[int, int] = (400, 1200),
    delay_distribution: DelayDistribution = "uniform",
    coprime: bool = False,
    sort_delays: bool = False,
    num_inputs: int = 1,
    num_outputs: int = 1,
    io_type: str = "normalised",
    input_scale: float = 1.0,
    output_scale: float = 1.0,
    direct_gain: float | None = 0.0,
    rt: float | None = 2.0,
    rt_nyquist: float | None = None,
    rt_crossover: float | None = None,
    output_gain_db: ArrayLike | None = None,
    output_gain_db_nyquist: ArrayLike | None = None,
    output_crossover: float | None = None,
    rng: np.random.Generator | int | None = None,
    return_design: Literal[True],
) -> tuple[FDNBuild, FDNDesign]: ...


def fdn_build_gallery(
    N: int | None = None,
    *,
    fs: float = 48_000.0,
    delays: np.ndarray | None = None,
    delay_range: tuple[int, int] = (400, 1200),
    delay_distribution: DelayDistribution = "uniform",
    coprime: bool = False,
    sort_delays: bool = False,
    num_inputs: int = 1,
    num_outputs: int = 1,
    io_type: str = "normalised",
    input_scale: float = 1.0,
    output_scale: float = 1.0,
    direct_gain: float | None = 0.0,
    rt: float | None = 2.0,
    rt_nyquist: float | None = None,
    rt_crossover: float | None = None,
    output_gain_db: ArrayLike | None = None,
    output_gain_db_nyquist: ArrayLike | None = None,
    output_crossover: float | None = None,
    rng: np.random.Generator | int | None = None,
    return_design: bool = False,
) -> FDNBuild | tuple[FDNBuild, FDNDesign]:
    """Build a complete FDN and optionally return its non-inferable design.

    The feedback matrix is random orthogonal. Sampled delays support the same
    distributions and coprimality option as :func:`sample_delay_lengths`.
    Explicit delays remain purely numerical and therefore produce no delay
    design record.

    In-loop attenuation and optional output EQ are first-order shelves. Set
    ``return_design=True`` to return ``(build, design)`` for direct use in an
    :class:`pyFDN.FDNPreset`; the default remains the plain :class:`FDNBuild`.
    """
    if fs <= 0:
        raise ValueError("fs must be positive")
    if rt is not None and rt <= 0:
        raise ValueError("rt must be positive")
    if rt_nyquist is not None and rt_nyquist <= 0:
        raise ValueError("rt_nyquist must be positive")

    local_rng = _build_rng(rng, 0)
    delay_design: dict[str, Any] | None = None
    if delays is not None:
        if delay_distribution != "uniform" or coprime:
            raise ValueError(
                "delay_distribution and coprime apply only when delays are sampled"
            )
        delays_array = np.asarray(delays, dtype=int).ravel()
        if N is None:
            N = delays_array.size
        elif delays_array.size != N:
            raise ValueError("delays must contain exactly N values")
        if sort_delays:
            delays_array = np.sort(delays_array)
    else:
        if N is None:
            raise ValueError("N must be provided when delays is omitted")
        delays_array = sample_delay_lengths(
            N,
            delay_range,
            distribution=delay_distribution,
            coprime=coprime,
            sort=sort_delays,
            rng=local_rng,
        )
        delay_design = {
            "type": delay_distribution,
            "range": list(delay_range),
            "coprime": coprime,
            "sort": sort_delays,
        }

    if N < 1:
        raise ValueError("N must be positive")
    if np.any(delays_array < 1):
        raise ValueError("all delays must be positive")

    A = _random_orthogonal(N, local_rng)
    B, C, D, normalized_io_type = _build_io_matrices(
        N,
        num_inputs,
        num_outputs,
        io_type,
        input_scale,
        output_scale,
        direct_gain,
        local_rng,
    )

    post_delay: np.ndarray | None = None
    attenuation_design: dict[str, Any] | None = None
    if rt is not None:
        from ..eq import decay_to_first_order_shelf

        rt_nyquist = rt if rt_nyquist is None else rt_nyquist
        post_delay, attenuation_design = decay_to_first_order_shelf(
            rt,
            rt_nyquist,
            rt_crossover,
            delays_array,
            float(fs),
            return_design=True,
        )

    post_output, output_design = _build_post_output(
        num_outputs,
        float(fs),
        output_gain_db,
        output_gain_db_nyquist,
        output_crossover,
    )
    build = FDNBuild(
        A,
        B,
        C,
        D,
        delays_array,
        float(fs),
        post_delay=post_delay,
        post_output=post_output,
    )
    if not return_design:
        return build

    design: FDNDesign = {"feedback_matrix": {"type": "orthogonal"}}
    if delay_design is not None:
        design["delays"] = delay_design
    if normalized_io_type in IO_MATRIX_TYPES:
        design["input_matrix"] = {"type": normalized_io_type}
        design["output_matrix"] = {"type": normalized_io_type}
    if attenuation_design is not None:
        design["post_delay"] = attenuation_design
    if output_design is not None:
        design["post_output"] = output_design
    return build, design


__all__ = ["FDNDesign", "fdn_build_gallery"]
