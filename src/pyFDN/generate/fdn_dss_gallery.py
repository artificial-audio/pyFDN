"""Generate a random vanilla FDN as a DSS system plus its baked filters."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..auxiliary.utils import as_generator
from ..dss import DSS
from .orthogonal import random_orthogonal
from .sample_delay_lengths import DelayDistribution, sample_delay_lengths


def _build_io_matrices(
    N: int,
    num_inputs: int,
    num_outputs: int,
    io_type: str,
    input_scale: float,
    output_scale: float,
    direct_gain: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return B, C, D


def _build_post_output(
    num_outputs: int,
    fs: float,
    gain_db: ArrayLike | None,
    gain_db_nyquist: ArrayLike | None,
    crossover: float | None,
) -> np.ndarray | None:
    """Design a per-output first-order shelf."""
    if gain_db is None and gain_db_nyquist is None:
        if crossover is not None:
            raise ValueError(
                "output_gain_db or output_gain_db_nyquist must be set to "
                "configure the post_output hook"
            )
        return None
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
        fs,
        crossover=crossover,
    )


def fdn_dss_gallery(
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
) -> tuple[DSS, dict[str, np.ndarray | None]]:
    """Generate a random FDN: ``(dss, filters)``.

    The feedback matrix is random orthogonal. Sampled delays support the same
    distributions and coprimality option as :func:`sample_delay_lengths`.

    In-loop attenuation (``filters["post_delay"]``) and the optional output EQ
    (``filters["post_output"]``) are first-order shelf SOS banks designed at
    ``fs``, ready for ``pyFDN.dss_to_td(*dss, **filters)``.
    """
    if fs <= 0:
        raise ValueError("fs must be positive")
    if rt is not None and rt <= 0:
        raise ValueError("rt must be positive")
    if rt_nyquist is not None and rt_nyquist <= 0:
        raise ValueError("rt_nyquist must be positive")

    local_rng = as_generator(0 if rng is None else rng)
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

    if N < 1:
        raise ValueError("N must be positive")
    if np.any(delays_array < 1):
        raise ValueError("all delays must be positive")

    A = random_orthogonal(N, rng=local_rng)
    B, C, D = _build_io_matrices(
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
    if rt is not None:
        from ..eq import decay_to_first_order_shelf

        rt_nyquist = rt if rt_nyquist is None else rt_nyquist
        post_delay = decay_to_first_order_shelf(
            rt, rt_nyquist, delays_array, float(fs), crossover=rt_crossover
        )

    post_output = _build_post_output(
        num_outputs,
        float(fs),
        output_gain_db,
        output_gain_db_nyquist,
        output_crossover,
    )
    filters = {
        "post_delay": post_delay,
        "post_matrix": None,
        "post_output": post_output,
    }
    return DSS(delays_array, A, B, C, D), filters


__all__ = ["fdn_dss_gallery"]
