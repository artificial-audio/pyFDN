"""Generate delay-line lengths for an FDN.

Delay lengths are drawn within a range according to one of three
distributions and can optionally be made mutually coprime, which is the
classic strategy for avoiding coinciding echoes and degenerate (overlapping)
modes in a feedback delay network.
"""

from __future__ import annotations

from math import gcd, log

import numpy as np

_DISTRIBUTIONS = ("uniform", "lognormal", "geometric")


def _as_generator(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Coerce a generator, integer seed, or ``None`` into a Generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _sample(
    N: int,
    low: int,
    high: int,
    distribution: str,
    generator: np.random.Generator,
) -> np.ndarray:
    """Draw ``N`` real-valued delay targets in ``[low, high]``."""
    if distribution == "uniform":
        # Flat in linear space.
        return generator.uniform(low, high, size=N)

    if distribution == "geometric":
        # Flat in log space: geometrically spaced, equal probability per octave.
        u = generator.uniform(0.0, 1.0, size=N)
        return low * (high / low) ** u

    if distribution == "lognormal":
        # Gaussian in log space, centred on the geometric mean of the range.
        # ``sigma`` defaults so that ``[low, high]`` spans roughly ±2 sigma.
        log_mid = 0.5 * (log(low) + log(high))
        log_sigma = 0.25 * (log(high) - log(low))
        samples = np.exp(generator.normal(log_mid, log_sigma, size=N))
        return np.clip(samples, low, high)

    raise ValueError(
        f"Unknown distribution {distribution!r}. Supported: {_DISTRIBUTIONS}"
    )


def _nearest_coprime(targets: np.ndarray, low_clip: int = 2) -> np.ndarray:
    """Snap real-valued targets to distinct, pairwise-coprime integers.

    Each target is replaced by the nearest integer that shares no common
    factor with any previously chosen delay, which keeps the result close to
    the requested distribution while guaranteeing pairwise coprimality (and
    therefore distinct lengths). Values are never pushed below ``low_clip``.
    """
    chosen: list[int] = []
    result = np.empty(targets.size, dtype=int)
    # Assign larger targets first: they have more room to move and primes are
    # sparser, so collisions are resolved where there is the most freedom.
    order = np.argsort(targets)[::-1]
    for idx in order:
        target = max(int(round(targets[idx])), low_clip)
        offset = 0
        while True:
            candidates = (target,) if offset == 0 else (target + offset, target - offset)
            for candidate in candidates:
                if candidate < low_clip or candidate in chosen:
                    continue
                if all(gcd(candidate, other) == 1 for other in chosen):
                    chosen.append(candidate)
                    result[idx] = candidate
                    break
            else:
                offset += 1
                continue
            break
    return result


def sample_delay_lengths(
    N: int,
    delay_range: tuple[int, int] = (400, 1200),
    *,
    distribution: str = "uniform",
    coprime: bool = False,
    sort: bool = False,
    sigma: float | None = None,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Generate ``N`` delay-line lengths in samples.

    Targets are drawn within ``delay_range`` according to ``distribution`` and,
    when ``coprime`` is set, snapped to the nearest distinct, pairwise-coprime
    integers. A local :class:`numpy.random.Generator` is used so passing an
    integer seed (or generator) makes the result reproducible without touching
    NumPy's global random state.

    Args:
        N: Number of delay lines.
        delay_range: Inclusive ``(low, high)`` range in samples.
        distribution: Sampling distribution for the delay lengths:

            * ``"uniform"`` – flat in linear space.
            * ``"geometric"`` – flat in log space (log-uniform), i.e.
              geometrically spaced with equal probability per octave.
            * ``"lognormal"`` – Gaussian in log space, centred on the
              geometric mean of the range. The range span roughly ±2 sigma.

        coprime: When ``True``, snap the sampled values to the nearest distinct,
            pairwise-coprime integers. Coprime delays avoid coinciding echoes
            and degenerate modes; snapping may nudge values slightly outside
            ``delay_range``.
        sort: Sort the returned delays in ascending order.
        rng: Local NumPy generator or integer seed.

    Returns:
        Integer array of shape ``(N,)`` with the delay lengths in samples.

    Example::

        sample_delay_lengths(8)                                   # uniform, may repeat
        sample_delay_lengths(8, (500, 4000), distribution="geometric")
        sample_delay_lengths(8, (500, 4000), distribution="lognormal", coprime=True)
    """
    if N < 1:
        raise ValueError("N must be positive")
    low, high = delay_range
    if low < 1 or high <= low:
        raise ValueError("delay_range must satisfy 1 <= low < high")
    if sigma is not None and sigma <= 0:
        raise ValueError("sigma must be positive")

    generator = _as_generator(rng)
    targets = _sample(N, low, high, distribution, generator)

    if coprime:
        delays = _nearest_coprime(targets, low_clip=2)
    else:
        delays = np.clip(np.round(targets), 1, None).astype(int)

    if sort:
        delays = np.sort(delays)
    return delays
