"""Conservative static-DSS response bounds; inconclusive cases use direct checks.

For P(z)=diag(z**m)-A and Hhat=sum u_k v_k w_k/(z-p_k),
P Hhat-I = Q(z) + sum P(p_k)u_k v_k w_k/(z-p_k).
Q has degree max(m)-1. Its coefficient norms and the small null-vector
residuals bound the error on the original validation grid without dense
modal sums at every point. Blocked compensated moments limit cancellation.
See the report for the norm inequalities and floating-point allowances.
"""

from collections.abc import Callable

import numpy as np


def dss_response_bound(
    poles: np.ndarray,
    outputs: np.ndarray,
    inputs: np.ndarray,
    weights: np.ndarray,
    points: np.ndarray,
    a: np.ndarray,
    delays: np.ndarray,
    tolerance: float,
    probe: Callable[[complex], np.ndarray],
) -> tuple[float, float] | None:
    """Return L2/peak upper bounds only if both certify the requested tolerance.

    Assumes ordinary IEEE round-to-nearest arithmetic and stable BLAS kernels;
    this is not directed-rounding interval arithmetic. The original direct
    validator remains the fallback for unsupported or inconclusive cases.
    """
    order, n = poles.size, delays.size
    maximum_delay = int(np.max(delays))
    if (
        n < 32
        or order < 512
        or maximum_delay > 256
        or maximum_delay * n * n * 16 > 128 * 2**20
        or not np.isrealobj(a)
        or a.shape != (n, n)
        or outputs.shape != (order, n)
        or inputs.shape != (order, n)
        or weights.shape != (order,)
        # Keep the integer-power roundoff model in a moderate-magnitude domain.
        or np.any(abs(poles) < 0.5)
        or np.any(abs(poles) > 2.0)
    ):
        return None
    eps = np.finfo(float).eps
    inflation = 1 + 32 * eps * max(order, n * n, maximum_delay)

    def gamma(operations: int | np.ndarray) -> float | np.ndarray:
        value = np.asarray(operations) * eps
        return value / (1 - value)

    radius = float(abs(points[0]))
    low, high = radius * (1 - 8 * eps), radius * (1 + 8 * eps)
    # Gershgorin on A.T A gives a norm upper bound with a dot-product allowance.
    absolute_a = abs(a)
    gram = a.T @ a
    gram_guard = gamma(2 * n + 8) * (absolute_a.T @ absolute_a)
    norm_a = float(np.sqrt(np.max(np.sum(abs(gram) + gram_guard, axis=1)))) * inflation
    inverse_gap = low ** int(np.min(delays)) - norm_a
    norm_p = high**maximum_delay + norm_a
    if not np.isfinite(inverse_gap) or inverse_gap <= np.sqrt(eps) * norm_p:
        return None

    coefficients = np.zeros((maximum_delay, n, n), dtype=complex)
    roundoff = np.zeros((n, n))
    powers = np.ones(order, dtype=complex)
    absolute_powers = np.ones(order)
    absolute_outputs, absolute_inputs = abs(outputs), abs(inputs)
    absolute_weights = abs(weights)
    for t in range(maximum_delay):
        moment = np.zeros((n, n), dtype=complex)
        compensation = np.zeros_like(moment)
        for start in range(0, order, 32):
            stop = start + 32
            product = (
                outputs[start:stop].T * (weights[start:stop] * powers[start:stop])
            ) @ inputs[start:stop]
            increment = product - compensation
            total = moment + increment
            compensation = (total - moment) - increment
            moment = total
        active = np.flatnonzero(delays > t)
        coefficients[delays[active] - 1 - t, active] = moment[active]
        absolute_moment = absolute_outputs.T @ (
            (absolute_weights * absolute_powers)[:, None] * absolute_inputs
        )
        # Complex products, power recurrence, short dots, and compensated sums.
        roundoff[active] += (
            gamma(8 * 32 + 8 * t + 32)
            * high ** (delays[active, None] - 1 - t)
            * absolute_moment[active]
        )
        powers *= poles
        absolute_powers *= abs(poles)
    coefficients[0] -= np.eye(n)
    polynomial_bound = (
        float(
            np.linalg.norm(
                np.sum(
                    abs(coefficients) * high ** np.arange(maximum_delay)[:, None, None],
                    axis=0,
                )
                + roundoff
            )
        )
        * inflation
    )

    pole_powers = poles[:, None] ** delays
    residuals = pole_powers * outputs - outputs @ a.T
    # A is real: separate the real-dot error from complex integer powers.
    residual_guard = (
        gamma(2 * n + 8) * (absolute_outputs @ absolute_a.T)
        + gamma(8 * delays[None, :] + 16) * abs(pole_powers) * absolute_outputs
    )
    residual_weights = (
        absolute_weights
        * np.linalg.norm(inputs, axis=1)
        * (np.linalg.norm(residuals, axis=1) + np.linalg.norm(residual_guard, axis=1))
        * inflation
    )
    bounds = np.empty(points.size)
    for start in range(0, points.size, 64):
        z = points[start : start + 64]
        distances = abs(z[:, None] - poles) * (1 - 8 * eps)
        rational_bound = (
            np.sum(residual_weights[None, :] / distances, axis=1) * inflation
        )
        bounds[start : start + 64] = (polynomial_bound + rational_bound) / inverse_gap
    reference_frobenius_lower = np.sqrt(n) / norm_p
    relative = (
        float(
            np.linalg.norm(bounds) / (np.sqrt(points.size) * reference_frobenius_lower)
        )
        * inflation
    )
    if not np.isfinite(relative) or relative > tolerance:
        return None

    peak_lower = 0.0
    for index in np.linspace(0, points.size - 1, min(64, points.size)).astype(int):
        point = complex(points[index])
        reference = probe(point)
        if reference.shape != (n, n) or not np.all(np.isfinite(reference)):
            return None
        p = np.diag(point**delays) - a
        product = p @ reference
        residual = product - np.eye(n)
        power_guard = gamma(8 * delays + 16) * high**delays + eps * (
            high**delays + abs(np.diag(a))
        )
        guard = (
            power_guard[:, None] * abs(reference)
            + gamma(8 * n + 8) * (abs(p) @ abs(reference))
            + eps * (abs(product) + np.eye(n))
        )
        solve_error = (
            float(np.linalg.norm(abs(residual) + guard)) * inflation / inverse_gap
        )
        peak_lower = max(peak_lower, float(np.max(abs(reference))) - solve_error)
    peak = float(np.max(bounds) / peak_lower) * inflation if peak_lower > 0 else np.inf
    if not np.isfinite(peak) or peak > tolerance:
        return None
    return float(relative), float(peak)
