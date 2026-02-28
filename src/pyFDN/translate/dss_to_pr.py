"""Delay state-space to poles/residues (dss2pr translation).

This module translates fdnToolbox's ``dss2pr.m`` pipeline and supports both
analytic and autograd z-domain probing backends for FLAMO graphs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.filters import ZFilter, ZScalar
from pyFDN.auxiliary.flamo_autograd_probe import flamo_graph_to_autograd_zfilter
from pyFDN.auxiliary.flamo_probe import flamo_graph_to_zfilter


class _ZFilterLeaf:
    """Wrap a ZFilter as a leaf accepted by ``flamo_probe``."""

    def __init__(self, zf: ZFilter):
        self._zf = zf
        n, m = zf.size()
        self.input_channels = m
        self.output_channels = n

    def at(self, z: complex | np.ndarray) -> np.ndarray:
        return self._zf.at(z)

    def der(self, z: complex | np.ndarray) -> np.ndarray:
        return self._zf.der(z)


def _as_probe_filter(
    value: Any,
    *,
    name: str,
    delay_units_hint: int | None = None,
    probe_backend: str = "manual",
) -> tuple[ZFilter, int, ZFilter | None]:
    """
    Convert numeric / ZFilter / FLAMO graph to a probe-capable ZFilter adapter.

    Returns
    -------
    probe_filter:
        A ZFilter-like object exposing ``at``/``der`` via flamo probing.
    delay_units:
        Delay-unit count used by dss2pr pole-count initialization.
    original_zfilter:
        The underlying ZFilter (if available), else ``None``.
    """
    backend = str(probe_backend).lower()
    if backend not in {"manual", "autograd"}:
        raise ValueError("probe_backend must be 'manual' or 'autograd'")

    if isinstance(value, ZFilter):
        zf = value
        # Keep numeric/ZFilter leaves on the analytic adapter.
        return flamo_graph_to_zfilter(_ZFilterLeaf(zf)), int(zf.number_of_delay_units), zf

    if isinstance(value, (np.ndarray, list, tuple)):
        zf = ZFilter.from_any(value)
        # Keep numeric/ZFilter leaves on the analytic adapter.
        return flamo_graph_to_zfilter(_ZFilterLeaf(zf)), int(zf.number_of_delay_units), zf

    # Assume FLAMO graph-like object
    if delay_units_hint is None:
        warnings.warn(
            f"{name} was passed as a FLAMO graph without delay_units_hint; "
            "assuming 0 matrix delay units for pole initialization.",
            stacklevel=2,
        )
        delay_units = 0
    else:
        delay_units = int(delay_units_hint)
    adapter = (
        flamo_graph_to_autograd_zfilter
        if backend == "autograd"
        else flamo_graph_to_zfilter
    )
    return adapter(value), delay_units, None


def _rcond(mat: np.ndarray) -> float:
    try:
        cond = np.linalg.cond(mat)
    except np.linalg.LinAlgError:
        return 0.0
    if not np.isfinite(cond) or cond == 0:
        return 0.0
    return float(1.0 / cond)


def _adjugate(a: np.ndarray) -> np.ndarray:
    """Adjugate matrix, robust for singular/complex matrices."""
    arr = np.asarray(a, dtype=np.complex128)
    m, n = arr.shape
    if m != n:
        raise ValueError("Adjugate expects a square matrix.")
    if n < 2:
        return np.ones((1, 1), dtype=np.complex128)

    u, s, vh = np.linalg.svd(arr, full_matrices=True)
    v = vh.conj().T
    # Product of all singular values except the current one.
    s_ex = np.ones(n, dtype=np.complex128)
    for i in range(n):
        if n == 1:
            s_ex[i] = 1.0
        else:
            s_ex[i] = np.prod(np.delete(s, i))
    det_uv = np.linalg.det(u @ v.conj().T)
    return det_uv * ((v * s_ex.reshape(1, -1)) @ u.conj().T)


@dataclass
class _FDNLoop:
    """Probe helper replicating fdnToolbox's zFDNloop semantics."""

    delays: np.ndarray
    forward_probe: ZFilter
    forward_inv_probe: ZFilter
    feedback_probe: ZFilter
    feedback_inv_probe: ZFilter | None
    number_of_delay_units: int
    number_of_matrix_delays: int

    def __post_init__(self):
        self.delays = np.asarray(self.delays, dtype=np.float64).ravel()
        self.n = int(self.delays.size)

    def _delay_inv_at(self, z: complex) -> np.ndarray:
        return np.diag(np.power(z, self.delays)).astype(np.complex128)

    def _delay_inv_der(self, z: complex) -> np.ndarray:
        return np.diag(self.delays * np.power(z, self.delays - 1.0)).astype(np.complex128)

    def forward_at(self, z: complex) -> np.ndarray:
        return self._delay_inv_at(z) @ np.asarray(self.forward_inv_probe.at(z), dtype=np.complex128)

    def forward_der(self, z: complex) -> np.ndarray:
        d_delay = self._delay_inv_der(z)
        delay = self._delay_inv_at(z)
        f_inv = np.asarray(self.forward_inv_probe.at(z), dtype=np.complex128)
        df_inv = np.asarray(self.forward_inv_probe.der(z), dtype=np.complex128)
        return d_delay @ f_inv + delay @ df_inv

    def at(self, z: complex) -> np.ndarray:
        return self.forward_at(z) - np.asarray(self.feedback_probe.at(z), dtype=np.complex128)

    def der(self, z: complex) -> np.ndarray:
        return self.forward_der(z) - np.asarray(self.feedback_probe.der(z), dtype=np.complex128)

    # Reverse form used by quality checks for |z| > 1
    def forward_at_inv(self, z: complex) -> np.ndarray:
        left = self._delay_inv_at(1.0 / z)
        f_inv = np.asarray(self.forward_inv_probe.at(z), dtype=np.complex128)
        return left @ np.linalg.inv(f_inv)

    def feedback_at_inv(self, z: complex) -> np.ndarray:
        if self.feedback_inv_probe is not None:
            return np.asarray(self.feedback_inv_probe.at(z), dtype=np.complex128)
        return np.linalg.inv(np.asarray(self.feedback_probe.at(z), dtype=np.complex128))

    def at_rev(self, z: complex) -> np.ndarray:
        return self.forward_at_inv(1.0 / z) - self.feedback_at_inv(1.0 / z)

    def inverse_newton_step(self, z: complex) -> complex:
        # EAI denominator: (d/dz) log det P(z) = trace(P^{-1} P').
        p = self.at(z)
        dp = self.der(z)
        try:
            return np.trace(np.linalg.solve(p, dp))
        except np.linalg.LinAlgError:
            return np.inf + 0j


def _pole_quality(poles: np.ndarray, loop: _FDNLoop) -> np.ndarray:
    quality = np.zeros_like(poles, dtype=np.float64)
    for i, pole in enumerate(np.asarray(poles, dtype=np.complex128).ravel()):
        if np.abs(pole) > 1:
            m = loop.at_rev(1.0 / pole)
        else:
            m = loop.at(pole)

        if np.ndim(m) == 0:
            q = float(np.abs(m))
        else:
            q = _rcond(np.asarray(m, dtype=np.complex128))

        if np.max(np.abs(np.asarray(m))) > 1e10:
            q = 1e10
        quality[i] = q
    return quality


def _sort_by(a: np.ndarray, key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ind = np.argsort(key)
    return a[ind], ind


def _compute_deflation(
    it: int,
    poles: np.ndarray,
    inv_newton_step: complex,
    *,
    deflation_type: str,
    number_of_neighbors: int,
    deflation_max_error: float,
    steps: int,
) -> tuple[complex, bool]:
    pole = poles[it]
    if deflation_type == "fullDeflation":
        neighbor_distance = pole - poles
        neighbor_distance[it] = 1.0 / np.finfo(float).eps
        return np.sum(1.0 / neighbor_distance), True
    if deflation_type == "noDeflation":
        return 0.0 + 0.0j, False
    if deflation_type != "neighborDeflation":
        raise ValueError(f"Unknown deflation type: {deflation_type}")

    number_of_poles = poles.size
    if steps == 1:
        neighbor_deflation = 0.0 + 0.0j
        factor_nonneighbor = (number_of_poles - 1) / 2.0
    else:
        n_neigh = int(max(0, min(number_of_neighbors, number_of_poles - 1)))
        if n_neigh % 2 != 0:
            n_neigh -= 1
        if n_neigh <= 0:
            neighbor_deflation = 0.0 + 0.0j
            factor_nonneighbor = (number_of_poles - 1) / 2.0
        else:
            offsets = np.concatenate(
                [np.arange(-n_neigh // 2, 0), np.arange(1, n_neigh // 2 + 1)]
            )
            idx = (it + offsets) % number_of_poles
            neighbor_deflation = np.sum(1.0 / (pole - poles[idx]))
            factor_nonneighbor = (number_of_poles - n_neigh - 1) / 2.0

    equi_deflation = np.conj(pole) * factor_nonneighbor
    deflation = neighbor_deflation + equi_deflation

    if steps != 1 and np.abs(inv_newton_step - deflation) < deflation_max_error:
        return _compute_deflation(
            it,
            poles,
            inv_newton_step,
            deflation_type="fullDeflation",
            number_of_neighbors=number_of_neighbors,
            deflation_max_error=deflation_max_error,
            steps=steps,
        )
    return deflation, False


def _refine_pole_positions(
    poles: np.ndarray,
    loop: _FDNLoop,
    *,
    quality_threshold: float,
    maximum_iterations: int,
    deflation_type: str,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    poles, _ = _sort_by(poles, np.angle(poles))
    number_of_poles = poles.size

    step_counter = 0
    exact_counter = 0
    record_poles: list[np.ndarray] = [poles.copy()]

    number_of_neighbors = int(round(number_of_poles / 100.0 / 2.0) * 2.0)
    deflation_max_error = 1000.0

    quality = _pole_quality(poles, loop)
    quality_last = quality.copy()

    current_deflation_type = deflation_type
    if verbose:
        print(
            f"Ehrlich-Aberth Iteration with {number_of_poles} poles "
            f"and a maximum of {maximum_iterations} iterations"
        )

    for steps in range(1, maximum_iterations + 1):
        if current_deflation_type == "neighborDeflation":
            poles, sort_ind = _sort_by(poles, np.angle(poles))
            quality = quality[sort_ind]
            quality_last = quality_last[sort_ind]

        non_converged = np.where(quality > quality_threshold)[0]

        # Same late-stage switch as MATLAB reference.
        if non_converged.size < number_of_poles / 10.0:
            current_deflation_type = "fullDeflation"

        for it in non_converged:
            if quality[it] <= quality_threshold:
                continue
            step_counter += 1
            inv_newton = loop.inverse_newton_step(poles[it])
            deflation, is_exact = _compute_deflation(
                it,
                poles,
                inv_newton,
                deflation_type=current_deflation_type,
                number_of_neighbors=number_of_neighbors,
                deflation_max_error=deflation_max_error,
                steps=steps,
            )
            denom = inv_newton - deflation
            if not np.isfinite(denom) or np.abs(denom) == 0:
                continue
            poles[it] = poles[it] - 1.0 / denom
            quality[it] = _pole_quality(np.array([poles[it]]), loop)[0]
            exact_counter += int(is_exact)

        if verbose:
            record_poles.append(poles.copy())

        max_improvement = float(np.abs(np.max(quality_last - quality)))
        if max_improvement < quality_threshold:
            if verbose:
                print("No further improvement possible")
            break
        if verbose:
            print(
                f"Iteration: {steps}, Maximum Improvement: {max_improvement}, "
                f"Worst Pole Quality: {np.max(quality)}, "
                f"Number of Non-converged Poles: {non_converged.size}"
            )
        quality_last = quality.copy()

    if verbose:
        print(f"Number of Exact Deflations: {exact_counter}")

    meta = {
        "stepCounter": int(step_counter),
        "exactCounter": int(exact_counter),
        "recordNeighborDeflation": [],
        "recordNewton": [],
        "recordPoles": np.asarray(record_poles, dtype=np.complex128),
    }
    return poles, quality, meta


def _reduce_conjugate_pairs(
    poles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    number_of_poles = poles.size

    # nearestneighbour([real; imag], [real; -imag]) translation
    lhs = np.column_stack([np.real(poles), np.imag(poles)])
    rhs = np.column_stack([np.real(poles), -np.imag(poles)])
    dist = np.linalg.norm(lhs[:, None, :] - rhs[None, :, :], axis=2)
    pair_index = np.argmin(dist, axis=1)

    pair_type = np.zeros(number_of_poles, dtype=int)
    for it in range(number_of_poles):
        if pair_type[it] != 0:
            continue
        if it == pair_index[it]:
            pair_type[it] = 1  # self-pair (real pole)
        elif it == pair_index[pair_index[it]]:
            pair_type[it] = 2
            pair_type[pair_index[it]] = 3
        else:
            pair_type[it] = -1

    if np.any(pair_type == -1):
        warnings.warn(
            f"{np.sum(pair_type == -1)} poles could not be paired",
            stacklevel=2,
        )

    is_conjugate = np.ones(number_of_poles, dtype=bool)
    is_conjugate[pair_type == 1] = False

    non_paired = poles[pair_type == -1]
    select = (pair_type == 1) | (pair_type == 2) | (pair_type == -1)
    poles_out = poles[select]
    poles_out = np.real(poles_out) + 1j * np.abs(np.imag(poles_out))
    is_conjugate_out = is_conjugate[select]
    return poles_out, is_conjugate_out, non_paired


def _evaluate_direct_term(direct: Any) -> np.ndarray:
    if isinstance(direct, ZFilter):
        return np.asarray(direct.at(1.0 + 0j), dtype=np.complex128)
    if isinstance(direct, (np.ndarray, list, tuple)):
        return np.asarray(direct, dtype=np.complex128)
    probe, _, _ = _as_probe_filter(direct, name="D")
    return np.asarray(probe.at(1.0 + 0j), dtype=np.complex128)


def _dss_to_res(
    poles: np.ndarray,
    loop: _FDNLoop,
    b_probe: ZFilter,
    c_probe: ZFilter,
    direct: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    number_of_poles = poles.size

    b_ref = np.asarray(b_probe.at(1.0 + 0j), dtype=np.complex128)
    c_ref = np.asarray(c_probe.at(1.0 + 0j), dtype=np.complex128)
    number_of_inputs = b_ref.shape[1]
    number_of_outputs = c_ref.shape[0]
    n = loop.n

    r_den = np.zeros(number_of_poles, dtype=np.complex128)
    for it, pole in enumerate(poles):
        r_den[it] = np.trace(_adjugate(loop.at(pole)) @ loop.der(pole))

    with np.errstate(divide="ignore", invalid="ignore"):
        undriven = 1.0 / r_den
    is_multiple = ~np.isfinite(undriven)
    if np.any(is_multiple):
        warnings.warn("There are multipoles. The residues are set to zero.", stacklevel=2)
        undriven[is_multiple] = 0.0
        r_den[is_multiple] = np.inf

    r_nom = np.zeros((number_of_poles, number_of_outputs, number_of_inputs), dtype=np.complex128)
    eig_right = np.zeros((n, number_of_poles), dtype=np.complex128)
    eig_left = np.zeros((n, number_of_poles), dtype=np.complex128)

    for it, pole in enumerate(poles):
        b = np.asarray(b_probe.at(pole), dtype=np.complex128)
        c = np.asarray(c_probe.at(pole), dtype=np.complex128)
        l = np.asarray(loop.at(pole), dtype=np.complex128)
        adj_p = _adjugate(l)

        r_nom[it, :, :] = c @ adj_p @ b

        # Rank-1 decomposition via dominant singular triplet.
        u, s, vh = np.linalg.svd(adj_p, full_matrices=False)
        s1 = s[0] if s.size > 0 else 0.0
        denom = np.sqrt(r_den[it])
        if np.abs(denom) > 0 and np.isfinite(denom):
            eig_right[:, it] = u[:, 0] * np.sqrt(s1) / denom
            eig_left[:, it] = vh.conj().T[:, 0] * np.conj(np.sqrt(s1)) / np.conj(denom)

    with np.errstate(divide="ignore", invalid="ignore"):
        driven = r_nom / r_den[:, None, None]
    driven = np.where(np.isfinite(driven), driven, 0.0)
    direct_term = _evaluate_direct_term(direct)
    eigenvectors = {"right": eig_right, "left": eig_left}
    return driven, direct_term, undriven, eigenvectors


def dss_to_pr(
    delays: ArrayLike,
    A: Any,
    B: Any,
    C: Any,
    D: Any,
    *,
    inverse_matrix: Any | None = None,
    deflation_type: str = "fullDeflation",
    absorption_filters: Any | None = None,
    reject_unstable_poles: bool = False,
    quality_threshold: float | None = None,
    maximum_iterations: int = 50,
    verbose: bool = True,
    feedback_delay_units: int | None = None,
    probe_backend: str = "manual",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    From delay state-space to poles and residues (fdnToolbox dss2pr translation).

    Parameters
    ----------
    delays : array-like
        Delay lengths in samples, shape ``(N,)``.
    A, B, C, D
        Feedback / input / output / direct terms.
        Each may be numeric, ``ZFilter``, or a FLAMO graph.
    inverse_matrix : optional
        Optional inverse of ``A`` (same accepted types as ``A``).
    deflation_type : {"fullDeflation", "noDeflation", "neighborDeflation"}
        Ehrlich-Aberth deflation type.
    absorption_filters : optional
        Per-delay absorption filter (typically diagonal ``ZFilter``).
        Defaults to identity.
    reject_unstable_poles : bool
        If ``True``, poles with ``|p| > 1`` are removed.
    quality_threshold : float, optional
        Pole quality threshold (default: ``1000 * eps``).
    maximum_iterations : int
        Maximum EAI iterations.
    verbose : bool
        Print refinement progress.
    feedback_delay_units : int, optional
        Delay-unit hint when ``A`` is passed as a FLAMO graph.
    probe_backend : {"manual", "autograd"}
        z-domain probing backend for FLAMO graph inputs. ``"autograd"``
        evaluates module values through the graph and obtains derivatives via
        torch autograd.

    Returns
    -------
    residues, poles, direct, is_conjugate_pole_pair, meta_data
    """
    # FLAMO-graph probing in this legacy entrypoint is kept for compatibility.
    # Prefer dss_to_pr_flamo / dss_to_pr_flamo_direct for new code.
    graph_like = any(
        not isinstance(v, (np.ndarray, list, tuple, ZFilter)) and v is not None
        for v in (A, B, C, absorption_filters, inverse_matrix)
    )
    if graph_like or str(probe_backend).lower() != "manual":
        warnings.warn(
            "dss_to_pr FLAMO probing paths are compatibility mode. "
            "Use dss_to_pr_flamo (or dss_to_pr_flamo_direct) for the maintained FLAMO architecture.",
            DeprecationWarning,
            stacklevel=2,
        )

    delays_arr = np.asarray(delays, dtype=int).ravel()
    if delays_arr.ndim != 1 or delays_arr.size == 0:
        raise ValueError("delays must be a non-empty 1-D array")

    n = delays_arr.size
    if quality_threshold is None:
        quality_threshold = 1000.0 * np.finfo(float).eps

    # Setup loop terms
    if absorption_filters is None:
        absorption_filters = ZScalar(np.ones((n, 1), dtype=np.float64), is_diagonal=True)

    fwd_probe, fwd_delay_units, fwd_zf = _as_probe_filter(
        absorption_filters,
        name="absorption_filters",
        delay_units_hint=0,
        probe_backend=probe_backend,
    )
    if fwd_zf is None:
        raise ValueError(
            "absorption_filters as FLAMO graph is not supported in dss_to_pr yet. "
            "Pass a numeric matrix/SOS or ZFilter."
        )
    fwd_inv_probe = flamo_graph_to_zfilter(_ZFilterLeaf(fwd_zf.inverse()))

    fb_probe, fb_delay_units, fb_zf = _as_probe_filter(
        A,
        name="A",
        delay_units_hint=feedback_delay_units,
        probe_backend=probe_backend,
    )
    fb_inv_probe: ZFilter | None = None
    if inverse_matrix is not None:
        fb_inv_probe, _, _ = _as_probe_filter(
            inverse_matrix,
            name="inverse_matrix",
            delay_units_hint=0,
            probe_backend=probe_backend,
        )
    elif fb_zf is not None:
        fb_inv_probe = flamo_graph_to_zfilter(_ZFilterLeaf(fb_zf.inverse()))

    b_probe, _, _ = _as_probe_filter(
        B, name="B", delay_units_hint=0, probe_backend=probe_backend
    )
    c_probe, _, _ = _as_probe_filter(
        C, name="C", delay_units_hint=0, probe_backend=probe_backend
    )

    loop = _FDNLoop(
        delays=delays_arr,
        forward_probe=fwd_probe,
        forward_inv_probe=fwd_inv_probe,
        feedback_probe=fb_probe,
        feedback_inv_probe=fb_inv_probe,
        number_of_matrix_delays=int(fb_delay_units),
        number_of_delay_units=int(np.sum(delays_arr) + fwd_delay_units + fb_delay_units),
    )

    number_of_poles = int(loop.number_of_delay_units)
    if number_of_poles <= 0:
        raise ValueError("Computed number_of_poles must be positive.")

    # Pole initialization on the unit circle
    pole_angles = np.linspace(0.0, 2.0 * np.pi, number_of_poles, endpoint=False)
    poles = np.exp(1j * pole_angles)

    poles, quality, meta_refine = _refine_pole_positions(
        poles,
        loop,
        quality_threshold=float(quality_threshold),
        maximum_iterations=int(maximum_iterations),
        deflation_type=str(deflation_type),
        verbose=bool(verbose),
    )

    meta_data: dict[str, Any] = dict(meta_refine)
    meta_data["refinedPoles"] = poles.copy()

    if reject_unstable_poles:
        stable = np.abs(poles) <= 1.0
        poles = poles[stable]
        quality = quality[stable]

    is_converged = quality < float(quality_threshold) * 1000.0
    poles = poles[is_converged]
    meta_data["convergedPoles"] = poles.copy()

    if poles.size != number_of_poles:
        warnings.warn(
            f"Some poles did not converge: {poles.size} instead of {number_of_poles}",
            stacklevel=2,
        )

    poles, is_conjugate_pole_pair, non_paired = _reduce_conjugate_pairs(poles)
    meta_data["nonPairedPoles"] = non_paired

    if verbose:
        final_count = int(np.sum(is_conjugate_pole_pair.astype(int) + 1))
        print(f"Final number of poles are: {final_count} of possible {number_of_poles}")

    residues, direct, undriven, eigenvectors = _dss_to_res(
        poles,
        loop,
        b_probe,
        c_probe,
        D,
    )
    meta_data["undrivenResidues"] = undriven
    meta_data["eigenvectors"] = eigenvectors

    return residues, poles, direct, is_conjugate_pole_pair, meta_data

