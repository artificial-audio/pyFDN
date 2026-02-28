"""FLAMO-only modal decomposition entry point (no ZFilter dependency here)."""

from __future__ import annotations

from collections import OrderedDict
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.flamo_runtime_probe import (
    probe_flamo_recursion_runtime,
    probe_flamo_runtime,
)
from pyFDN.translate.dss_to_flamo import dss_to_flamo


class _IdentityProbe:
    """Constant identity matrix probe used for empty module chains."""

    def __init__(self, size: int):
        self.size = int(size)
        self.output_channels = self.size
        self.input_channels = self.size
        self._eye = np.eye(self.size, dtype=np.complex128)

    def at(self, z: complex) -> np.ndarray:
        return self._eye

    def der(self, z: complex) -> np.ndarray:
        return np.zeros_like(self._eye)


class _FlamoGraphProbe:
    """Probe adapter for FLAMO graph objects via autograd probing."""

    def __init__(self, model: Any):
        self.model = model
        h0 = np.asarray(
            probe_flamo_runtime(model, 1.0 + 0j, derivative=False), dtype=np.complex128
        )
        if h0.ndim != 2:
            raise ValueError(f"Graph probe at scalar z must be 2-D, got {h0.shape}")
        self.output_channels, self.input_channels = h0.shape

    def at(self, z: complex) -> np.ndarray:
        return np.asarray(
            probe_flamo_runtime(self.model, z, derivative=False), dtype=np.complex128
        )

    def der(self, z: complex) -> np.ndarray:
        _, dh = probe_flamo_runtime(self.model, z, derivative=True)
        return np.asarray(dh, dtype=np.complex128)


class _FlamoRecursionCharacteristicProbe:
    """Probe adapter for Recursion.probe_recursion(z) = I - F(z)B(z)."""

    def __init__(self, recursion: Any):
        self.recursion = recursion
        p0 = np.asarray(
            probe_flamo_recursion_runtime(recursion, 1.0 + 0j, derivative=False),
            dtype=np.complex128,
        )
        if p0.ndim != 2:
            raise ValueError(f"Recursion characteristic probe must be 2-D, got {p0.shape}")
        self.output_channels, self.input_channels = p0.shape

    def at(self, z: complex) -> np.ndarray:
        return np.asarray(
            probe_flamo_recursion_runtime(self.recursion, z, derivative=False),
            dtype=np.complex128,
        )

    def der(self, z: complex) -> np.ndarray:
        _, dp = probe_flamo_recursion_runtime(self.recursion, z, derivative=True)
        return np.asarray(dp, dtype=np.complex128)


@dataclass
class _CharacteristicDecomposition:
    """Explicit transfer decomposition H(z)=C(z)P(z)^{-1}B(z)+D(z)."""

    p_probe: Any
    b_probe: Any
    c_probe: Any
    d_probe: Any


def _as_module_list(node: Any) -> list[Any]:
    """Return modules in processing order for a FLAMO node/series."""
    try:
        modules = list(node)
    except Exception:
        return [node]
    if len(modules) == 0:
        return [node]
    return modules


def _build_flamo_series(modules: list[Any]) -> Any:
    """Build a FLAMO Series model from a list of modules."""
    if len(modules) == 1:
        return modules[0]
    try:
        from flamo.processor import system as flamo_system  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "FLAMO system.Series is required to compose branch subchains."
        ) from exc
    return flamo_system.Series(
        OrderedDict((f"m{i}", module) for i, module in enumerate(modules))
    )


def _probe_from_modules(modules: list[Any], *, identity_dim: int | None = None) -> Any:
    """Create a probe adapter from a FLAMO module chain."""
    if len(modules) == 0:
        if identity_dim is None:
            raise ValueError("Empty module chain requires identity_dim.")
        return _IdentityProbe(identity_dim)
    return _FlamoGraphProbe(_build_flamo_series(modules))


def _extract_flamo_recursion_probes(
    model: Any,
    recursion_module: Any,
    delays: np.ndarray,
) -> _CharacteristicDecomposition:
    """
    Extract B/C/D probes and loop probes from a dss_to_flamo-like graph.

    Expected core topology:
        Parallel(sum_output=True):
            branchA = Series(input_gain, Recursion(feedforward, feedback), output_gain)
            branchB = direct gain path
    """
    core = model.get_core() if callable(getattr(model, "get_core", None)) else model

    if not hasattr(core, "branchA") or not hasattr(core, "branchB"):
        raise ValueError(
            "flamo_to_pr expects a FLAMO core with branchA/branchB "
            "(e.g., produced by dss_to_flamo)."
        )

    fdn_branch = core.branchA
    direct_branch = core.branchB
    fdn_modules = _as_module_list(fdn_branch)

    rec_indices = [i for i, mod in enumerate(fdn_modules) if mod is recursion_module]
    if len(rec_indices) != 1:
        raise ValueError(
            "recursion_module is not uniquely contained in branchA. "
            "Pass the exact recursion instance used in the model."
        )
    rec_idx = int(rec_indices[0])

    n = int(np.asarray(delays).size)
    in_modules = fdn_modules[:rec_idx]
    out_modules = fdn_modules[rec_idx + 1 :]
    ff_modules = _as_module_list(recursion_module.feedforward)
    b_probe = _probe_from_modules(in_modules + ff_modules, identity_dim=n)
    c_probe = _probe_from_modules(out_modules, identity_dim=n)
    direct_probe = _probe_from_modules(_as_module_list(direct_branch))
    characteristic_probe = _FlamoRecursionCharacteristicProbe(recursion_module)
    return _CharacteristicDecomposition(
        p_probe=characteristic_probe,
        b_probe=b_probe,
        c_probe=c_probe,
        d_probe=direct_probe,
    )


def _rcond(mat: np.ndarray) -> float:
    try:
        cond = np.linalg.cond(mat)
    except np.linalg.LinAlgError:
        return 0.0
    if not np.isfinite(cond) or cond == 0:
        return 0.0
    return float(1.0 / cond)


@dataclass
class _FDNLoopFlamo:
    delays: np.ndarray
    characteristic_probe: Any
    number_of_delay_units: int
    number_of_matrix_delays: int

    def __post_init__(self):
        self.delays = np.asarray(self.delays, dtype=np.float64).ravel()
        self.n = int(self.delays.size)
        out_ch = int(getattr(self.characteristic_probe, "output_channels"))
        in_ch = int(getattr(self.characteristic_probe, "input_channels"))
        if out_ch != self.n or in_ch != self.n:
            raise ValueError(
                "Recursion characteristic probe dimensions must match "
                f"delay count {self.n}, got ({out_ch},{in_ch})."
            )

    def at(self, z: complex) -> np.ndarray:
        return np.asarray(self.characteristic_probe.at(z), dtype=np.complex128)

    def der(self, z: complex) -> np.ndarray:
        return np.asarray(self.characteristic_probe.der(z), dtype=np.complex128)

    def inverse_newton_step(self, z: complex) -> complex:
        p = self.at(z)
        dp = self.der(z)
        try:
            newton = np.trace(np.linalg.solve(p, dp))
        except np.linalg.LinAlgError:
            return np.inf + 0j
        return newton + self.number_of_delay_units / z


def _pole_quality(poles: np.ndarray, loop: _FDNLoopFlamo) -> np.ndarray:
    quality = np.zeros_like(poles, dtype=np.float64)
    for i, pole in enumerate(np.asarray(poles, dtype=np.complex128).ravel()):
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

    n_poles = poles.size
    if steps == 1:
        neighbor_deflation = 0.0 + 0.0j
        factor_nonneighbor = (n_poles - 1) / 2.0
    else:
        n_neigh = int(max(0, min(number_of_neighbors, n_poles - 1)))
        if n_neigh % 2 != 0:
            n_neigh -= 1
        if n_neigh <= 0:
            neighbor_deflation = 0.0 + 0.0j
            factor_nonneighbor = (n_poles - 1) / 2.0
        else:
            offsets = np.concatenate(
                [np.arange(-n_neigh // 2, 0), np.arange(1, n_neigh // 2 + 1)]
            )
            idx = (it + offsets) % n_poles
            neighbor_deflation = np.sum(1.0 / (pole - poles[idx]))
            factor_nonneighbor = (n_poles - n_neigh - 1) / 2.0

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
    loop: _FDNLoopFlamo,
    *,
    quality_threshold: float,
    maximum_iterations: int,
    deflation_type: str,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    poles, _ = _sort_by(poles, np.angle(poles))
    n_poles = poles.size

    step_counter = 0
    exact_counter = 0
    record_poles: list[np.ndarray] = [poles.copy()]

    number_of_neighbors = int(round(n_poles / 100.0 / 2.0) * 2.0)
    deflation_max_error = 1000.0

    quality = _pole_quality(poles, loop)
    quality_last = quality.copy()
    current_deflation = deflation_type

    if verbose:
        print(
            f"Ehrlich-Aberth Iteration with {n_poles} poles and a maximum of "
            f"{maximum_iterations} iterations"
        )

    for steps in range(1, maximum_iterations + 1):
        if current_deflation == "neighborDeflation":
            poles, sort_ind = _sort_by(poles, np.angle(poles))
            quality = quality[sort_ind]
            quality_last = quality_last[sort_ind]

        non_converged = np.where(quality > quality_threshold)[0]
        if non_converged.size < n_poles / 10.0:
            current_deflation = "fullDeflation"

        for it in non_converged:
            if quality[it] <= quality_threshold:
                continue
            step_counter += 1
            inv_newton = loop.inverse_newton_step(poles[it])
            deflation, is_exact = _compute_deflation(
                it,
                poles,
                inv_newton,
                deflation_type=current_deflation,
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
    n_poles = poles.size
    lhs = np.column_stack([np.real(poles), np.imag(poles)])
    rhs = np.column_stack([np.real(poles), -np.imag(poles)])
    dist = np.linalg.norm(lhs[:, None, :] - rhs[None, :, :], axis=2)
    pair_index = np.argmin(dist, axis=1)

    pair_type = np.zeros(n_poles, dtype=int)
    for it in range(n_poles):
        if pair_type[it] != 0:
            continue
        if it == pair_index[it]:
            pair_type[it] = 1
        elif it == pair_index[pair_index[it]]:
            pair_type[it] = 2
            pair_type[pair_index[it]] = 3
        else:
            pair_type[it] = -1

    if np.any(pair_type == -1):
        warnings.warn(f"{np.sum(pair_type == -1)} poles could not be paired", stacklevel=2)

    is_conjugate = np.ones(n_poles, dtype=bool)
    is_conjugate[pair_type == 1] = False
    non_paired = poles[pair_type == -1]
    select = (pair_type == 1) | (pair_type == 2) | (pair_type == -1)
    poles_out = poles[select]
    poles_out = np.real(poles_out) + 1j * np.abs(np.imag(poles_out))
    return poles_out, is_conjugate[select], non_paired


def _dss_to_res_flamo(
    poles: np.ndarray,
    loop: _FDNLoopFlamo,
    decomposition: _CharacteristicDecomposition,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    n_poles = poles.size

    b_ref = np.asarray(decomposition.b_probe.at(1.0 + 0j), dtype=np.complex128)
    c_ref = np.asarray(decomposition.c_probe.at(1.0 + 0j), dtype=np.complex128)
    n_in = b_ref.shape[1]
    n_out = c_ref.shape[0]
    n = loop.n

    r_den = np.zeros(n_poles, dtype=np.complex128)
    r_nom = np.zeros((n_poles, n_out, n_in), dtype=np.complex128)
    eig_right = np.zeros((n, n_poles), dtype=np.complex128)  # r_i
    eig_left = np.zeros((n, n_poles), dtype=np.complex128)   # l_i (for l_i^H P = 0)

    for it, pole in enumerate(poles):
        p = np.asarray(decomposition.p_probe.at(pole), dtype=np.complex128)
        dp = np.asarray(decomposition.p_probe.der(pole), dtype=np.complex128)
        b = np.asarray(decomposition.b_probe.at(pole), dtype=np.complex128)
        c = np.asarray(decomposition.c_probe.at(pole), dtype=np.complex128)

        # Null vectors for simple poles:
        # P(lambda) r = 0,  l^H P(lambda) = 0
        u, s, vh = np.linalg.svd(p, full_matrices=False)
        r = vh.conj().T[:, -1]
        l = u[:, -1]

        denom = np.vdot(l, dp @ r)  # l^H (dP/dz) r
        r_den[it] = denom
        eig_right[:, it] = r
        eig_left[:, it] = l

        cr = c @ r.reshape(-1, 1)
        lh_b = np.conj(l).reshape(1, -1) @ b
        r_nom[it, :, :] = cr @ lh_b

    with np.errstate(divide="ignore", invalid="ignore"):
        undriven = 1.0 / r_den
    is_multiple = ~np.isfinite(undriven)
    if np.any(is_multiple):
        warnings.warn("There are multipoles. The residues are set to zero.", stacklevel=2)
        undriven[is_multiple] = 0.0
        r_den[is_multiple] = np.inf

    with np.errstate(divide="ignore", invalid="ignore"):
        residues = r_nom / r_den[:, None, None]
    residues = np.where(np.isfinite(residues), residues, 0.0)
    direct_term = np.asarray(decomposition.d_probe.at(1.0 + 0j), dtype=np.complex128)
    eigenvectors = {"right": eig_right, "left": eig_left}
    return residues, direct_term, undriven, eigenvectors


def flamo_to_pr(
    model: Any,
    delays: ArrayLike,
    *,
    recursion_module: Any,
    deflation_type: str = "fullDeflation",
    reject_unstable_poles: bool = False,
    quality_threshold: float | None = None,
    maximum_iterations: int = 50,
    verbose: bool = True,
    feedback_delay_units: int | None = 0,
    absorption_delay_units: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Poles/residues directly from a FLAMO model and explicit recursion module.

    Passing ``recursion_module`` avoids ambiguous recursion discovery and makes
    the decomposition path strictly tied to FLAMO's native recursion probe API.
    """
    delays_arr = np.asarray(delays, dtype=int).ravel()
    if delays_arr.ndim != 1 or delays_arr.size == 0:
        raise ValueError("delays must be a non-empty 1-D array")

    decomposition = _extract_flamo_recursion_probes(model, recursion_module, delays_arr)
    fb_delay_units_i = int(feedback_delay_units or 0)
    fwd_delay_units_i = int(absorption_delay_units or 0)

    if quality_threshold is None:
        quality_threshold = 1000.0 * np.finfo(float).eps

    loop = _FDNLoopFlamo(
        delays=delays_arr,
        characteristic_probe=decomposition.p_probe,
        number_of_matrix_delays=fb_delay_units_i,
        number_of_delay_units=int(np.sum(delays_arr) + fwd_delay_units_i + fb_delay_units_i),
    )

    n_poles = int(loop.number_of_delay_units)
    if n_poles <= 0:
        raise ValueError("Computed number_of_poles must be positive.")

    # Pole initialization
    pole_angles = np.linspace(0.0, 2.0 * np.pi, n_poles, endpoint=False)
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
    if not np.any(is_converged):
        is_converged = np.ones_like(quality, dtype=bool)
    poles = poles[is_converged]
    meta_data["convergedPoles"] = poles.copy()

    if poles.size != n_poles:
        warnings.warn(
            f"Some poles did not converge: {poles.size} instead of {n_poles}",
            stacklevel=2,
        )

    poles, is_conjugate, non_paired = _reduce_conjugate_pairs(poles)
    meta_data["nonPairedPoles"] = non_paired
    if verbose:
        final_count = int(np.sum(is_conjugate.astype(int) + 1))
        print(f"Final number of poles are: {final_count} of possible {n_poles}")

    residues, direct, undriven, eigenvectors = _dss_to_res_flamo(
        poles, loop, decomposition
    )
    meta_data["undrivenResidues"] = undriven
    meta_data["eigenvectors"] = eigenvectors
    meta_data["decomposition"] = {
        "P": decomposition.p_probe,
        "B": decomposition.b_probe,
        "C": decomposition.c_probe,
        "D": decomposition.d_probe,
    }

    return residues, poles, direct, is_conjugate, meta_data


def dss_to_pr_flamo(
    delays: ArrayLike,
    A: Any,
    B: Any,
    C: Any,
    D: Any,
    *,
    deflation_type: str = "fullDeflation",
    inverse_matrix: Any | None = None,
    absorption_filters: Any | None = None,
    reject_unstable_poles: bool = False,
    quality_threshold: float | None = None,
    maximum_iterations: int = 50,
    verbose: bool = True,
    feedback_delay_units: int | None = 0,
    absorption_delay_units: int | None = 0,
    Fs: float = 1.0,
    nfft: int = 2**16,
    device: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    DSS -> FLAMO -> PR wrapper.

    Converts delay state-space to a FLAMO core using :func:`dss_to_flamo`, then
    decomposes that FLAMO model via :func:`flamo_to_pr`.
    """
    if absorption_filters is not None:
        raise ValueError(
            "dss_to_pr_flamo no longer accepts absorption_filters directly. "
            "Build a FLAMO model with dss_to_flamo(post_delay_module=...) and "
            "use flamo_to_pr(model, delays, recursion_module=...)."
        )

    if inverse_matrix is not None:
        raise ValueError(
            "inverse_matrix is not supported in strict FLAMO-native mode. "
            "Please rely on native Recursion.probe_recursion APIs."
        )

    delays_arr = np.asarray(delays, dtype=int).ravel()
    if delays_arr.ndim != 1 or delays_arr.size == 0:
        raise ValueError("delays must be a non-empty 1-D array")

    if not all(isinstance(v, (np.ndarray, list, tuple)) for v in (A, B, C, D)):
        raise TypeError(
            "dss_to_pr_flamo expects numeric DSS matrices (A,B,C,D). "
            "For existing FLAMO graph models, use "
            "flamo_to_pr(model, delays, recursion_module=...)."
        )

    model = dss_to_flamo(
        A=np.asarray(A, dtype=np.float64),
        B=np.asarray(B, dtype=np.float64),
        C=np.asarray(C, dtype=np.float64),
        D=np.asarray(D, dtype=np.float64),
        m=delays_arr,
        Fs=float(Fs),
        nfft=int(nfft),
        device=device,
        shell=False,
    )

    core = model.get_core() if callable(getattr(model, "get_core", None)) else model
    branch_a = getattr(core, "branchA", None)
    branch_modules = _as_module_list(branch_a)
    if len(branch_modules) < 2:
        raise ValueError("dss_to_flamo core branchA layout unexpected for recursion extraction.")
    recursion_module = branch_modules[1]
    if not (hasattr(recursion_module, "feedforward") and hasattr(recursion_module, "feedback")):
        raise ValueError("Expected recursion module at branchA index 1 in dss_to_flamo core.")

    return flamo_to_pr(
        model,
        delays_arr,
        recursion_module=recursion_module,
        deflation_type=deflation_type,
        reject_unstable_poles=reject_unstable_poles,
        quality_threshold=quality_threshold,
        maximum_iterations=maximum_iterations,
        verbose=verbose,
        feedback_delay_units=feedback_delay_units,
        absorption_delay_units=absorption_delay_units,
    )

