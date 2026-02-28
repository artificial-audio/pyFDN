"""FLAMO-only modal decomposition entry point (no ZFilter dependency here)."""

from __future__ import annotations

from collections import OrderedDict
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
import torch

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


def _infer_model_device(model: Any) -> torch.device:
    params = getattr(model, "parameters", None)
    if callable(params):
        try:
            first = next(params())
            return first.device
        except Exception:
            pass
    return torch.device("cpu")


def _infer_model_complex_dtype(model: Any) -> torch.dtype:
    dt = getattr(model, "dtype", None)
    if dt in (torch.float16, torch.float32):
        return torch.complex64
    return torch.complex128


def _as_torch_complex_scalar(z: complex, *, model: Any) -> torch.Tensor:
    return torch.tensor(
        complex(np.asarray(z, dtype=np.complex128)),
        device=_infer_model_device(model),
        dtype=_infer_model_complex_dtype(model),
    )


class _FlamoGraphProbe:
    """Probe adapter for FLAMO graph objects via autograd probing."""

    def __init__(self, model: Any):
        self.model = model
        h0 = np.asarray(self.at(1.0 + 0j), dtype=np.complex128)
        if h0.ndim != 2:
            raise ValueError(f"Graph probe at scalar z must be 2-D, got {h0.shape}")
        self.output_channels, self.input_channels = h0.shape

    def at(self, z: complex) -> np.ndarray:
        z_t = _as_torch_complex_scalar(z, model=self.model)
        out = self.model.probe(z_t)
        if isinstance(out, tuple):
            if len(out) == 0:
                raise RuntimeError("model.probe returned empty tuple")
            out = out[0]
        return np.asarray(out.detach().cpu().numpy(), dtype=np.complex128)

    def der(self, z: complex) -> np.ndarray:
        z_t = _as_torch_complex_scalar(z, model=self.model)
        if callable(getattr(self.model, "probe_with_derivative", None)):
            out = self.model.probe_with_derivative(z_t)
        else:
            out = self.model.probe(z_t, derivative=True)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise RuntimeError(
                "FLAMO model must expose derivative probe support "
                "(probe_with_derivative or probe(..., derivative=True))."
            )
        return np.asarray(out[1].detach().cpu().numpy(), dtype=np.complex128)


class _FlamoRecursionCharacteristicProbe:
    """Probe adapter for Recursion.probe_recursion(z) = I - F(z)B(z)."""

    def __init__(self, recursion: Any):
        self.recursion = recursion
        p0 = np.asarray(self.at(1.0 + 0j), dtype=np.complex128)
        if p0.ndim != 2:
            raise ValueError(f"Recursion characteristic probe must be 2-D, got {p0.shape}")
        self.output_channels, self.input_channels = p0.shape

    def at(self, z: complex) -> np.ndarray:
        z_t = _as_torch_complex_scalar(z, model=self.recursion)
        out = self.recursion.probe_recursion(z_t)
        if isinstance(out, tuple):
            if len(out) == 0:
                raise RuntimeError("probe_recursion returned empty tuple")
            out = out[0]
        return np.asarray(out.detach().cpu().numpy(), dtype=np.complex128)

    def der(self, z: complex) -> np.ndarray:
        z_t = _as_torch_complex_scalar(z, model=self.recursion)
        if callable(getattr(self.recursion, "probe_recursion_with_derivative", None)):
            out = self.recursion.probe_recursion_with_derivative(z_t)
        else:
            out = self.recursion.probe_recursion(z_t, derivative=True)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise RuntimeError(
                "Recursion must expose derivative characteristic probe support "
                "(probe_recursion_with_derivative or probe_recursion(..., derivative=True))."
            )
        return np.asarray(out[1].detach().cpu().numpy(), dtype=np.complex128)

    def log_det_derivative(self, z: complex) -> complex:
        """(d/dz) log det P(z) via FLAMO native recursion probe when available."""
        if not callable(getattr(self.recursion, "log_det_derivative", None)):
            return np.nan + 0j
        z_t = _as_torch_complex_scalar(z, model=self.recursion)
        out = self.recursion.log_det_derivative(z_t)
        return complex(out.detach().cpu().numpy())

    def log_det_derivative_w(self, w: complex) -> complex:
        """(d/dw) log det P(w) at w=z^{-1}, used for stable w-plane refinement."""
        if not callable(getattr(self.recursion, "log_det_derivative_w", None)):
            return np.nan + 0j
        w_t = _as_torch_complex_scalar(w, model=self.recursion)
        out = self.recursion.log_det_derivative_w(w_t)
        return complex(out.detach().cpu().numpy())


@dataclass
class _CharacteristicDecomposition:
    """Explicit transfer decomposition H(z)=C(z)P(z)^{-1}B(z)+D(z)."""

    p_probe: Any
    b_probe: Any
    c_probe: Any
    d_probe: Any


def _decomposition_to_public_dict(
    decomposition: _CharacteristicDecomposition,
) -> dict[str, Any]:
    return {
        "P": decomposition.p_probe,
        "B": decomposition.b_probe,
        "C": decomposition.c_probe,
        "D": decomposition.d_probe,
    }


def _coerce_decomposition(
    decomposition: Any,
) -> _CharacteristicDecomposition:
    if isinstance(decomposition, _CharacteristicDecomposition):
        return decomposition
    if isinstance(decomposition, dict):
        required = {"P", "B", "C", "D"}
        missing = required.difference(decomposition.keys())
        if missing:
            miss = ", ".join(sorted(missing))
            raise ValueError(f"decomposition is missing required keys: {miss}")
        return _CharacteristicDecomposition(
            p_probe=decomposition["P"],
            b_probe=decomposition["B"],
            c_probe=decomposition["C"],
            d_probe=decomposition["D"],
        )
    raise TypeError(
        "decomposition must be a dict with keys P/B/C/D "
        "or a _CharacteristicDecomposition instance."
    )


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


def flamo_extract_pr_decomposition(
    model: Any,
    delays: ArrayLike,
    *,
    recursion_module: Any,
) -> dict[str, Any]:
    """
    Extract explicit H(z)=C(z)P(z)^{-1}B(z)+D(z) probes from a FLAMO model.
    """
    delays_arr = np.asarray(delays, dtype=int).ravel()
    if delays_arr.ndim != 1 or delays_arr.size == 0:
        raise ValueError("delays must be a non-empty 1-D array")
    decomposition = _extract_flamo_recursion_probes(model, recursion_module, delays_arr)
    return _decomposition_to_public_dict(decomposition)


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
    characteristic_probe: Any
    number_of_delay_units: int

    def __post_init__(self):
        out_ch = int(getattr(self.characteristic_probe, "output_channels"))
        in_ch = int(getattr(self.characteristic_probe, "input_channels"))
        if out_ch != in_ch:
            raise ValueError(
                "Characteristic probe must be square, "
                f"got ({out_ch},{in_ch})."
            )
        self.n = out_ch

    def at(self, z: complex) -> np.ndarray:
        return np.asarray(self.characteristic_probe.at(z), dtype=np.complex128)

    def der(self, z: complex) -> np.ndarray:
        return np.asarray(self.characteristic_probe.der(z), dtype=np.complex128)

    def log_det_derivative_z(self, z: complex) -> complex:
        """Return (d/dz) log det P(z)."""
        probe = self.characteristic_probe
        if callable(getattr(probe, "log_det_derivative", None)):
            try:
                out = probe.log_det_derivative(z)
                if np.isfinite(out):
                    return out
            except Exception:
                pass
        p = self.at(z)
        dp = self.der(z)
        try:
            out = np.trace(np.linalg.solve(p, dp))
            if np.isfinite(out):
                return out
        except np.linalg.LinAlgError:
            pass
        return np.inf + 0j

    def log_det_derivative_w(self, w: complex) -> complex:
        """
        Return (d/dw) log det P(1/w), i.e. Newton term in w-domain.

        If FLAMO exposes log_det_derivative_w directly we use it; otherwise we
        convert from z-domain derivative with chain rule:
            d/dw log det P(1/w) = -(1/w^2) * d/dz log det P(z), z=1/w.
        """
        probe = self.characteristic_probe
        if callable(getattr(probe, "log_det_derivative_w", None)):
            try:
                out = probe.log_det_derivative_w(w)
                if np.isfinite(out):
                    return out
            except Exception:
                pass
        if np.abs(w) < 1e-14:
            return np.inf + 0j
        z = 1.0 / w
        nz = self.log_det_derivative_z(z)
        if not np.isfinite(nz):
            return np.inf + 0j
        return -(1.0 / (w * w)) * nz

    def inverse_newton_step(self, z: complex, *, use_w_plane_for_small_z: bool = True) -> complex:
        """Backwards-compatible alias for z-domain Newton term."""
        return self.log_det_derivative_z(z)

    def inverse_newton_step_w(self, w: complex) -> complex:
        """Newton term directly in w-domain."""
        return self.log_det_derivative_w(w)


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


def _pole_quality_w(roots_w: np.ndarray, loop: _FDNLoopFlamo) -> np.ndarray:
    """Pole quality evaluated in z-domain for roots represented in w-domain."""
    roots_w = np.asarray(roots_w, dtype=np.complex128).ravel()
    z = np.empty_like(roots_w)
    nonzero = np.abs(roots_w) > 1e-14
    z[nonzero] = 1.0 / roots_w[nonzero]
    z[~nonzero] = np.inf + 0j
    quality = np.zeros_like(roots_w, dtype=np.float64)
    valid = np.isfinite(z)
    if np.any(valid):
        quality[valid] = _pole_quality(z[valid], loop)
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


def _refine_pole_positions_w(
    roots_w: np.ndarray,
    loop: _FDNLoopFlamo,
    *,
    quality_threshold: float,
    maximum_iterations: int,
    deflation_type: str,
    verbose: bool,
    refinement_tol: float | None = None,
    use_w_plane_step: bool = True,
    use_w_plane_for_small_z: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    roots_w = np.asarray(roots_w, dtype=np.complex128).ravel()
    roots_w, _ = _sort_by(roots_w, np.angle(roots_w))
    n_poles = roots_w.size

    step_counter = 0
    exact_counter = 0
    record_roots_w: list[np.ndarray] = [roots_w.copy()]

    number_of_neighbors = int(round(n_poles / 100.0 / 2.0) * 2.0)
    deflation_max_error = 1000.0

    quality = _pole_quality_w(roots_w, loop)
    quality_last = quality.copy()
    current_deflation = deflation_type

    if not use_w_plane_step:
        warnings.warn(
            "use_w_plane_step=False is deprecated; refinement is now always in w-domain.",
            stacklevel=2,
        )
    _ = use_w_plane_for_small_z  # kept for API compatibility

    if verbose:
        print(
            f"Ehrlich-Aberth Iteration in w-domain with {n_poles} poles and a maximum of "
            f"{maximum_iterations} iterations"
        )

    steps = 0
    for steps in range(1, maximum_iterations + 1):
        if current_deflation == "neighborDeflation":
            roots_w, sort_ind = _sort_by(roots_w, np.angle(roots_w))
            quality = quality[sort_ind]
            quality_last = quality_last[sort_ind]

        roots_w_old = roots_w.copy()
        non_converged = np.where(quality > quality_threshold)[0]
        if non_converged.size < n_poles / 10.0:
            current_deflation = "fullDeflation"

        for it in non_converged:
            if quality[it] <= quality_threshold:
                continue
            step_counter += 1
            w_i = roots_w[it]
            inv_newton_w = loop.inverse_newton_step_w(w_i)
            deflation, is_exact = _compute_deflation(
                it,
                roots_w,
                inv_newton_w,
                deflation_type=current_deflation,
                number_of_neighbors=number_of_neighbors,
                deflation_max_error=deflation_max_error,
                steps=steps,
            )
            denom = inv_newton_w - deflation
            if not np.isfinite(denom) or np.abs(denom) < 1e-20:
                continue
            roots_w[it] = w_i - 1.0 / denom
            quality[it] = _pole_quality_w(np.array([roots_w[it]]), loop)[0]
            exact_counter += int(is_exact)

        if verbose:
            record_roots_w.append(roots_w.copy())

        if refinement_tol is not None:
            max_step = float(np.max(np.abs(roots_w - roots_w_old)))
            if max_step < refinement_tol:
                if verbose:
                    print(f"Converged (max |Δw| = {max_step:.3e} < {refinement_tol})")
                break
        else:
            max_improvement = float(np.abs(np.max(quality_last - quality)))
            if max_improvement < quality_threshold:
                if verbose:
                    print("No further improvement possible")
                break
        if verbose:
            max_improvement = float(np.abs(np.max(quality_last - quality)))
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
        "iterations": int(steps),
        "exactCounter": int(exact_counter),
        "recordNeighborDeflation": [],
        "recordNewton": [],
        "recordRootsW": np.asarray(record_roots_w, dtype=np.complex128),
    }
    return roots_w, quality, meta


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
    model: Any | None = None,
    delays: ArrayLike | None = None,
    *,
    recursion_module: Any | None = None,
    decomposition: Any | None = None,
    deflation_type: str = "fullDeflation",
    reject_unstable_poles: bool = False,
    quality_threshold: float | None = None,
    maximum_iterations: int = 50,
    verbose: bool = True,
    feedback_delay_units: int | None = 0,
    absorption_delay_units: int | None = 0,
    refinement_tol: float | None = None,
    use_w_plane_step: bool = True,
    use_w_plane_for_small_z: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Poles/residues from explicit H(z)=C(z)P(z)^{-1}B(z)+D(z) decomposition.

    Provide either:
      - ``decomposition`` with keys ``{"P","B","C","D"}``, or
      - ``model`` + ``recursion_module`` to extract that decomposition.

    Notes
    -----
    Refinement is performed in w-domain (``w = z^{-1}``) and converted back to
    z-domain only at the end before residue evaluation.
    ``use_w_plane_step`` and ``use_w_plane_for_small_z`` are retained for
    backward compatibility but no longer change this behavior.
    """
    if delays is None:
        raise ValueError("delays is required.")
    delays_arr = np.asarray(delays, dtype=int).ravel()
    if delays_arr.ndim != 1 or delays_arr.size == 0:
        raise ValueError("delays must be a non-empty 1-D array")

    if decomposition is None:
        if model is None or recursion_module is None:
            raise ValueError(
                "Either provide decomposition={P,B,C,D} or pass model and recursion_module."
            )
        decomposition_obj = _extract_flamo_recursion_probes(model, recursion_module, delays_arr)
    else:
        decomposition_obj = _coerce_decomposition(decomposition)

    fb_delay_units_i = int(feedback_delay_units or 0)
    fwd_delay_units_i = int(absorption_delay_units or 0)
    p_out = int(getattr(decomposition_obj.p_probe, "output_channels"))
    p_in = int(getattr(decomposition_obj.p_probe, "input_channels"))
    if p_out != delays_arr.size or p_in != delays_arr.size:
        raise ValueError(
            "Characteristic probe dimensions must match delay count: "
            f"expected ({delays_arr.size},{delays_arr.size}), got ({p_out},{p_in})."
        )

    if quality_threshold is None:
        quality_threshold = 1e-7

    loop = _FDNLoopFlamo(
        characteristic_probe=decomposition_obj.p_probe,
        number_of_delay_units=int(np.sum(delays_arr) + fwd_delay_units_i + fb_delay_units_i),
    )

    n_poles = int(loop.number_of_delay_units)
    if n_poles <= 0:
        raise ValueError("Computed number_of_poles must be positive.")

    # Initialize on unit circle in w-domain and refine there.
    root_angles = np.linspace(0.0, 2.0 * np.pi, n_poles, endpoint=False)
    roots_w = np.exp(1j * root_angles)

    roots_w, quality, meta_refine = _refine_pole_positions_w(
        roots_w,
        loop,
        quality_threshold=float(quality_threshold),
        maximum_iterations=int(maximum_iterations),
        deflation_type=str(deflation_type),
        verbose=bool(verbose),
        refinement_tol=refinement_tol,
        use_w_plane_step=bool(use_w_plane_step),
        use_w_plane_for_small_z=bool(use_w_plane_for_small_z),
    )

    meta_data: dict[str, Any] = dict(meta_refine)
    refined_roots_w = roots_w.copy()
    meta_data["refinedRootsW"] = refined_roots_w
    valid_refined = np.abs(refined_roots_w) > 1e-14
    refined_poles = np.full_like(refined_roots_w, np.inf + 0j)
    refined_poles[valid_refined] = 1.0 / refined_roots_w[valid_refined]
    meta_data["refinedPoles"] = refined_poles

    if reject_unstable_poles:
        stable = np.abs(roots_w) >= 1.0
        roots_w = roots_w[stable]
        quality = quality[stable]

    is_converged = quality < float(quality_threshold) * 1000.0
    if not np.any(is_converged):
        is_converged = np.ones_like(quality, dtype=bool)
    roots_w = roots_w[is_converged]
    meta_data["convergedRootsW"] = roots_w.copy()

    if roots_w.size != n_poles:
        warnings.warn(
            f"Some poles did not converge: {roots_w.size} instead of {n_poles}",
            stacklevel=2,
        )

    valid_conv = np.abs(roots_w) > 1e-14
    poles = np.full_like(roots_w, np.inf + 0j)
    poles[valid_conv] = 1.0 / roots_w[valid_conv]
    finite_mask = np.isfinite(poles)
    if not np.all(finite_mask):
        warnings.warn(
            f"Dropping {np.sum(~finite_mask)} non-finite poles from converged w roots.",
            stacklevel=2,
        )
        poles = poles[finite_mask]
    if poles.size == 0:
        fallback = refined_poles[np.isfinite(refined_poles)]
        if fallback.size == 0:
            raise ValueError("No finite poles available after w->z conversion.")
        warnings.warn(
            "No converged finite poles remained after filtering; using refined finite poles.",
            stacklevel=2,
        )
        poles = fallback
    meta_data["convergedPoles"] = poles.copy()

    poles, is_conjugate, non_paired = _reduce_conjugate_pairs(poles)
    meta_data["nonPairedPoles"] = non_paired
    if verbose:
        final_count = int(np.sum(is_conjugate.astype(int) + 1))
        print(f"Final number of poles are: {final_count} of possible {n_poles}")

    residues, direct, undriven, eigenvectors = _dss_to_res_flamo(
        poles, loop, decomposition_obj
    )
    meta_data["undrivenResidues"] = undriven
    meta_data["eigenvectors"] = eigenvectors
    meta_data["decomposition"] = _decomposition_to_public_dict(decomposition_obj)

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
    refinement_tol: float | None = None,
    use_w_plane_step: bool = True,
    use_w_plane_for_small_z: bool = True,
    dtype: Any = None,
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
        dtype=dtype,
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
        refinement_tol=refinement_tol,
        use_w_plane_step=use_w_plane_step,
        use_w_plane_for_small_z=use_w_plane_for_small_z,
    )

