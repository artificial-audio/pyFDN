"""FLAMO-only modal decomposition entry point.

What FLAMO must provide
----------------------
- P(z): Recursion.probe_recursion(z).
- P(w): Recursion.probe_recursion_w(w).

All derivatives are built in _FDNLoopFlamo (grad for d/dw log det P(w), JVP for
dP/dz(z)) and stored on the loop. No derivative APIs required from FLAMO.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike

from pyFDN.auxiliary.poles import reduce_conjugate_pairs
from pyFDN.translate.dss_to_flamo import dss_to_flamo


class _IdentityProbe:
    """Constant identity matrix probe used for empty module chains. Returns torch."""

    def __init__(
        self,
        size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.size = int(size)
        self.output_channels = self.size
        self.input_channels = self.size
        self._device = device or torch.device("cpu")
        self._dtype = dtype or torch.complex128
        self._eye = torch.eye(self.size, device=self._device, dtype=self._dtype)
        self._zero = torch.zeros(
            self.size, self.size, device=self._device, dtype=self._dtype
        )

    def at_z(self, z: complex) -> torch.Tensor:
        return self._eye

    def at_w(self, w: complex) -> torch.Tensor:
        return self._eye

    def at(self, z: complex) -> torch.Tensor:
        return self.at_z(z)

    def der(self, z: complex) -> torch.Tensor:
        return self._zero


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


def _to_numpy(t: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensor to numpy; safe for conjugate bit. Pass-through for ndarray."""
    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().resolve_conj().numpy()


def _device_dtype_from_decomposition(
    decomposition: Any,
) -> tuple[torch.device, torch.dtype]:
    """Infer device and complex dtype from decomposition's P (recursion)."""
    rec = decomposition.p_probe
    return _infer_model_device(rec), _infer_model_complex_dtype(rec)


def _make_log_det_derivative_w(recursion: Any):
    """Build (d/dw) log det P(w) once using grad; uses recursion.probe_recursion_w(w)."""

    def d_log_det_w(w):
        w_var = _as_torch_complex_scalar(w, model=recursion)
        w_var = w_var.detach().clone().requires_grad_(True)
        P_w = recursion.probe_recursion_w(w_var)
        y = torch.logdet(P_w)
        (g,) = torch.autograd.grad(y, w_var, torch.ones_like(y))
        return g.conj().detach()

    return d_log_det_w


def _make_P_and_dP_dz(recursion: Any):
    """Build (P(z), dP/dz(z)) once using JVP; uses recursion.probe_recursion(z)."""

    def get_P_and_dP_dz(z):
        z_var = _as_torch_complex_scalar(z, model=recursion)
        dz = torch.ones_like(z_var)
        P, dP_dz = torch.autograd.functional.jvp(
            recursion.probe_recursion, (z_var,), (dz,)
        )
        return P.detach(), dP_dz.detach()

    return get_P_and_dP_dz


def _make_log_det_derivative_z(recursion: Any):
    """Build (d/dz) log det P(z) once using grad; uses recursion.probe_recursion(z)."""

    def d_log_det_z(z):
        z_var = (
            _as_torch_complex_scalar(z, model=recursion)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        P_z = recursion.probe_recursion(z_var)
        y = torch.logdet(P_z)
        (g,) = torch.autograd.grad(y, z_var, torch.ones_like(y))
        return g.conj().detach()

    return d_log_det_z


def _sort_by_torch(
    a: torch.Tensor, key: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort tensor a by key (by angle for complex); return (a_sorted, indices)."""
    key_np = _to_numpy(key)
    ind = np.argsort(key_np)
    ind_t = torch.as_tensor(ind, device=a.device, dtype=torch.long)
    return a[ind_t], ind_t


@dataclass
class _CharacteristicDecomposition:
    """H(z)=C(z)P(z)^{-1}B(z)+D(z) with probes for B,C,D."""

    p_probe: Any
    f_probe: Any
    in_probe: Any
    out_probe: Any
    direct_probe: Any


def _decomposition_to_public_dict(
    decomposition: _CharacteristicDecomposition,
) -> dict[str, Any]:
    """Public dict: P, F (feedforward), B (input path), C, D probes. H = C P^{-1} F B + D."""
    return {
        "P": decomposition.p_probe,
        "F": decomposition.f_probe,
        "B": decomposition.in_probe,
        "C": decomposition.out_probe,
        "D": decomposition.direct_probe,
    }


def _as_module_list(node: Any) -> list[Any]:
    """Return modules in processing order for a FLAMO node/series."""
    try:
        modules = list(node)
    except Exception:
        return [node]
    if len(modules) == 0:
        return [node]
    return modules


@dataclass
class FlamoDecompositionForPR:
    """
    Decomposition of a FLAMO model into small subgraphs for poles/residues.
    All subgraph fields are FLAMO modules (with .probe(z)); None means identity.
    """

    recursion_module: Any
    delays: np.ndarray
    in_subgraph: Any | None
    f_subgraph: Any
    out_subgraph: Any | None
    direct_subgraph: Any


def _series_slice_to_subgraph(series: Any, start: int, end: int) -> Any | None:
    """
    Return the slice of a FLAMO Series as a subgraph (same module refs).
    Returns None for empty slice, single module ref, or new Series(OrderedDict(slice)).
    """
    n = end - start
    if n <= 0:
        return None
    if n == 1:
        return series[start]
    try:
        from flamo.processor import system as flamo_system  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "FLAMO system.Series is required for multi-module subgraphs."
        ) from exc
    items = list(series._modules.items())[start:end]
    return flamo_system.Series(OrderedDict(items))


def flamo_decompose_for_pr(model: Any) -> FlamoDecompositionForPR:
    """
    Decompose a FLAMO model into the subgraphs needed for poles/residues.

    Expects core with branchA (Series of input_gain, Recursion(feedforward, feedback), output_gain)
    and branchB (direct path). Returns small FLAMO subgraphs (no probing); pass the result
    to :func:`flamo_to_pr` as ``decomposition=...``.
    """
    core = model.get_core() if callable(getattr(model, "get_core", None)) else model
    if not hasattr(core, "branchA") or not hasattr(core, "branchB"):
        raise ValueError(
            "Model core must have branchA and branchB (e.g., from dss_to_flamo)."
        )
    fdn_branch = core.branchA
    direct_branch = core.branchB
    fdn_modules = _as_module_list(fdn_branch)
    recs = [
        m for m in fdn_modules if hasattr(m, "feedforward") and hasattr(m, "feedback")
    ]
    if len(recs) != 1:
        raise ValueError("branchA must contain exactly one Recursion.")
    recursion_module = recs[0]
    rec_idx = fdn_modules.index(recursion_module)
    n_fdn = len(fdn_modules)
    delays = _delays_from_recursion(recursion_module)
    if delays.size == 0:
        raise ValueError("Recursion has no delays (empty delay module).")

    in_subgraph = _series_slice_to_subgraph(fdn_branch, 0, rec_idx)
    out_subgraph = _series_slice_to_subgraph(fdn_branch, rec_idx + 1, n_fdn)
    return FlamoDecompositionForPR(
        recursion_module=recursion_module,
        delays=delays,
        in_subgraph=in_subgraph,
        f_subgraph=recursion_module.feedforward,
        out_subgraph=out_subgraph,
        direct_subgraph=direct_branch,
    )


def _delays_from_recursion(recursion_module: Any) -> np.ndarray:
    """
    Return 1D array of delay lengths in samples from the recursion's feedforward.
    Looks at the delay module in the recursion and sums the number of delays (per line).
    """
    ff = recursion_module.feedforward
    delay_mod = getattr(ff, "delay", ff)
    param = delay_mod.param
    if callable(getattr(delay_mod, "map", None)):
        sec = delay_mod.map(param)
    else:
        sec = param
    samples = delay_mod.s2sample(sec)
    out = np.asarray(samples.detach().cpu().numpy(), dtype=np.float64).ravel()
    return np.asarray(np.round(out), dtype=int)


def _extract_flamo_recursion_probes(
    decomposition: FlamoDecompositionForPR,
) -> _CharacteristicDecomposition:
    """Build characteristic decomposition from pre-decomposed FLAMO subgraphs."""
    rec = decomposition.recursion_module
    return _CharacteristicDecomposition(
        p_probe=rec,
        f_probe=decomposition.f_subgraph,
        in_probe=decomposition.in_subgraph,
        out_probe=decomposition.out_subgraph,
        direct_probe=decomposition.direct_subgraph,
    )


def flamo_extract_pr_decomposition(model: Any) -> dict[str, Any]:
    """
    Extract the H(z)=C P(z)^{-1}B+D probes from a FLAMO model.

    Uses :func:`flamo_decompose_for_pr` then builds probe adapters. Returns a dict
    with keys ``"P"``, ``"F"`` (feedforward), ``"B"`` (input path), ``"C"``, ``"D"``.
    For poles/residues use
    :func:`flamo_to_pr`(model) or :func:`flamo_to_pr`(decomposition=...).
    """
    decomp = flamo_decompose_for_pr(model)
    decomposition = _extract_flamo_recursion_probes(decomp)
    return _decomposition_to_public_dict(decomposition)


def _rcond(mat: np.ndarray) -> float:
    try:
        cond = np.linalg.cond(mat)
    except np.linalg.LinAlgError:
        return 0.0
    if not np.isfinite(cond) or cond == 0:
        return 0.0
    return float(1.0 / cond)


def _rcond_torch(mat: torch.Tensor) -> float:
    """Reciprocal condition number for torch (complex) matrix."""
    try:
        m = mat.resolve_conj()
        cond = torch.linalg.cond(m)
    except Exception:
        return 0.0
    c = float(cond.cpu().numpy())
    if not np.isfinite(c) or c == 0:
        return 0.0
    return float(1.0 / c)


@dataclass
class _FDNLoopFlamo:
    """Loop built from Recursion; P(z)/P(w) via probe_recursion/probe_recursion_w; derivatives built once here."""

    recursion: Any

    def __post_init__(self):
        rec = self.recursion
        z0 = _as_torch_complex_scalar(1.0 + 0j, model=rec)
        p0 = rec.probe_recursion(z0)
        if isinstance(p0, tuple):
            p0 = p0[0]
        if p0.ndim != 2 or p0.shape[0] != p0.shape[1]:
            raise ValueError(f"Recursion P(z) must be square 2-D, got {p0.shape}")
        self.n = p0.shape[0]
        self._device = _infer_model_device(rec)
        self._dtype = _infer_model_complex_dtype(rec)
        self._inverse_newton_step_w_fn = _make_log_det_derivative_w(rec)
        self._get_P_and_dP_dz = _make_P_and_dP_dz(rec)
        self._log_det_derivative_z_fn = _make_log_det_derivative_z(rec)

    def at_z(self, z: complex) -> torch.Tensor:
        """P(z) via Recursion.probe_recursion(z)."""
        out = self.recursion.probe_recursion(z)
        return out.detach()

    def at_w(self, w: complex) -> torch.Tensor:
        """P(w) via Recursion.probe_recursion_w(w)."""
        out = self.recursion.probe_recursion_w(w)
        return out.detach()

    def der(self, z: complex) -> torch.Tensor:
        """dP/dz(z) from built-once JVP callable."""
        return self._get_P_and_dP_dz(z)[1]

    def get_P_and_dP_dz(self, z: complex) -> tuple[torch.Tensor, torch.Tensor]:
        """(P(z), dP/dz(z)) from built-once JVP callable."""
        return self._get_P_and_dP_dz(z)

    def inverse_newton_step_w(self, w: complex) -> torch.Tensor:
        """(d/dw) log det P(w) for Newton refinement."""
        return self._inverse_newton_step_w_fn(w)

    def log_det_derivative_z(self, z: complex) -> torch.Tensor:
        """(d/dz) log det P(z)."""
        return self._log_det_derivative_z_fn(z)

    def log_det_derivative_w(self, w: complex) -> torch.Tensor:
        """(d/dw) log det P(w)."""
        return self._inverse_newton_step_w_fn(w)


def _pole_quality_z(
    poles_z: torch.Tensor,
    loop: _FDNLoopFlamo,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pole quality in z-domain: rcond(P(z)) for each pole z. Returns torch tensor."""
    poles_flat = poles_z.ravel()
    n = poles_flat.shape[0]
    quality = torch.zeros(n, device=device, dtype=torch.float64)
    for i in range(n):
        z = poles_flat[i].item()
        m = loop.at_z(z)
        if m.ndim == 0:
            q = m.abs().item()
        else:
            q = _rcond_torch(m)
        if torch.isfinite(m).all() and m.abs().max().item() > 1e10:
            q = 1e10
        quality[i] = q
    return quality


def _pole_quality_w(
    roots_w: torch.Tensor,
    loop: _FDNLoopFlamo,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Pole quality for roots w: rcond(P(1/w)).
    Uses z-domain when |w| > 1; else w-domain. Returns torch tensor.
    """
    roots_flat = roots_w.ravel()
    n = roots_flat.shape[0]
    quality = torch.zeros(n, device=device, dtype=torch.float64)
    for i in range(n):
        w = roots_flat[i].item()
        if abs(w) > 1:
            z = 1.0 / w
            q = _pole_quality_z(
                torch.tensor([z], device=device, dtype=dtype), loop, device, dtype
            )[0].item()
        else:
            m = loop.at_w(w)
            if m.ndim == 0:
                q = m.abs().item()
            else:
                q = _rcond_torch(m)
            if torch.isfinite(m).all() and m.abs().max().item() > 1e10:
                q = 1e10
        quality[i] = q
    return quality


def _compute_deflation(
    it: int,
    roots_w: torch.Tensor,
    inv_newton_step: torch.Tensor,
    *,
    deflation_type: str,
    number_of_neighbors: int,
    deflation_max_error: float,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, bool]:
    """Deflation term in w-domain. All torch; returns (deflation scalar tensor, is_exact)."""
    pole = roots_w[it]
    n_poles = roots_w.shape[0]
    # Large value so 1/(self-distance) ≈ 0 in deflation sum
    huge = 1.0 / np.finfo(float).eps
    huge_t = torch.tensor(huge + 0.0j, device=device, dtype=dtype)

    if deflation_type == "fullDeflation":
        neighbor_distance = pole - roots_w
        neighbor_distance = neighbor_distance.clone()
        neighbor_distance[it] = huge_t
        deflation = (1.0 / neighbor_distance).sum()
        return deflation, True
    if deflation_type == "noDeflation":
        return torch.tensor(0.0 + 0.0j, device=device, dtype=dtype), False
    if deflation_type != "neighborDeflation":
        raise ValueError(f"Unknown deflation type: {deflation_type}")

    if steps == 1:
        neighbor_deflation = torch.tensor(0.0 + 0.0j, device=device, dtype=dtype)
        factor_nonneighbor = (n_poles - 1) / 2.0
    else:
        n_neigh = int(max(0, min(number_of_neighbors, n_poles - 1)))
        if n_neigh % 2 != 0:
            n_neigh -= 1
        if n_neigh <= 0:
            neighbor_deflation = torch.tensor(0.0 + 0.0j, device=device, dtype=dtype)
            factor_nonneighbor = (n_poles - 1) / 2.0
        else:
            offsets = np.concatenate(
                [np.arange(-n_neigh // 2, 0), np.arange(1, n_neigh // 2 + 1)]
            )
            idx = (it + offsets) % n_poles
            idx_t = torch.as_tensor(idx, device=roots_w.device, dtype=torch.long)
            neighbor_deflation = (1.0 / (pole - roots_w[idx_t])).sum()
            factor_nonneighbor = (n_poles - n_neigh - 1) / 2.0

    equi_deflation = pole.conj() * factor_nonneighbor
    deflation = neighbor_deflation + equi_deflation
    if steps != 1 and (inv_newton_step - deflation).abs().item() < deflation_max_error:
        return _compute_deflation(
            it,
            roots_w,
            inv_newton_step,
            deflation_type="fullDeflation",
            number_of_neighbors=number_of_neighbors,
            deflation_max_error=deflation_max_error,
            steps=steps,
            device=device,
            dtype=dtype,
        )
    return deflation, False


def _refine_pole_positions_w(
    roots_w: torch.Tensor,
    loop: _FDNLoopFlamo,
    *,
    device: torch.device,
    dtype: torch.dtype,
    quality_threshold: float,
    maximum_iterations: int,
    deflation_type: str,
    verbose: bool,
    refinement_tol: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Refine roots in w-domain. Newton step uses loop.inverse_newton_step_w (built once in loop)."""
    roots_w = roots_w.ravel().clone()
    roots_w, _ = _sort_by_torch(roots_w, torch.angle(roots_w))
    n_poles = roots_w.shape[0]

    newton_step_counter = 0
    exact_counter = 0
    record_roots_w: list[torch.Tensor] = [roots_w.clone()]

    number_of_neighbors = int(round(n_poles / 100.0 / 2.0) * 2.0)
    deflation_max_error = 1000.0

    quality = _pole_quality_w(roots_w, loop, device, dtype)
    quality_last = quality.clone()
    current_deflation = deflation_type

    if verbose:
        print(
            f"Ehrlich-Aberth Iteration in w-domain with {n_poles} poles and a maximum of "
            f"{maximum_iterations} iterations"
        )

    for iteration_counter in range(1, maximum_iterations + 1):
        if current_deflation == "neighborDeflation":
            roots_w, sort_ind = _sort_by_torch(roots_w, torch.angle(roots_w))
            quality = quality[sort_ind]
            quality_last = quality_last[sort_ind]

        roots_w_old = roots_w.clone()
        non_converged = (quality > quality_threshold).nonzero(as_tuple=True)[0]

        if non_converged.numel() < n_poles / 10.0:
            current_deflation = "fullDeflation"

        for idx in range(non_converged.numel()):
            it = int(non_converged[idx].item())
            if quality[it] <= quality_threshold:
                continue
            if quality[it] > 10000.0:
                new_w = torch.tensor(
                    np.exp(1j * np.random.uniform(0, 2 * np.pi)),
                    device=device,
                    dtype=dtype,
                )
                roots_w = roots_w.clone()
                roots_w[it] = new_w
                q_new = _pole_quality_w(roots_w[it : it + 1], loop, device, dtype)[0]
                quality = quality.clone()
                quality[it] = q_new
                if verbose:
                    print(
                        f"Pole {it} was at {roots_w_old[it].item():.3f} and set to random "
                        f"value on unit circle: {roots_w[it].item():.3f}"
                    )
                continue

            newton_step_counter += 1
            w_i = roots_w[it].item()
            inv_newton_w = loop.inverse_newton_step_w(w_i).resolve_conj()
            deflation, is_exact = _compute_deflation(
                it,
                roots_w,
                inv_newton_w,
                deflation_type=current_deflation,
                number_of_neighbors=number_of_neighbors,
                deflation_max_error=deflation_max_error,
                steps=iteration_counter,
                device=device,
                dtype=dtype,
            )
            denom = inv_newton_w - deflation
            denom_val = denom.abs().item()
            if not torch.isfinite(denom).all() or denom_val < 1e-20:
                continue
            new_val = roots_w[it] - 1.0 / denom
            roots_w = roots_w.clone()
            roots_w[it] = new_val
            q_new = _pole_quality_w(roots_w[it : it + 1], loop, device, dtype)[0]
            quality = quality.clone()
            quality[it] = q_new
            exact_counter += int(is_exact)

        if verbose:
            record_roots_w.append(roots_w.clone())

        if refinement_tol is not None:
            max_step = (roots_w - roots_w_old).abs().max().item()
            if max_step < refinement_tol:
                if verbose:
                    print(f"Converged (max |Δw| = {max_step:.3e} < {refinement_tol})")
                break
        else:
            max_improvement = (quality_last - quality).abs().max().item()
            if max_improvement < quality_threshold:
                if verbose:
                    print("No further improvement possible")
                break
        if verbose:
            max_improvement = (quality_last - quality).abs().max().item()
            n_nc = non_converged.numel()
            print(
                f"Iter: {iteration_counter}, "
                f"Max Improvement: {max_improvement:.3e}, "
                f"Worst Pole Quality: {quality.max().item():.3e}, "
                f"Number of Non-converged Poles: {n_nc}"
            )
        quality_last = quality.clone()

    if verbose:
        print(f"Number of Exact Deflations: {exact_counter}")
        print(f"Number of Newton Steps: {newton_step_counter}")
        print(f"Number of Poles: {n_poles}")
        print(
            f"Number of Non-converged Poles: {(quality > quality_threshold).sum().item()}"
        )
    meta = {
        "newtonStepCounter": int(newton_step_counter),
        "iterations": int(iteration_counter),
        "exactCounter": int(exact_counter),
        "recordNeighborDeflation": [],
        "recordNewton": [],
        "recordRootsW": np.asarray(
            [_to_numpy(r) for r in record_roots_w], dtype=np.complex128
        ),
    }
    return roots_w, quality, meta


def _dss_to_res_flamo(
    poles: np.ndarray,
    loop: _FDNLoopFlamo,
    decomposition: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Residues from poles; P and dP/dz from loop (built once in _FDNLoopFlamo)."""
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    n_poles = poles.size
    n_in = int(decomposition.in_probe.input_channels)
    n_out = int(decomposition.out_probe.output_channels)
    n = loop.n

    if n_poles == 0:
        warnings.warn(
            "FLAMO pole finder returned no poles; residues are empty.",
            stacklevel=2,
        )
        direct_term = decomposition.direct_probe.probe(1.0 + 0j)
        return (
            np.zeros((0, n_out, n_in), dtype=np.complex128),
            _to_numpy(direct_term),
            np.zeros(0, dtype=np.complex128),
            {
                "right": np.zeros((n, 0), dtype=np.complex128),
                "left": np.zeros((n, 0), dtype=np.complex128),
            },
        )

    p0, _ = loop.get_P_and_dP_dz(poles[0])
    device, dtype = p0.device, p0.dtype

    r_den = torch.zeros(n_poles, device=device, dtype=dtype)
    r_nom = torch.zeros((n_poles, n_out, n_in), device=device, dtype=dtype)
    eig_right = torch.zeros((n, n_poles), device=device, dtype=dtype)
    eig_left = torch.zeros((n, n_poles), device=device, dtype=dtype)

    for it, pole in enumerate(poles):
        p, dp = loop.get_P_and_dP_dz(pole)
        f_at = decomposition.f_probe.probe(pole)
        in_at = decomposition.in_probe.probe(pole)
        b = f_at @ in_at
        c = decomposition.out_probe.probe(pole)

        u, s, vh = torch.linalg.svd(p)
        r = vh.conj().T[:, -1]
        l = u[:, -1]

        denom = torch.vdot(l, (dp @ r).ravel())  # l^H (dP/dz) r
        r_den[it] = denom
        eig_right[:, it] = r
        eig_left[:, it] = l

        cr = c @ r.reshape(-1, 1)
        lh_b = l.conj().reshape(1, -1) @ b
        r_nom[it, :, :] = cr @ lh_b

    with np.errstate(divide="ignore", invalid="ignore"):
        undriven = 1.0 / r_den
    is_multiple = ~torch.isfinite(undriven)
    if is_multiple.any():
        warnings.warn(
            "There are multipoles. The residues are set to zero.", stacklevel=2
        )
        undriven = torch.where(is_multiple, torch.zeros_like(undriven), undriven)

    residues = r_nom / r_den[:, None, None]
    zero = torch.tensor(0.0 + 0.0j, device=device, dtype=dtype)
    residues = torch.where(torch.isfinite(residues), residues, zero)
    direct_term = decomposition.direct_probe.probe(1.0 + 0j)
    # direct_term = 0.0 + 0.0j # TODO: fix later by removing this whole direct term

    return (
        _to_numpy(residues),
        _to_numpy(direct_term),
        _to_numpy(undriven),
        {"right": _to_numpy(eig_right), "left": _to_numpy(eig_left)},
    )


def flamo_to_pr(
    model: Any | None = None,
    *,
    decomposition: FlamoDecompositionForPR | None = None,
    deflation_type: str = "fullDeflation",
    reject_unstable_poles: bool = False,
    quality_threshold: float = 1e-10,
    maximum_iterations: int = 50,
    refinement_tol: float = 1e-12,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Poles/residues from a FLAMO transfer H(z)=C(z)P(z)^{-1}B(z)+D(z).

    Pass either a full FLAMO **model** (decomposition is done via
    :func:`flamo_decompose_for_pr`) or a **decomposition** returned by that
    function. Poles are refined in the w-domain (w = 1/z) then converted to z.

    For DSS matrices (A,B,C,D) use :func:`dss_to_pr_flamo` instead.
    """
    if decomposition is None:
        if model is None:
            raise ValueError("Provide model or decomposition.")
        decomposition = flamo_decompose_for_pr(model)
    delays_arr = decomposition.delays
    decomposition_obj = _extract_flamo_recursion_probes(decomposition)

    loop = _FDNLoopFlamo(recursion=decomposition_obj.p_probe)
    device, dtype = _device_dtype_from_decomposition(decomposition_obj)

    n_poles = int(np.sum(delays_arr))

    # Initialize on unit circle in w-domain (torch), refine in torch
    root_angles = np.linspace(0.0, 2.0 * np.pi, n_poles, endpoint=False)
    roots_w = torch.tensor(
        np.exp(1j * root_angles).astype(np.complex128),
        device=device,
        dtype=dtype,
    )

    roots_w, quality, meta_refine = _refine_pole_positions_w(
        roots_w,
        loop,
        device=device,
        dtype=dtype,
        quality_threshold=float(quality_threshold),
        maximum_iterations=int(maximum_iterations),
        deflation_type=str(deflation_type),
        verbose=bool(verbose),
        refinement_tol=refinement_tol,
    )

    meta_data: dict[str, Any] = dict(meta_refine)
    meta_data["refinedRootsW"] = _to_numpy(roots_w)

    is_stable = roots_w.abs() >= 1.0
    is_converged = quality < float(quality_threshold) * 1000.0
    if reject_unstable_poles:
        is_valid = is_stable & is_converged
    else:
        is_valid = is_converged
    roots_w = roots_w[is_valid]
    quality = quality[is_valid]

    if verbose:
        print(f"Number of Stable Poles: {is_stable.sum().item()}")
        print(f"Number of Converged Poles: {is_converged.sum().item()}")
        print(f"Number of Valid Poles: {is_valid.sum().item()}")

    poles_torch = 1.0 / roots_w
    # Convert to numpy only for scipy.optimize.linear_sum_assignment in reduce_conjugate_pairs
    poles_np = _to_numpy(poles_torch)
    poles, is_conjugate, non_paired = reduce_conjugate_pairs(poles_np, verbose=verbose)
    meta_data["nonPairedPoles"] = non_paired

    residues, direct, undriven, eigenvectors = _dss_to_res_flamo(
        poles, loop, decomposition_obj
    )
    meta_data["undrivenResidues"] = undriven
    meta_data["eigenvectors"] = eigenvectors
    meta_data["decomposition"] = _decomposition_to_public_dict(decomposition_obj)
    meta_data["loop"] = loop

    return residues, poles, direct, is_conjugate, meta_data


def dss_to_pr_flamo(
    delays: ArrayLike,
    A: Any,
    B: Any,
    C: Any,
    D: Any,
    *,
    deflation_type: str = "fullDeflation",
    reject_unstable_poles: bool = False,
    quality_threshold: float = 1e-7,
    maximum_iterations: int = 50,
    refinement_tol: float = 1e-12,
    verbose: bool = True,
    dtype: Any = None,
    Fs: float = 1.0,
    nfft: int = 2**16,
    device: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    DSS -> FLAMO -> PR in one call.

    Builds a FLAMO model with :func:`dss_to_flamo`, then runs :func:`flamo_to_pr`
    on it. For an existing FLAMO model use :func:`flamo_to_pr` directly.
    """
    delays_arr = np.asarray(delays, dtype=int).ravel()
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

    return flamo_to_pr(
        model,
        deflation_type=deflation_type,
        reject_unstable_poles=reject_unstable_poles,
        quality_threshold=quality_threshold,
        maximum_iterations=maximum_iterations,
        refinement_tol=refinement_tol,
        verbose=verbose,
    )
