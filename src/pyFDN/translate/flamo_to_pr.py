"""Modal decomposition for a FLAMO FDN model via Ehrlich-Aberth iteration.

Poles of ``H(z) = C(z)P(z)^{-1}B(z) + D(z)`` are found in the w = 1/z domain,
polished via SVD null-vector Newton, then converted to z.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from pyFDN.auxiliary.poles import reduce_conjugate_pairs

# ───────────────────────────────────────────────────────────────────────────
# Small infrastructure helpers
# ───────────────────────────────────────────────────────────────────────────


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


def _sort_by_torch(
    a: torch.Tensor, key: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort tensor a by key (by angle for complex); return (a_sorted, indices)."""
    key_np = _to_numpy(key)
    ind = np.argsort(key_np)
    ind_t = torch.as_tensor(ind, device=a.device, dtype=torch.long)
    return a[ind_t], ind_t


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


# ───────────────────────────────────────────────────────────────────────────
# Derivative builders (used by _FDNLoopFlamo)
# ───────────────────────────────────────────────────────────────────────────


def _make_log_det_derivative_w(recursion: Any) -> Callable[[complex], torch.Tensor]:
    """Build (d/dw) log det P(w) once using grad; uses recursion.probe_recursion_w(w)."""

    def d_log_det_w(w: complex) -> torch.Tensor:
        w_var = _as_torch_complex_scalar(w, model=recursion)
        w_var = w_var.detach().clone().requires_grad_(True)
        P_w = recursion.probe_recursion_w(w_var)
        y = torch.logdet(P_w)
        (g,) = torch.autograd.grad(y, w_var, torch.ones_like(y))
        return g.conj().detach()

    return d_log_det_w


def _make_P_and_dP_dz(
    recursion: Any,
) -> Callable[[complex], tuple[torch.Tensor, torch.Tensor]]:
    """Build (P(z), dP/dz(z)) once using JVP; uses recursion.probe_recursion(z)."""

    def get_P_and_dP_dz(z: complex) -> tuple[torch.Tensor, torch.Tensor]:
        z_var = _as_torch_complex_scalar(z, model=recursion)
        dz = torch.ones_like(z_var)
        P, dP_dz = torch.autograd.functional.jvp(
            recursion.probe_recursion,
            (z_var,),
            (dz,),  # type: ignore[no-untyped-call]
        )
        return P.detach(), dP_dz.detach()

    return get_P_and_dP_dz


# ───────────────────────────────────────────────────────────────────────────
# FLAMO model decomposition (public)
# ───────────────────────────────────────────────────────────────────────────


def _decomposition_to_public_dict(
    decomposition: FlamoDecompositionForPR,
) -> dict[str, Any]:
    """Public dict: P, F (feedforward), B (input path), C, D probes. H = C P^{-1} F B + D."""
    return {
        "P": decomposition.recursion_module,
        "F": decomposition.f_subgraph,
        "B": decomposition.in_subgraph,
        "C": decomposition.out_subgraph,
        "D": decomposition.direct_subgraph,
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
        from flamo.processor import system as flamo_system
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

    Supported architecture (see :func:`flamo_to_pr` for the full contract):
    branchA must contain exactly **one, non-nested** ``Recursion``, and the
    delay lines must live in that recursion's **feedforward** path. SOS-format
    filters (incl. biquad/SVF cascades) anywhere in the feedforward/feedback are
    fine; a ``Recursion`` nested inside another loop is **not** supported.
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


def _iter_leaf_modules(node: Any) -> Iterator[Any]:
    """Yield the leaf modules of a FLAMO subgraph, descending into Series-like
    containers. Non-iterable modules (Gain, Delay, SOSFilter, …) are leaves."""
    try:
        children = list(node)
    except TypeError:
        children = []
    if not children:
        yield node
        return
    for child in children:
        yield from _iter_leaf_modules(child)


def _sos_section_order(module: Any) -> int:
    """Order (pole count) an SOS-format filter contributes to the loop's
    characteristic polynomial: the rational order summed over sections and
    channels. Returns 0 for non-SOS modules (gains, delays, FIR), which add no
    poles of their own.

    Recognizes any module exposing ``n_sections`` and a ``map(param)`` yielding
    second-order-section coefficients of shape ``(n_sections, 6, *channels)`` with
    rows ``[b0, b1, b2, a0, a1, a2]`` — how SOS filters (and biquad/SVF cascades
    expressed as SOS) store coefficients in FLAMO.

    Per section/channel the order is the larger of the numerator and denominator
    degrees: 2 if ``a2`` or ``b2`` is non-zero, else 1 if ``a1`` or ``b1`` is
    non-zero, else 0.
    """
    n_sec = getattr(module, "n_sections", None)
    mapping = getattr(module, "map", None)
    param = getattr(module, "param", None)
    if n_sec is None or not callable(mapping) or param is None:
        return 0
    try:
        mapped = mapping(param).detach().cpu().numpy()
    except Exception:
        return 0
    if mapped.ndim < 2 or mapped.shape[0] != int(n_sec) or mapped.shape[1] != 6:
        return 0
    # (sections, [b0,b1,b2,a0,a1,a2], chan)
    c = mapped.reshape(int(n_sec), 6, -1)
    tol = 1e-12
    deg2 = (np.abs(c[:, 2, :]) > tol) | (np.abs(c[:, 5, :]) > tol)  # b2 or a2
    deg1 = (np.abs(c[:, 1, :]) > tol) | (np.abs(c[:, 4, :]) > tol)  # b1 or a1
    order_per = np.where(deg2, 2, np.where(deg1, 1, 0))
    return int(order_per.sum())


def _count_loop_filter_poles(recursion_module: Any) -> int:
    """Extra poles contributed by IIR filters in the recursion's feedforward and
    feedback paths, beyond the pure delays (which are counted via the delays).

    Walks Series containers; nested ``Recursion`` modules are treated as opaque
    leaves and contribute 0 (nested loops are unsupported — see
    :func:`flamo_to_pr`)."""
    total = 0
    for branch in (recursion_module.feedforward, recursion_module.feedback):
        for leaf in _iter_leaf_modules(branch):
            total += _sos_section_order(leaf)
    return total


def flamo_extract_pr_decomposition(model: Any) -> dict[str, Any]:
    """
    Extract the H(z)=C P(z)^{-1}B+D probes from a FLAMO model.

    Returns a dict with keys ``"P"``, ``"F"`` (feedforward), ``"B"`` (input
    path), ``"C"``, ``"D"``. For poles/residues use :func:`flamo_to_pr`.
    """
    return _decomposition_to_public_dict(flamo_decompose_for_pr(model))


# ───────────────────────────────────────────────────────────────────────────
# FDN loop (P(z), dP/dz, Newton step) — used by EAI and SVD refinement
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class _FDNLoopFlamo:
    """Loop built from Recursion; P(z)/P(w) via probe_recursion/probe_recursion_w; derivatives built once here."""

    recursion: Any

    def __post_init__(self) -> None:
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

    def at_z(self, z: complex) -> torch.Tensor:
        """P(z) via Recursion.probe_recursion(z)."""
        out = self.recursion.probe_recursion(z)
        return out.detach()

    def at_w(self, w: complex) -> torch.Tensor:
        """P(w) via Recursion.probe_recursion_w(w)."""
        out = self.recursion.probe_recursion_w(w)
        return out.detach()

    def get_P_and_dP_dz(self, z: complex) -> tuple[torch.Tensor, torch.Tensor]:
        """(P(z), dP/dz(z)) from built-once JVP callable."""
        return self._get_P_and_dP_dz(z)

    def inverse_newton_step_w(self, w: complex) -> torch.Tensor:
        """(d/dw) log det P(w) for Newton refinement."""
        return self._inverse_newton_step_w_fn(w)


# ───────────────────────────────────────────────────────────────────────────
# Ehrlich-Aberth iteration (pole quality, deflation, refinement loop)
# ───────────────────────────────────────────────────────────────────────────


def _matrix_quality(m: torch.Tensor) -> float:
    """rcond(P) as a pole-quality score; 1e10 sentinel for blown-up matrices.

    A non-finite ``P`` (e.g. a root that ran off to ``w → ∞`` / ``z → 0`` where
    the loop's ``z^{-m}`` terms overflow) must score *worst*, not best — otherwise
    ``_rcond_torch`` returns 0 for it and the runaway is mistaken for a perfectly
    converged pole and survives into residue extraction.
    """
    if not torch.isfinite(m).all():
        return 1e10
    q = m.abs().item() if m.ndim == 0 else _rcond_torch(m)
    if m.abs().max().item() > 1e10:
        q = 1e10
    return q


def _pole_quality_w(
    roots_w: torch.Tensor,
    loop: _FDNLoopFlamo,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pole quality rcond(P(1/w)) per root. Probes in z when |w|>1 (pole inside
    the unit disk, where P is better conditioned), else in w."""
    roots_flat = roots_w.ravel()
    quality = torch.zeros(roots_flat.shape[0], device=device, dtype=torch.float64)
    for i in range(roots_flat.shape[0]):
        w = roots_flat[i].item()
        m = loop.at_z(1.0 / w) if abs(w) > 1 else loop.at_w(w)
        quality[i] = _matrix_quality(m)
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
        "recordRootsW": np.asarray(
            [_to_numpy(r) for r in record_roots_w], dtype=np.complex128
        ),
    }
    return roots_w, quality, meta


def _refine_pole_via_svd(
    z_init: complex,
    loop: _FDNLoopFlamo,
    *,
    max_iters: int = 5,
    tol: float = 1e-14,
) -> complex:
    """Refine a single approximate pole via SVD null-vector Newton:
    ``dz = -sigma_min / (l^H · dP/dz · r)``.
    """
    z = complex(z_init)
    for _ in range(max_iters):
        P, dP = loop.get_P_and_dP_dz(z)
        if not (torch.isfinite(P).all() and torch.isfinite(dP).all()):
            break
        # Scale P and dP together before the SVD: poles far from the unit circle
        # (|z| << 1 makes the loop's z^{-m} terms huge) otherwise yield an
        # ill-conditioned matrix that fails to decompose. Null vectors are
        # scale-invariant and dz cancels the shared scale, so the step is unchanged.
        scale = P.abs().max()
        if scale > 0 and torch.isfinite(scale):
            P, dP = P / scale, dP / scale
        try:
            u, s, vh = torch.linalg.svd(P)
        except Exception:
            break
        sigma_min = float(s[-1].cpu().item())
        r = vh.conj().T[:, -1]  # right singular vector for sigma_min
        l = u[:, -1]  # left singular vector for sigma_min
        denom = torch.vdot(l, dP @ r)  # l^H (dP/dz) r
        denom_c = complex(denom.resolve_conj().cpu().numpy())
        if abs(denom_c) < 1e-30 or not np.isfinite(denom_c):
            break
        dz = -sigma_min / denom_c
        z = z + dz
        if abs(dz) < tol:
            break
    return z


def _conjugate_symmetrize(poles: np.ndarray, *, tol: float = 1e-6) -> np.ndarray:
    """Enforce conjugate-pair symmetry on a pole set: average each pole with
    the conjugate of its nearest match and snap near-real poles to the real axis.
    """
    poles = np.asarray(poles, dtype=np.complex128).copy()
    n = poles.size
    out = poles.copy()
    used = np.zeros(n, dtype=bool)
    for i in range(n):
        if used[i]:
            continue
        scale = max(abs(poles[i]), 1.0)
        # Near-real: snap imaginary part to zero
        if abs(poles[i].imag) < tol * scale:
            out[i] = complex(poles[i].real, 0.0)
            used[i] = True
            continue
        # Find nearest unused candidate for conjugate match
        target = poles[i].conjugate()
        mask = ~used.copy()
        mask[i] = False
        if not mask.any():
            used[i] = True
            continue
        idx = np.where(mask)[0]
        dists = np.abs(poles[idx] - target)
        k = int(np.argmin(dists))
        j = int(idx[k])
        if dists[k] < tol * scale:
            avg = 0.5 * (poles[i] + poles[j].conjugate())
            out[i] = avg
            out[j] = avg.conjugate()
            used[i] = True
            used[j] = True
        else:
            used[i] = True
    return out


# ───────────────────────────────────────────────────────────────────────────
# Residue extractor
# ───────────────────────────────────────────────────────────────────────────


def _dss_to_res_flamo(
    poles: np.ndarray,
    loop: _FDNLoopFlamo,
    decomposition: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Residues from poles using FLAMO probes for B, C, D."""
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    n_poles = poles.size
    n_in = int(decomposition.in_subgraph.input_channels)
    n_out = int(decomposition.out_subgraph.output_channels)
    n = loop.n

    p0, _ = loop.get_P_and_dP_dz(poles[0])
    device, dtype = p0.device, p0.dtype

    r_den = torch.zeros(n_poles, device=device, dtype=dtype)
    r_nom = torch.zeros((n_poles, n_out, n_in), device=device, dtype=dtype)
    eig_right = torch.zeros((n, n_poles), device=device, dtype=dtype)
    eig_left = torch.zeros((n, n_poles), device=device, dtype=dtype)

    for it, pole in enumerate(poles):
        p, dp = loop.get_P_and_dP_dz(pole)
        f_at = decomposition.f_subgraph.probe(pole)
        in_at = decomposition.in_subgraph.probe(pole)
        b = f_at @ in_at
        c = decomposition.out_subgraph.probe(pole)

        # Scale p before the SVD (null vectors are scale-invariant) so poles far
        # from the unit circle stay well-conditioned; dp is left unscaled for denom.
        scale = p.abs().max()
        p_svd = p / scale if (scale > 0 and torch.isfinite(scale)) else p
        u, s, vh = torch.linalg.svd(p_svd)
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
    direct_term = decomposition.direct_subgraph.probe(1.0 + 0j)

    return (
        _to_numpy(residues),
        _to_numpy(direct_term),
        _to_numpy(undriven),
        {"right": _to_numpy(eig_right), "left": _to_numpy(eig_left)},
    )


# ───────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────


def flamo_to_pr(
    model: Any | None = None,
    *,
    decomposition: FlamoDecompositionForPR | None = None,
    deflation_type: str = "fullDeflation",
    reject_unstable_poles: bool = False,
    quality_threshold: float = 1e-10,
    maximum_iterations: int = 50,
    refinement_tol: float = 1e-12,
    svd_refine: bool = True,
    symmetrize: bool = True,
    verbose: bool = True,
    num_poles: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Poles/residues from a FLAMO transfer ``H(z) = C(z)P(z)^{-1}B(z) + D(z)``.

    Pass either a FLAMO ``model`` or a ``decomposition`` from
    :func:`flamo_decompose_for_pr`.

    Supported architecture
    ----------------------
    The model must reduce to a **single, non-nested** ``Recursion`` (the canonical
    FDN: ``Parallel(branchA=Series(in_gain, Recursion, out_gain), branchB=direct)``,
    as produced by :func:`pyFDN.translate.dss_to_flamo.dss_to_flamo`). Two
    constraints are assumed and **not** auto-detected:

    - **Delays live in the recursion's feedforward path.** Delay lengths are read
      from ``feedforward`` to size the pole search.
    - **No nested recursion.** A ``Recursion`` placed inside another loop's
      feedforward/feedback (e.g. an allpass realized as its own sub-FDN) is *not*
      supported: its internal poles are folded into ``F(z)`` by ``probe`` but are
      not roots of the outer ``det P`` and would be silently dropped.

    SOS-format filters (one-pole/biquad/SVF cascades stored as second-order
    sections) anywhere in the feedforward or feedback are supported: their poles
    are added to the pole count via :func:`_count_loop_filter_poles` so the
    Ehrlich-Aberth search is sized correctly.

    Parameters
    ----------
    num_poles : int or None, default None
        Override the auto-detected pole count (``sum(delays)`` plus IIR poles from
        loop filters). When given, this exact value sizes the Ehrlich-Aberth root
        search instead of the detected order. Use it when the loop order is known
        independently (e.g. the characteristic-polynomial degree) or when filters
        sit where :func:`_count_loop_filter_poles` cannot see them.
    svd_refine : bool, default True
        Run a per-pole SVD null-vector Newton step after the Ehrlich-Aberth loop.
    symmetrize : bool, default True
        Enforce exact conjugate-pair symmetry on the refined pole set before
        :func:`reduce_conjugate_pairs`.
    """
    if decomposition is None:
        if model is None:
            raise ValueError("Provide model or decomposition.")
        decomposition = flamo_decompose_for_pr(model)
    delays_arr = decomposition.delays

    rec = decomposition.recursion_module
    loop = _FDNLoopFlamo(recursion=rec)
    device = _infer_model_device(rec)
    dtype = _infer_model_complex_dtype(rec)

    # Pole count = delay-line poles (sum of delays) + IIR poles from any SOS-format
    # filters in the loop. Pure-delay loops keep the original sum(delays) count.
    # A caller-supplied ``num_poles`` overrides this auto-detected order entirely
    # (e.g. when filters live where the walker can't see them, or to match a known
    # characteristic-polynomial degree).
    n_delay_poles = int(np.sum(delays_arr))
    n_filter_poles = _count_loop_filter_poles(rec)
    n_poles = n_delay_poles + n_filter_poles
    if num_poles is not None:
        if int(num_poles) <= 0:
            raise ValueError(f"num_poles must be a positive integer, got {num_poles}.")
        n_poles = int(num_poles)
        if verbose:
            print(f"Overriding detected pole count: num_poles={n_poles}")

    # Seed equi-spaced on the unit circle in the w = 1/z domain. With IIR filters
    # in the loop the poles spread inward (off the unit circle) and plain
    # Ehrlich-Aberth tends to leave one root homeless — it runs off to w → ∞ while
    # two seeds collide on a single pole. Over-provisioning the seed count gives the
    # iteration enough roots that every true pole is covered; surplus roots diverge
    # and are dropped by the convergence filter below (deflation keeps them from
    # duplicating real poles). Pure-delay loops keep the exact original seeding.
    n_extra = max(0, n_poles - n_delay_poles)
    n_seed = n_poles
    if n_extra > 0:
        n_seed = n_poles + max(8, 2 * n_extra)
    root_angles = np.linspace(0.0, 2.0 * np.pi, n_seed, endpoint=False)
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

    if svd_refine:
        poles_np = np.array(
            [_refine_pole_via_svd(complex(z), loop) for z in poles_np],
            dtype=np.complex128,
        )
        meta_data["polesBeforeSymmetrize"] = poles_np.copy()

    if symmetrize:
        sym_tol = max(float(quality_threshold) * 1000.0, 1e-9)
        poles_np = _conjugate_symmetrize(poles_np, tol=sym_tol)

    poles, is_conjugate, non_paired = reduce_conjugate_pairs(poles_np, verbose=verbose)
    meta_data["nonPairedPoles"] = non_paired

    residues, direct, undriven, eigenvectors = _dss_to_res_flamo(
        poles, loop, decomposition
    )
    meta_data["undrivenResidues"] = undriven
    meta_data["eigenvectors"] = eigenvectors
    meta_data["decomposition"] = _decomposition_to_public_dict(decomposition)
    meta_data["loop"] = loop

    return residues, poles, direct, is_conjugate, meta_data
