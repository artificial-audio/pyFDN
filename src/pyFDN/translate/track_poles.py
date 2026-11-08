"""Follow a few poles of a FLAMO loop while its parameters move along a path.

Continuation after Bai & Schlecht, "Continuation method for feedback delay
network modal decomposition", ICASSP 2026: an Euler predictor from the
eigenvalue derivative and a Newton corrector. Here it follows a handful of
dominant poles instead of all of them, and any parameter of the loop can be on
the path (a loop gain, the feedback matrix, filter coefficients, a coupling
FIR). The loop's characteristic matrix ``P(z) = I - F(z) B(z)`` comes from
FLAMO's ``Recursion.probe_recursion``; its derivatives come from autograd.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import numpy as np
import torch
from torch.func import functional_call, jvp

from pyFDN.translate.flamo_to_pr import (
    _count_loop_filter_poles,
    flamo_decompose_for_pr,
)

ParameterPath = Callable[[torch.Tensor], Mapping[str, torch.Tensor]]


class PoleTracks(NamedTuple):
    """Poles followed along a parameter path, as returned by :func:`track_poles`.

    ``t`` holds the accepted path positions, shape ``(T,)``, in tracking order.
    ``poles`` has shape ``(T, K)``; a track that could not be continued is NaN
    from the step after it was lost. ``left`` and ``right`` have shape
    ``(T, K, n)``: the singular vectors of the loop's ``n x n`` characteristic
    matrix at each pole, i.e. how strongly each loop channel carries the mode.
    """

    t: np.ndarray
    poles: np.ndarray
    left: np.ndarray
    right: np.ndarray


class _Probe(torch.nn.Module):
    """``forward(z)`` is the characteristic matrix of one FLAMO ``Recursion``."""

    def __init__(self, recursion: Any) -> None:
        super().__init__()
        self.recursion = recursion

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.recursion.probe_recursion(z)


class _Loop:
    """``P(z, t)``: the loop's characteristic matrix with parameters ``path(t)``."""

    def __init__(
        self, model: Any, path: ParameterPath | None, order: int | None, t: float
    ) -> None:
        if hasattr(model, "probe_recursion"):
            if order is None:
                raise ValueError(
                    "order= is required for a bare Recursion: the degree of "
                    "det P(z), e.g. the sum of its delays (an upper bound is fine)"
                )
            recursion = model
        else:
            decomposition = flamo_decompose_for_pr(model)
            recursion = decomposition.recursion_module
            if order is None:
                order = int(decomposition.delays.sum())
                order += _count_loop_filter_poles(recursion)
        params = dict(recursion.named_parameters())
        if any(p.dtype != torch.float64 for p in params.values()):
            raise ValueError("track poles on a float64 model (dtype=torch.float64)")
        self.device = next(iter(params.values())).device
        self.probe = _Probe(recursion)
        self.path = path
        self.spacing = 2 * np.pi / order  # mean pole spacing on the unit circle
        if path is not None:
            unknown = set(path(self.real(t))) - set(params)
            if unknown:
                raise ValueError(
                    f"path sets unknown parameters {sorted(unknown)}; "
                    f"the loop has {sorted(params)}"
                )

    def cplx(self, z: Any) -> torch.Tensor:
        return torch.as_tensor(z, dtype=torch.complex128, device=self.device)

    def real(self, t: Any) -> torch.Tensor:
        return torch.as_tensor(t, dtype=torch.float64, device=self.device)

    def __call__(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        params = {} if self.path is None else self.path(t)
        return functional_call(
            self.probe, {f"recursion.{k}": v for k, v in params.items()}, (z,)
        )


def _derivative(
    f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``f(x)`` and its derivative along ``x`` (forward mode)."""
    result = jvp(f, (x,), (torch.ones_like(x),))
    return result[0], result[1]


def _correct(
    loop: _Loop, z: complex, t: float, max_newton: int, tol: float = 1e-12
) -> tuple[complex, torch.Tensor, torch.Tensor, bool]:
    """SVD-Newton on ``P(z, t)``: ``dz = -s_min / (l^H P_z r)``."""
    t_ = loop.real(t)
    for _ in range(max_newton):
        P, P_z = _derivative(lambda zz: loop(zz, t_), loop.cplx(z))
        U, s, Vh = torch.linalg.svd(P)
        left, right = U[:, -1], Vh[-1].conj().resolve_conj()
        slope = complex(torch.vdot(left, P_z @ right))
        if slope == 0 or not np.isfinite(slope):
            break
        dz = -float(s[-1]) / slope
        z += dz
        if abs(dz) <= tol * max(1.0, abs(z)):
            return z, left, right, True
    return z, left, right, False


def _velocity(
    loop: _Loop, z: complex, left: torch.Tensor, right: torch.Tensor, t: float
) -> complex:
    """``dz/dt = -(l^H P_t r) / (l^H P_z r)`` (eigenvalue derivative)."""
    z_, t_ = loop.cplx(z), loop.real(t)
    _, P_z = _derivative(lambda zz: loop(zz, t_), z_)
    _, P_t = _derivative(lambda tt: loop(z_, tt), t_)
    num = complex(torch.vdot(left, P_t @ right))
    return -num / complex(torch.vdot(left, P_z @ right))


def _scan(
    loop: _Loop, t: float, radius: float, nfft: int, count: int, max_newton: int
) -> list[complex]:
    """Refine the ``count`` deepest dips of ``s_min(P)`` on ``|z| = radius``."""
    w = 2 * np.pi * np.arange(nfft // 2 + 1) / nfft
    z = loop.cplx(radius * np.exp(1j * w))
    t_ = loop.real(t)
    batched = torch.vmap(lambda zz: loop(zz, t_))
    s_min = torch.cat(
        [torch.linalg.svdvals(batched(chunk))[:, -1] for chunk in torch.split(z, 4096)]
    )
    s = s_min.cpu().numpy()
    padded = np.r_[s[1], s, s[-2]]  # mirror: dips at 0 and pi count too
    mid = padded[1:-1]
    dips = np.flatnonzero((mid < padded[:-2]) & (mid <= padded[2:]))
    poles = []
    for i in dips[np.argsort(s[dips])[:count]]:
        pole, _, _, ok = _correct(loop, complex(z[i]), t, max_newton)
        if ok:
            poles.append(pole)
    return poles


def _distinct_upper(poles: list[complex], tol: float) -> list[complex]:
    """One of each pole with Im >= 0, largest ``|z|`` first."""
    out: list[complex] = []
    for p in sorted(poles, key=abs, reverse=True):
        if p.imag >= -tol and all(abs(p - q) > tol for q in out):
            out.append(p)
    return out


def seed_poles(
    model: Any,
    num_poles: int,
    *,
    path: ParameterPath | None = None,
    t: float = 0.0,
    order: int | None = None,
    radius: float | None = None,
    nfft: int | None = None,
    max_newton: int = 8,
) -> np.ndarray:
    """Find the ``num_poles`` largest-magnitude poles of a FLAMO loop.

    Scans the smallest singular value of the characteristic matrix on a circle
    ``|z| = radius`` just outside the poles, refines the deepest dips with
    Newton, and ranks the converged poles by ``|z|``. Without a ``radius`` the
    scan starts a quarter pole spacing outside the unit circle and repeats once
    or twice just outside the largest pole found, so poles well inside the
    circle (a lossy loop) or outside it (an unstable one) are ranked correctly.

    Parameters
    ----------
    model : FLAMO model or ``Recursion``
        A float64 model with one non-nested ``Recursion`` (as from
        :func:`pyFDN.dss_to_flamo`), or the ``Recursion`` itself.
    num_poles : int
        Number of poles to return.
    path : callable, optional
        Parameter path as in :func:`track_poles`; the loop is evaluated at
        ``path(t)``. Default: the model's current parameters.
    t : float
        Path position to seed at.
    order : int, optional
        Degree of ``det P(z)`` (an upper bound is fine); sets the scan
        resolution. Inferred from the delays and loop filters of a model;
        required for a bare ``Recursion``.
    radius : float, optional
        Scan on this circle only.
    nfft : int, optional
        Scan points on the full circle. Default: a power of two ``>= 8 * order``.
    max_newton : int
        Newton iterations per candidate.

    Returns
    -------
    np.ndarray
        Complex poles with ``Im z >= 0``, largest ``|z|`` first. Conjugates of
        a real-coefficient loop are implied.
    """
    loop = _Loop(model, path, order, t)
    order = round(2 * np.pi / loop.spacing)
    if nfft is None:
        nfft = max(2**10, 1 << int(np.ceil(np.log2(8 * order))))
    tol = 1e-6 * loop.spacing
    auto = radius is None
    scan_radius = 1 + loop.spacing / 4 if radius is None else radius
    found: list[complex] = []
    for _ in range(3):
        found = _distinct_upper(
            found + _scan(loop, t, scan_radius, nfft, 4 * num_poles, max_newton), tol
        )
        if not auto or not found:
            break
        target = abs(found[0]) + loop.spacing / 4
        if abs(target - scan_radius) <= loop.spacing / 4:
            break
        scan_radius = target
    return np.array(found[:num_poles], dtype=np.complex128)


def track_poles(
    model: Any,
    path: ParameterPath,
    poles: Any,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    order: int | None = None,
    max_step: float = 0.1,
    min_step: float = 1e-4,
    max_newton: int = 8,
) -> PoleTracks:
    """Follow poles of a FLAMO loop while its parameters move along a path.

    A predictor-corrector continuation: an Euler step in ``log z`` from the
    eigenvalue derivative, then SVD-Newton at the new path position. A step is
    accepted only if every track converges, moves less than a quarter of the
    mean pole spacing ``2 pi / order`` from its prediction, keeps its singular
    vector (``|r_new^H r_old| >= 0.9``, so it has not jumped to a neighbour)
    and stays apart from the other tracks; otherwise the step is halved. A
    track still failing at ``min_step`` is marked lost (NaN) and the others go
    on.

    Parameters
    ----------
    model : FLAMO model or ``Recursion``
        A float64 model with one non-nested ``Recursion`` (as from
        :func:`pyFDN.dss_to_flamo`), or the ``Recursion`` itself. Nested loops
        put the inner loop's poles into ``det P`` and are not supported.
    path : callable
        ``path(t)`` maps a 0-d float64 tensor ``t`` to new values for some of
        the ``Recursion``'s parameters, by their ``named_parameters()`` names
        (stored values, before FLAMO's ``map``). Build them with torch
        operations so the derivative in ``t`` flows. For a gain, step in dB:
        ``lambda t: {"feedback.param": g0 * (g1 / g0) ** t * A}``.
    poles : array_like
        Poles at ``t0``, e.g. from :func:`seed_poles`.
    t0, t1 : float
        Start and end of the path; ``t1 < t0`` tracks backwards.
    order : int, optional
        Degree of ``det P(z)``; see :func:`seed_poles`.
    max_step, min_step : float
        Bounds on the step in ``t``.
    max_newton : int
        Newton iterations allowed per step.

    Returns
    -------
    PoleTracks
        Accepted path positions with the poles and their singular vectors.
    """
    loop = _Loop(model, path, order, t0)
    start = np.atleast_1d(np.asarray(poles, dtype=np.complex128))
    refined = [_correct(loop, complex(p), t0, max_newton) for p in start]
    not_poles = [k for k, (*_, ok) in enumerate(refined) if not ok]
    if not_poles:
        raise ValueError(f"poles {not_poles} are not poles of the loop at t0={t0}")
    state = {k: (z, left, right) for k, (z, left, right, _) in enumerate(refined)}
    size = refined[0][1].shape[0] if refined else 0

    def row() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = np.full(len(start), np.nan, dtype=np.complex128)
        left = np.full((len(start), size), np.nan, dtype=np.complex128)
        right = left.copy()
        for k, (zk, lk, rk) in state.items():
            z[k], left[k], right[k] = zk, lk.cpu().numpy(), rk.cpu().numpy()
        return z, left, right

    direction = 1.0 if t1 >= t0 else -1.0
    t, h = t0, max_step / 2
    ts, rows = [t0], [row()]
    while state and direction * (t1 - t) > 1e-12:
        step = direction * min(h, abs(t1 - t))
        moved, failed = {}, set()
        for k, (z, left, right) in state.items():
            guess = z * np.exp(step * _velocity(loop, z, left, right, t) / z)
            zc, lc, rc, ok = _correct(loop, guess, t + step, max_newton)
            same_mode = abs(complex(torch.vdot(right, rc))) >= 0.9
            if ok and same_mode and abs(zc - guess) <= loop.spacing / 4:
                moved[k] = (zc, lc, rc)
            else:
                failed.add(k)
        keys = list(moved)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if abs(moved[a][0] - moved[b][0]) < 1e-6 * loop.spacing:
                    failed |= {a, b}
        if failed:
            if h / 2 >= min_step:
                h /= 2
            else:
                state = {k: v for k, v in state.items() if k not in failed}
            continue
        state, t = moved, t + step
        ts.append(t)
        rows.append(row())
        h = min(1.5 * h, max_step)
    poles_t, left_t, right_t = (np.stack(x) for x in zip(*rows, strict=True))
    return PoleTracks(np.array(ts), poles_t, left_t, right_t)


__all__ = ["PoleTracks", "seed_poles", "track_poles"]
