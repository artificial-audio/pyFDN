"""Direct FLAMO/autograd modal decomposition (dss2pr_direct-style).

This module is intentionally independent of ZFilter classes. It uses only
FLAMO graph probing (autograd backend) and numeric helpers.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.flamo_runtime_probe import probe_flamo_runtime
from pyFDN.auxiliary.math import general_char_poly
from pyFDN.translate.dss_to_ss import dss_to_ss


class _NumericMatrixProbe:
    """Constant numeric matrix probe adapter with at/der methods."""

    def __init__(self, matrix: ArrayLike):
        mat = np.asarray(matrix, dtype=np.complex128)
        if mat.ndim == 0:
            mat = mat.reshape(1, 1)
        if mat.ndim != 2:
            raise ValueError(f"Expected 2-D matrix, got shape {mat.shape}")
        self._mat = mat
        self.output_channels, self.input_channels = mat.shape

    def at(self, z: complex) -> np.ndarray:
        return self._mat

    def der(self, z: complex) -> np.ndarray:
        return np.zeros_like(self._mat)

    @property
    def matrix(self) -> np.ndarray:
        return self._mat


class _FlamoGraphProbe:
    """FLAMO graph probe adapter backed by autograd z probing."""

    def __init__(self, model: Any):
        self.model = model
        h0 = np.asarray(
            probe_flamo_runtime(model, 1.0 + 0j, derivative=False),
            dtype=np.complex128,
        )
        if h0.ndim != 2:
            raise ValueError(f"Graph probe at scalar z must return 2-D matrix, got {h0.shape}")
        self.output_channels, self.input_channels = h0.shape

    def at(self, z: complex) -> np.ndarray:
        return np.asarray(
            probe_flamo_runtime(self.model, z, derivative=False),
            dtype=np.complex128,
        )

    def der(self, z: complex) -> np.ndarray:
        _, dh = probe_flamo_runtime(self.model, z, derivative=True)
        return np.asarray(dh, dtype=np.complex128)


class _PassthroughProbe:
    """Adapter for objects that already expose at/der."""

    def __init__(self, obj: Any):
        self.obj = obj
        out_ch = getattr(obj, "output_channels", None)
        in_ch = getattr(obj, "input_channels", None)
        if out_ch is None or in_ch is None:
            val = np.asarray(obj.at(1.0 + 0j), dtype=np.complex128)
            if val.ndim == 1:
                val = np.diag(val)
            if val.ndim != 2:
                raise ValueError("Passthrough probe requires 2-D .at(z) output")
            out_ch, in_ch = val.shape
        self.output_channels = int(out_ch)
        self.input_channels = int(in_ch)

    def at(self, z: complex) -> np.ndarray:
        val = np.asarray(self.obj.at(z), dtype=np.complex128)
        if val.ndim == 1:
            val = np.diag(val)
        return val

    def der(self, z: complex) -> np.ndarray:
        val = np.asarray(self.obj.der(z), dtype=np.complex128)
        if val.ndim == 1:
            val = np.diag(val)
        return val


def _to_probe(
    value: Any, *, name: str
) -> _NumericMatrixProbe | _FlamoGraphProbe | _PassthroughProbe:
    if isinstance(value, (np.ndarray, list, tuple)):
        return _NumericMatrixProbe(value)
    if hasattr(value, "at") and hasattr(value, "der"):
        return _PassthroughProbe(value)
    return _FlamoGraphProbe(value)


def _adjugate(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.complex128)
    m, n = arr.shape
    if m != n:
        raise ValueError("Adjugate expects a square matrix")
    if n < 2:
        return np.ones((1, 1), dtype=np.complex128)
    u, s, vh = np.linalg.svd(arr, full_matrices=True)
    v = vh.conj().T
    s_ex = np.ones(n, dtype=np.complex128)
    for i in range(n):
        s_ex[i] = np.prod(np.delete(s, i)) if n > 1 else 1.0
    det_uv = np.linalg.det(u @ v.conj().T)
    return det_uv * ((v * s_ex.reshape(1, -1)) @ u.conj().T)


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


@dataclass
class _SimpleLoop:
    delays: np.ndarray
    feedback: _NumericMatrixProbe | _FlamoGraphProbe

    def __post_init__(self):
        self.delays = np.asarray(self.delays, dtype=np.float64).ravel()
        self.n = int(self.delays.size)
        if self.feedback.input_channels != self.n or self.feedback.output_channels != self.n:
            raise ValueError(
                "Feedback dimensions must match number of delays: "
                f"expected ({self.n},{self.n}), got ({self.feedback.output_channels},{self.feedback.input_channels})"
            )

    def _delay_inv_at(self, z: complex) -> np.ndarray:
        return np.diag(np.power(z, self.delays)).astype(np.complex128)

    def _delay_inv_der(self, z: complex) -> np.ndarray:
        return np.diag(self.delays * np.power(z, self.delays - 1.0)).astype(np.complex128)

    def at(self, z: complex) -> np.ndarray:
        return self._delay_inv_at(z) - self.feedback.at(z)

    def der(self, z: complex) -> np.ndarray:
        return self._delay_inv_der(z) - self.feedback.der(z)


def _coerce_static_feedback_matrix(
    feedback: _NumericMatrixProbe | _FlamoGraphProbe,
    *,
    tol: float = 1e-10,
) -> np.ndarray:
    if isinstance(feedback, _NumericMatrixProbe):
        return feedback.matrix.astype(np.complex128)

    z1 = 0.73 + 0.19j
    z2 = 1.11 - 0.27j
    a1 = feedback.at(z1)
    a2 = feedback.at(z2)
    if not np.allclose(a1, a2, rtol=tol, atol=tol):
        raise ValueError(
            "dss_to_pr_flamo_direct currently requires a z-invariant feedback graph "
            "for direct pole extraction (eig/roots)."
        )
    return np.asarray(a1, dtype=np.complex128)


def _dss_to_res_flamo(
    poles: np.ndarray,
    loop: _SimpleLoop,
    b_probe: _NumericMatrixProbe | _FlamoGraphProbe,
    c_probe: _NumericMatrixProbe | _FlamoGraphProbe,
    d_term: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    n_poles = poles.size
    n = loop.n

    b0 = np.asarray(b_probe.at(1.0 + 0j), dtype=np.complex128)
    c0 = np.asarray(c_probe.at(1.0 + 0j), dtype=np.complex128)
    n_in = b0.shape[1]
    n_out = c0.shape[0]

    r_den = np.zeros(n_poles, dtype=np.complex128)
    for it, pole in enumerate(poles):
        r_den[it] = np.trace(_adjugate(loop.at(pole)) @ loop.der(pole))

    with np.errstate(divide="ignore", invalid="ignore"):
        undriven = 1.0 / r_den
    is_multiple = ~np.isfinite(undriven)
    if np.any(is_multiple):
        warnings.warn("There are multipoles. The residues are set to zero.", stacklevel=2)
        undriven[is_multiple] = 0.0
        r_den[is_multiple] = np.inf

    r_nom = np.zeros((n_poles, n_out, n_in), dtype=np.complex128)
    eig_right = np.zeros((n, n_poles), dtype=np.complex128)
    eig_left = np.zeros((n, n_poles), dtype=np.complex128)

    for it, pole in enumerate(poles):
        b = np.asarray(b_probe.at(pole), dtype=np.complex128)
        c = np.asarray(c_probe.at(pole), dtype=np.complex128)
        l = np.asarray(loop.at(pole), dtype=np.complex128)
        adj = _adjugate(l)

        r_nom[it, :, :] = c @ adj @ b

        u, s, vh = np.linalg.svd(adj, full_matrices=False)
        s1 = s[0] if s.size else 0.0
        denom = np.sqrt(r_den[it])
        if np.abs(denom) > 0 and np.isfinite(denom):
            eig_right[:, it] = u[:, 0] * np.sqrt(s1) / denom
            eig_left[:, it] = vh.conj().T[:, 0] * np.conj(np.sqrt(s1)) / np.conj(denom)

    with np.errstate(divide="ignore", invalid="ignore"):
        residues = r_nom / r_den[:, None, None]
    residues = np.where(np.isfinite(residues), residues, 0.0)

    if isinstance(d_term, (np.ndarray, list, tuple)):
        direct = np.asarray(d_term, dtype=np.complex128)
    else:
        direct = np.asarray(_to_probe(d_term, name="D").at(1.0 + 0j), dtype=np.complex128)

    eigenvectors = {"right": eig_right, "left": eig_left}
    return residues, direct, undriven, eigenvectors


def dss_to_pr_flamo_direct(
    delays: ArrayLike,
    A: Any,
    B: Any,
    C: Any,
    D: Any,
    *,
    mode: str = "eig",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Direct poles/residues from DSS using FLAMO/autograd probe adapters.

    Parameters
    ----------
    delays
        Delay lengths in samples, shape (N,).
    A, B, C, D
        Feedback / input / output / direct terms as FLAMO graph objects or
        numeric matrices.
    mode
        Pole extraction mode:
        - ``"eig"``: via equivalent state-space matrix eigenvalues.
        - ``"roots"``: via generalized characteristic polynomial roots.
        - ``"polyeig"``: currently mapped to ``"roots"`` for this direct path.
    """
    delays_arr = np.asarray(delays, dtype=int).ravel()
    if delays_arr.ndim != 1 or delays_arr.size == 0:
        raise ValueError("delays must be a non-empty 1-D array")

    a_probe = _to_probe(A, name="A")
    b_probe = _to_probe(B, name="B")
    c_probe = _to_probe(C, name="C")

    loop = _SimpleLoop(delays=delays_arr, feedback=a_probe)
    a_static = _coerce_static_feedback_matrix(a_probe)

    mode_l = str(mode).lower()
    a_poly = np.real_if_close(a_static, tol=1000)
    if np.iscomplexobj(a_poly) and np.max(np.abs(np.imag(a_poly))) > 1e-12:
        raise ValueError(
            "roots/polyeig mode currently requires a real-valued static feedback matrix."
        )
    a_poly = np.asarray(np.real(a_poly), dtype=float)

    if mode_l == "eig":
        aa, _, _, _ = dss_to_ss(delays_arr, a_static)
        poles = np.linalg.eigvals(aa)
    elif mode_l == "roots":
        p = general_char_poly(delays_arr, a_poly)
        poles = np.roots(p)
    elif mode_l == "polyeig":
        warnings.warn(
            "mode='polyeig' is currently mapped to roots-based extraction in "
            "dss_to_pr_flamo_direct.",
            stacklevel=2,
        )
        p = general_char_poly(delays_arr, a_poly)
        poles = np.roots(p)
    else:
        raise ValueError("mode must be 'eig', 'roots', or 'polyeig'")

    poles, is_conjugate_pair, non_paired = _reduce_conjugate_pairs(poles)
    residues, direct, undriven, eigenvectors = _dss_to_res_flamo(
        poles, loop, b_probe, c_probe, D
    )

    meta_data: dict[str, Any] = {
        "nonPairedPoles": non_paired,
        "undrivenResidues": undriven,
        "eigenvectors": eigenvectors,
    }
    return residues, poles, direct, is_conjugate_pair, meta_data

