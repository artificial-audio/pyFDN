"""Compute residues from impulse response and poles (impz2res translation)."""

from __future__ import annotations

import numpy as np


def impz_to_res(
    impulse_response: np.ndarray,
    poles: np.ndarray,
    is_conjugate_pole_pair: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate residues from impulse response and known poles via least squares.

    Notes
    -----
    This function follows fdnToolbox's ``impz2res.m`` and is currently
    implemented for SISO impulse responses.
    """
    ir = np.asarray(impulse_response, dtype=np.float64)
    if ir.ndim != 1:
        raise ValueError("impz_to_res currently expects a 1-D SISO impulse response")

    poles = np.asarray(poles, dtype=np.complex128).ravel()
    pair_flag = np.asarray(is_conjugate_pole_pair).astype(np.int64).ravel()
    factor = pair_flag + 1

    impulse_response_length = int(4 * poles.size)
    if ir.size < impulse_response_length:
        raise ValueError(
            "impulse_response is shorter than 4 * len(poles), "
            "which is required by impz2res least-squares formulation"
        )

    # Remove direct term and crop as in MATLAB.
    ir_use = ir[1:impulse_response_length]

    t = np.arange(impulse_response_length, dtype=np.float64).reshape(-1, 1)
    angle = np.angle(poles).reshape(1, -1)
    mag = np.abs(poles).reshape(1, -1)

    c = factor.reshape(1, -1) * np.exp(1j * t * angle)
    e = np.exp(np.log(mag) * t)
    ce = np.real(c * e)

    u = np.concatenate([ce[:-1, :], ce[1:, :]], axis=1)
    b, _, _, _ = np.linalg.lstsq(u, ir_use, rcond=None)
    b0 = b[: poles.size]
    b1 = b[poles.size :]

    residues = (b0 + b1 * np.real(poles)) + 1j * (np.imag(poles) * b1)
    return residues, b0, b1

