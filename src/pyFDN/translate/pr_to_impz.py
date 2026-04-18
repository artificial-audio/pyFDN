"""From poles/residues to impulse response (pr2impz translation)."""

from __future__ import annotations

import numpy as np


def pr_to_impz(
    residues: np.ndarray,
    poles: np.ndarray,
    direct: np.ndarray,
    is_conjugate_pole_pair: np.ndarray,
    impulse_response_length: int,
    mode: str = "lowMemory",
) -> np.ndarray:
    """
    Synthesize impulse response from poles and residues.

    Parameters
    ----------
    residues
        Shape ``(num_poles, num_outputs, num_inputs)``.
    poles
        Pole vector of length ``num_poles``.
    direct
        Direct term, shape ``(num_outputs, num_inputs)``.
    is_conjugate_pole_pair
        Boolean/vector mask, same length as ``poles``.
    impulse_response_length
        Number of samples in the synthesized response.
    mode
        ``"fast"`` (vectorized) or ``"lowMemory"``.
    """
    residues = np.asarray(residues, dtype=np.complex128)
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    direct = np.asarray(direct, dtype=np.complex128)
    pair_flag = np.asarray(is_conjugate_pole_pair).astype(np.int64).ravel()

    print(mode)

    factor = pair_flag + 1
    num_poles = poles.size
    num_outputs = residues.shape[1]
    num_inputs = residues.shape[2]
    response = np.zeros((impulse_response_length, num_outputs, num_inputs), dtype=np.float64)

    t = np.arange(-1, impulse_response_length - 1, dtype=np.float64).reshape(-1, 1)
    angle = np.angle(poles).reshape(1, -1)
    mag = np.abs(poles).reshape(1, -1)

    if mode == "fast":
        c = factor.reshape(1, -1) * np.exp(1j * t * angle)
        e = np.exp(np.log(mag) * t)
        ce = c * e
        for in_idx in range(num_inputs):
            response[:, :, in_idx] = np.real(ce @ residues[:, :, in_idx])
    elif mode == "lowMemory":
        for pole_idx in range(num_poles):
            c = factor[pole_idx] * np.exp(1j * t[:, 0] * np.angle(poles[pole_idx]))
            e = np.exp(np.log(np.abs(poles[pole_idx])) * t[:, 0])
            ce = c * e
            for in_idx in range(num_inputs):
                response[:, :, in_idx] += np.real(
                    ce.reshape(-1, 1) @ residues[pole_idx, :, in_idx].reshape(1, -1)
                )
    else:
        raise ValueError("mode must be 'fast' or 'lowMemory'")

    response[0, :, :] = np.real_if_close(direct)
    return response

