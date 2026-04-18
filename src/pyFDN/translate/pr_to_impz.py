"""From poles/residues to impulse response (pr2impz translation)."""

from __future__ import annotations

import numpy as np


def pr_to_impz(
    residues: np.ndarray,
    poles: np.ndarray,
    direct: np.ndarray,
    is_conjugate_pole_pair: np.ndarray,
    impulse_response_length: int,
    mode: str = "chunked",
    max_memory_bytes: int = 256 * 1024 * 1024,
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
        ``"fast"`` (vectorized), ``"lowMemory"`` (pole-by-pole), or
        ``"chunked"`` (batched; chunk size derived from ``max_memory_bytes``).
    max_memory_bytes
        Memory budget in bytes for intermediate arrays when
        ``mode="chunked"``.  Defaults to 256 MiB.
    """
    residues = np.asarray(residues, dtype=np.complex128)
    poles = np.asarray(poles, dtype=np.complex128).ravel()
    direct = np.asarray(direct, dtype=np.complex128)
    pair_flag = np.asarray(is_conjugate_pole_pair).astype(np.int64).ravel()

    factor = pair_flag + 1
    num_poles = poles.size
    num_outputs = residues.shape[1]
    num_inputs = residues.shape[2]
    response = np.zeros((impulse_response_length, num_outputs, num_inputs), dtype=np.float64)

    t = np.arange(-1, impulse_response_length - 1, dtype=np.float64).reshape(-1, 1)
    angle = np.angle(poles).reshape(1, -1)
    mag = np.abs(poles).reshape(1, -1)

    if mode == "fast":
        e = np.exp(np.log(mag) * t)
        ce = factor.reshape(1, -1) * np.exp(1j * t * angle) * e
        response[:] = np.real(np.einsum('tp,poi->toi', ce, residues))
    elif mode == "lowMemory":
        for pole_idx in range(num_poles):
            c = factor[pole_idx] * np.exp(1j * t[:, 0] * angle[0, pole_idx])
            e = np.exp(np.log(mag[0, pole_idx]) * t[:, 0])
            ce = c * e
            response += np.real(
                np.outer(ce, residues[pole_idx]).reshape(impulse_response_length, num_outputs, num_inputs)
            )
    elif mode == "chunked":
        # Per pole: e (float64, 8 B) + complex-exp temp (complex128, 16 B) + ce (complex128, 16 B)
        bytes_per_pole = impulse_response_length * (8 + 16 + 16)
        chunk_size = max(1, min(num_poles, int(max_memory_bytes // bytes_per_pole)))
        for start in range(0, num_poles, chunk_size):
            end = min(start + chunk_size, num_poles)
            e = np.exp(np.log(mag[:, start:end]) * t)
            ce = factor[start:end].reshape(1, -1) * np.exp(1j * t * angle[:, start:end]) * e
            response += np.real(np.einsum('tp,poi->toi', ce, residues[start:end]))
    else:
        raise ValueError("mode must be 'fast', 'lowMemory', or 'chunked'")

    response[0, :, :] = np.real_if_close(direct)
    return response

