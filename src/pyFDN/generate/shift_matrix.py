from __future__ import annotations
from numpy.typing import ArrayLike
import numpy as np

from pyFDN.auxiliary.utils import ensure_3d, last_nonzero_indices

def shift_matrix(mat: ArrayLike, shift: ArrayLike, direction: str) -> np.ndarray:
    """Shift a polynomial matrix in time-domain by ``shift`` samples."""

    mat_arr = ensure_3d(mat).copy()
    shift_vec = np.asarray(shift, dtype=int)

    if direction == "left":
        if shift_vec.ndim != 1 or shift_vec.size != mat_arr.shape[0]:
            raise ValueError("Shift vector must match number of rows")
        required_space = last_nonzero_indices(mat_arr) + shift_vec[:, np.newaxis]
        additional_space = int(np.max(required_space) - mat_arr.shape[2])
        if additional_space > 0:
            pad = np.zeros((mat_arr.shape[0], mat_arr.shape[1], additional_space), dtype=mat_arr.dtype)
            mat_arr = np.concatenate([mat_arr, pad], axis=2)
        for idx in range(mat_arr.shape[0]):
            mat_arr[idx, :, :] = np.roll(mat_arr[idx, :, :], shift_vec[idx], axis=1)
        return mat_arr

    if direction == "right":
        if shift_vec.ndim != 1 or shift_vec.size != mat_arr.shape[1]:
            raise ValueError("Shift vector must match number of columns")
        required_space = last_nonzero_indices(mat_arr) + shift_vec[np.newaxis, :]
        additional_space = int(np.max(required_space) - mat_arr.shape[2])
        if additional_space > 0:
            pad = np.zeros((mat_arr.shape[0], mat_arr.shape[1], additional_space), dtype=mat_arr.dtype)
            mat_arr = np.concatenate([mat_arr, pad], axis=2)
        for idx in range(mat_arr.shape[1]):
            mat_arr[:, idx, :] = np.roll(mat_arr[:, idx, :], shift_vec[idx], axis=1)
        return mat_arr

    raise ValueError("direction must be 'left' or 'right'")
