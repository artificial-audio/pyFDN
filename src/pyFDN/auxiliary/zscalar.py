from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.zfilter import ZFilter


class ZScalar(ZFilter):
    """Constant matrix filter in the z-domain."""

    def __init__(self, matrix: ArrayLike, **kwargs) -> None:
        super().__init__()

        if not isinstance(matrix, np.ndarray):
            matrix = np.asarray(matrix, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("ZScalar expects a 2-D matrix")

        self.n, self.m = matrix.shape
        legacy_flag = kwargs.pop("isDiagonal", None)
        diagonal_flag = kwargs.pop("is_diagonal", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        if legacy_flag is not None and diagonal_flag is not None and legacy_flag != diagonal_flag:
            raise ValueError("Conflicting values for diagonal configuration")

        combined_flag = legacy_flag if legacy_flag is not None else diagonal_flag
        parse_args = {"isDiagonal": bool(combined_flag)} if combined_flag is not None else {}
        self.parse_arguments(parse_args)
        self.check_shape(self.m)

        self._matrix = matrix.astype(np.complex128, copy=False)
        self._matrix_der = np.zeros_like(self._matrix)
        self.number_of_delay_units = 0

    def _at(self, z: complex | np.ndarray) -> np.ndarray:
        return self._matrix

    def _der(self, z: complex | np.ndarray) -> np.ndarray:
        return self._matrix_der

    def inverse(self) -> "ZScalar":
        if self.is_diagonal:
            return ZScalar(1.0 / self._matrix, is_diagonal=True)
        return ZScalar(np.linalg.inv(self._matrix), is_diagonal=False)

    def dfilt_type(self) -> str:
        return "none"

    def dfilt_parameter(self, n: int, m: int):
        return self._matrix[n, m]
