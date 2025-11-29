from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.det_polynomial import det_polynomial
from pyFDN.auxiliary.poly_degree import poly_degree
from pyFDN.auxiliary.polydiag import polydiag
from pyFDN.auxiliary.tf_matrix import TFMatrix
from pyFDN.auxiliary.ztf import ZTF
from pyFDN.auxiliary.zfilter import ZFilter


class ZFIR(ZFilter):
    """FIR z-domain filter backed by ``TFMatrix``."""

    def __init__(self, b: ArrayLike, **kwargs) -> None:
        super().__init__()

        b_arr = np.asarray(b, dtype=np.complex128)
        if b_arr.ndim != 3:
            raise ValueError("ZFIR expects a 3-D array of FIR coefficients")

        self.n, self.m = b_arr.shape[:2]

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

        denominator = np.ones((self.n, self.m, 1), dtype=np.complex128)
        self._matrix = TFMatrix(b_arr, denominator)
        self._matrix_der = self._matrix.derive()
        self.number_of_delay_units = int(self._calculate_delays(b_arr))

    def _calculate_delays(self, numerator: np.ndarray) -> int:
        if self.is_diagonal:
            numerator_full = polydiag(np.transpose(numerator, (0, 2, 1)))
        else:
            numerator_full = numerator
        delays = det_polynomial(numerator_full, var='z^-1')
        return poly_degree(delays, 'z^-1')

    def _at(self, z: complex | np.ndarray) -> np.ndarray:
        return self._matrix.at(z)

    def _der(self, z: complex | np.ndarray) -> np.ndarray:
        return self._matrix_der.at(z)

    def inverse(self) -> ZTF:
        return ZTF(
            self._matrix.denominator,
            self._matrix.numerator,
            is_diagonal=self.is_diagonal,
        )

    def dfilt_type(self) -> str:
        return "dffir"

    def dfilt_parameter(self, n: int, m: int):
        return self._matrix.numerator[n, m, :]
