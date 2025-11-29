from __future__ import annotations

from typing import Tuple

import numpy as np

from pyFDN.auxiliary.matrix_polyval import matrix_polyval
from pyFDN.auxiliary.det_polynomial import det_polynomial
from pyFDN.auxiliary.poly_degree import poly_degree
from pyFDN.auxiliary.polydiag import polydiag
from pyFDN.auxiliary.zfilter import ZFilter
from pyFDN.helpers.utils import ensure_3d


class ZTF(ZFilter):
    """Simple z-domain transfer-function matrix wrapper."""

    def __init__(
        self,
        numerator: np.ndarray,
        denominator: np.ndarray,
        is_diagonal: bool | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        legacy_flag = kwargs.pop("isDiagonal", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        if legacy_flag is not None and is_diagonal is not None and legacy_flag != is_diagonal:
            raise ValueError("Conflicting values for diagonal configuration")

        diagonal_flag = legacy_flag if legacy_flag is not None else is_diagonal
        self.parse_arguments({"isDiagonal": bool(diagonal_flag)}) if diagonal_flag is not None else self.parse_arguments({})

        self.numerator = ensure_3d(np.asarray(numerator, dtype=np.complex128))
        self.denominator = ensure_3d(np.asarray(denominator, dtype=np.complex128))

        if self.numerator.shape != self.denominator.shape:
            raise ValueError("Numerator and denominator must share the same shape")

        self.n, self.m = self.numerator.shape[:2]
        self.check_shape(self.m)

        self._exponents = np.arange(self.numerator.shape[2] - 1, -1, -1, dtype=int)

        numerator_full: np.ndarray | None
        if self.is_diagonal:
            diag_coeffs = np.transpose(self.numerator, (0, 2, 1))[:, :, 0]
            numerator_full = polydiag(diag_coeffs)
        elif self.n == self.m:
            numerator_full = self.numerator
        else:
            numerator_full = None

        if numerator_full is not None:
            det_poly = det_polynomial(np.asarray(numerator_full, dtype=np.complex128), 'z^-1')
            degree = poly_degree(det_poly, 'z^-1')
            self.number_of_delay_units = max(int(degree), 0)
        else:
            self.number_of_delay_units = max(self.numerator.shape[2], self.denominator.shape[2]) - 1

    @property
    def shape(self) -> Tuple[int, int]:
        return self.n, self.m

    def _at(self, z: complex | np.ndarray) -> np.ndarray:
        z_val = self._as_scalar(z)
        result_num = matrix_polyval(self.numerator, z_val)
        result_den = matrix_polyval(self.denominator, z_val)
        return result_num / result_den

    def _der(self, z: complex | np.ndarray) -> np.ndarray:
        z_val = self._as_scalar(z)

        num = matrix_polyval(self.numerator, z_val)
        den = matrix_polyval(self.denominator, z_val)
        num_der = self._polyval_derivative(self.numerator, z_val)
        den_der = self._polyval_derivative(self.denominator, z_val)

        with np.errstate(divide="ignore", invalid="ignore"):
            result = (num_der * den - num * den_der) / (den ** 2)
        return np.where(np.isfinite(result), result, 0)

    def inverse(self) -> "ZTF":
        return ZTF(self.denominator, self.numerator, is_diagonal=self.is_diagonal)

    def dfilt_type(self) -> str:
        return "df2tf"

    def dfilt_parameter(self, n: int, m: int) -> dict[str, np.ndarray]:
        return {"b": self.numerator[n, m, :], "a": self.denominator[n, m, :]}

    def _as_scalar(self, z: complex | np.ndarray) -> complex:
        arr = np.asarray(z)
        if arr.ndim == 0:
            return complex(arr.item())
        if arr.size == 1:
            return complex(arr.reshape(-1)[0])
        raise ValueError("ZTF expects scalar evaluation points")

    def _polyval_derivative(self, coeffs: np.ndarray, z_val: complex) -> np.ndarray:
        if coeffs.shape[2] == 1:
            return np.zeros(coeffs.shape[:2], dtype=np.complex128)

        exponents = self._exponents
        valid = exponents > 0

        if not np.any(valid):
            return np.zeros(coeffs.shape[:2], dtype=np.complex128)

        coeffs_valid = coeffs[:, :, valid].astype(np.complex128, copy=False)
        exp_valid = exponents[valid]
        deriv_coeffs = coeffs_valid * exp_valid.reshape(1, 1, -1)
        z_powers = np.power(z_val, exp_valid - 1).reshape(1, 1, -1)
        return np.sum(deriv_coeffs * z_powers, axis=2)
