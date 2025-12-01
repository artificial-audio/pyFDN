"""z-Domain Filter structure classes and utilities."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.math import (
    matrix_convolution,
    matrix_polyder,
    matrix_polyval,
    negpolyder,
    det_polynomial,
    poly_degree,
    polydiag,
)
from pyFDN.auxiliary.utils import ensure_3d


class TFMatrix:
    """
    Implementation of transfer function matrix (in z-Domain).
    
    var = 'z^1' polynomial variable
    higher -> lower power: ..., z^3, z^2, z^1, 1
    
    var = 'z^-1' polynomial variable  
    lower -> higher power: 1, z^-1, z^-2, z^-3, ...
    """
    
    def __init__(self, numerator: np.ndarray, denominator: np.ndarray = None, var: str = 'z^-1'):
        """
        Initialize transfer function matrix.
        
        Args:
            numerator: Numerator polynomial coefficients
            denominator: Denominator polynomial coefficients (default: 1)
            var: Variable type ('z^1' or 'z^-1')
        """
        if isinstance(numerator, TFMatrix):
            # Copy constructor
            self.numerator = numerator.numerator
            self.denominator = numerator.denominator
            self.var = numerator.var
        else:
            # Regular constructor
            self.numerator = np.asarray(numerator)
            if denominator is None:
                self.denominator = np.ones_like(self.numerator[:, :, :1])
            else:
                self.denominator = np.asarray(denominator)
            self.var = var
        
        # Computation acceleration
        self.flip_numerator = np.flip(self.numerator, axis=2)
        self.flip_denominator = np.flip(self.denominator, axis=2)
    
    def derive(self) -> 'TFMatrix':
        """Compute the derivative of the transfer function matrix."""
        B = np.transpose(self.numerator, (2, 0, 1))
        A = np.transpose(self.denominator, (2, 0, 1))
        
        if self.var == 'z^1':
            num, den = matrix_polyder(B, A)
        elif self.var == 'z^-1':
            num, den = matrix_polyder(B, A, self.var)
        else:
            raise ValueError(f"Unknown variable type: {self.var}")
        
        num = np.transpose(num, (1, 2, 0))
        den = np.transpose(den, (1, 2, 0))
        
        return TFMatrix(num, den, self.var)
    
    def at(self, z: complex | np.ndarray) -> np.ndarray:
        """Evaluate transfer function matrix at z."""
        if self.var == 'z^1':
            num = matrix_polyval(self.numerator, z)
            den = matrix_polyval(self.denominator, z)
            return num / den
        elif self.var == 'z^-1':
            iz = 1 / z
            num = matrix_polyval(self.flip_numerator, iz)
            den = matrix_polyval(self.flip_denominator, iz)
            return num / den
        else:
            raise ValueError(f"Unknown variable type: {self.var}")
    
    def __mul__(self, other: 'TFMatrix') -> 'TFMatrix':
        """Multiply two transfer function matrices."""
        if not isinstance(other, TFMatrix):
            other = TFMatrix(other)
        
        num = matrix_convolution(self.numerator, other.numerator)
        den = matrix_convolution(self.denominator, other.denominator)
        
        return TFMatrix(num, den, self.var)
    
    def poles(self) -> np.ndarray:
        """Compute poles of the transfer function matrix."""
        roots_list = []
        n, m, length = self.denominator.shape
        
        for nn in range(n):
            for mm in range(m):
                poly_coeffs = self.denominator[nn, mm, :]
                # Remove leading zeros
                poly_coeffs = np.trim_zeros(poly_coeffs, 'f')
                if len(poly_coeffs) > 1:
                    roots_list.extend(np.roots(poly_coeffs))
        
        return np.unique(np.array(roots_list)) if roots_list else np.array([])


class ZFilter(ABC):
    """
    Abstract base class for z-domain filter structures.
    
    This class provides a common interface for various filter types (e.g., TF, SOS).
    It handles the distinction between full and diagonal matrix filters.
    """
    
    def __init__(self):
        self.number_of_delay_units = 0
        self.is_diagonal = False
        self.n = None
        self.m = None

    @staticmethod
    def from_any(m) -> "ZFilter":
        """
        Convert numeric input to ZFilter object if needed.

        Parameters
        ----------
        m : ndarray, list, or ZFilter
            Numeric matrix/array or ZFilter object.

        Returns
        -------
        zF : ZFilter
            Wrapped ZFilter object (ZScalar or ZFIR) or the input if already ZFilter.
        """
        if isinstance(m, (np.ndarray, list)):
            m = np.array(m)
            if m.ndim == 2:
                return ZScalar(m)
            else:
                return ZFIR(m)
        elif isinstance(m, ZFilter):
            return m
        else:
            raise TypeError("Type not defined for ZFilter conversion")
    
    def at(self, z: complex | np.ndarray) -> np.ndarray:
        """Evaluate the filter's transfer function at z."""
        if self.is_diagonal:
            val = self._at(z)
            return np.diag(val.flatten()) if val.ndim > 1 else np.diag(val)
        else:
            return self._at(z)
    
    def der(self, z: complex | np.ndarray) -> np.ndarray:
        """Evaluate the derivative of the filter's transfer function at z."""
        if self.is_diagonal:
            val = self._der(z)
            return np.diag(val.flatten()) if val.ndim > 1 else np.diag(val)
        else:
            return self._der(z)
    
    def parse_arguments(self, args: dict):
        """Parse input arguments."""
        self.is_diagonal = args.get('isDiagonal', self.default_shape())
    
    def check_shape(self, m: int):
        """Check if the input dimensions are valid for a diagonal filter."""
        if self.is_diagonal and m != 1:
            raise ValueError('For a diagonal filter matrix, provide a vector of filters.')
    
    def size(self) -> tuple[int, int]:
        """Return the size (n, m) of the filter matrix."""
        if self.n is None or self.m is None:
            raise ValueError('Size is not defined')
        return self.n, self.m
    
    def default_shape(self) -> bool:
        """Default shape is not diagonal."""
        return False
    
    @abstractmethod
    def _at(self, z: complex | np.ndarray) -> np.ndarray:
        """Raw, shape-independent evaluation of the transfer function."""
        pass
    
    @abstractmethod
    def _der(self, z: complex | np.ndarray) -> np.ndarray:
        """Raw, shape-independent evaluation of the transfer function's derivative."""
        pass
    
    @abstractmethod
    def inverse(self):
        """Get the inverse filter."""
        pass
    
    @abstractmethod
    def dfilt_type(self):
        """Get the corresponding dfilt filter type."""
        pass
    
    @abstractmethod
    def dfilt_parameter(self, n: int, m: int):
        """Get parameters in a format suitable for dfilt."""
        pass


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


class ZSOS(ZFilter):
    """z-domain second-order sections filter."""
    
    def __init__(self, sos: np.ndarray, **kwargs):
        super().__init__()
        self.parse_arguments(kwargs)
        
        # sos is [n,m,nsos,6]
        sos = np.asarray(sos)
        self.n, self.m, nsos, coeff_len = sos.shape
        self.check_shape(self.m)
        assert coeff_len == 6, 'SOS need to have 6 coefficients'
        
        self.sos = sos
        self.number_of_delay_units = self.n * nsos * 2
        
        # Precompute derivatives
        self.dsos = np.zeros((self.n, self.m, nsos, 10))
        for nn in range(self.n):
            for mm in range(self.m):
                for ss in range(nsos):
                    num = self.sos[nn, mm, ss, :3]
                    den = self.sos[nn, mm, ss, 3:6]
                    b, a = negpolyder(num, den, dont_truncate=True)
                    # Store in format [b(5), a(5)]
                    self.dsos[nn, mm, ss, :5] = b[:5] if len(b) >= 5 else np.pad(b, (0, 5-len(b)))
                    self.dsos[nn, mm, ss, 5:10] = a[:5] if len(a) >= 5 else np.pad(a, (0, 5-len(a)))
    
    def _at(self, z: complex | np.ndarray) -> np.ndarray:
        """Shape independent evaluation."""
        # Powers for z^0, z^-1, z^-2
        m = np.array([0, -1, -2]).reshape(1, 1, 1, 3)
        z_powers = np.power(z, m)
        
        num = np.sum(z_powers * self.sos[:, :, :, :3], axis=3)
        den = np.sum(z_powers * self.sos[:, :, :, 3:6], axis=3)
        
        val = np.prod(num, axis=2) / np.prod(den, axis=2)
        return val
    
    def _der(self, z: complex | np.ndarray) -> np.ndarray:
        """Shape independent derivative evaluation."""
        # Value of sos
        m = np.array([0, -1, -2]).reshape(1, 1, 1, 3)
        z_powers = np.power(z, m)
        
        num = np.sum(z_powers * self.sos[:, :, :, :3], axis=3)
        den = np.sum(z_powers * self.sos[:, :, :, 3:6], axis=3)
        h = num / den
        
        # Derivative of sos
        dm = np.array([0, -1, -2, -3, -4]).reshape(1, 1, 1, 5)
        z_powers_der = np.power(z, dm)
        
        dnum = np.sum(z_powers_der * self.dsos[:, :, :, :5], axis=3)
        dden = np.sum(z_powers_der * self.dsos[:, :, :, 5:10], axis=3)
        dh = dnum / dden
        
        # Product rule: (f * g * h)' = (f * g * h) * (f'/f + g'/g + h'/h)
        fgh = np.prod(h, axis=2)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = dh / h
            ratio = np.where(np.isfinite(ratio), ratio, 0)
        
        ffgghh = np.sum(ratio, axis=2)
        val = fgh * ffgghh
        
        return val
    
    def inverse(self) -> 'ZSOS':
        """Get the inverse filter."""
        # Switch the denominator and numerator
        isos = self.sos.copy()
        isos[:, :, :, :3] = self.sos[:, :, :, 3:6]
        isos[:, :, :, 3:6] = self.sos[:, :, :, :3]
        
        return ZSOS(isos, isDiagonal=self.is_diagonal)
    
    def dfilt_type(self) -> str:
        """Get the corresponding dfilt filter type."""
        return 'df2sos'
    
    def dfilt_parameter(self, n: int, m: int) -> dict:
        """Get parameters in dfilt format."""
        sos = np.transpose(self.sos[n, m, :, :], (0, 1))
        return {'sos': sos}



