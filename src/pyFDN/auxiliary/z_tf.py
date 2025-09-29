from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from pyFDN.auxiliary.tf_matrix import TFMatrix
from pyFDN.auxiliary.det_polynomial import det_polynomial
from pyFDN.auxiliary.poly_degree import poly_degree
from pyFDN.auxiliary.polydiag import polydiag
from pyFDN.auxiliary.z_filter import ZFilter


class ZTF(ZFilter):
    """z-domain transfer function filter. Represents H(z) = B(z) / A(z)."""
    
    def __init__(self, b: np.ndarray, a: np.ndarray, **kwargs):
        super().__init__()
        self.parse_arguments(kwargs)
        
        b = np.asarray(b)
        a = np.asarray(a)
        
        bn, bm, b_len = b.shape
        self.n, self.m, a_len = a.shape
        
        assert bn == self.n, 'Filter sizes need to match'
        assert bm == self.m, 'Filter sizes need to match'
        self.check_shape(self.m)
        
        self.matrix = TFMatrix(b, a, 'z^-1')
        self.matrix_der = self.matrix.derive()
        self.number_of_delay_units = self._get_delays(b)
    
    def _get_delays(self, numerator: np.ndarray) -> int:
        """Get the number of delay units."""
        if self.is_diagonal:
            numerator_full = polydiag(np.transpose(numerator, (0, 2, 1)))
        else:
            numerator_full = numerator
        
        delays = poly_degree(det_polynomial(numerator_full, 'z^-1'), 'z^-1')
        return delays
    
    def _at(self, z: complex | np.ndarray) -> np.ndarray:
        """Shape independent evaluation."""
        return self.matrix.at(z)
    
    def _der(self, z: complex | np.ndarray) -> np.ndarray:
        """Shape independent derivative evaluation using quotient rule."""
        # Direct computation of derivative using quotient rule
        # For H(z) = B(z)/A(z), H'(z) = (B'(z)*A(z) - B(z)*A'(z)) / (A(z))^2
        
        # Get the polynomial coefficients
        b = self.matrix.numerator  # [n, m, order]
        a = self.matrix.denominator
        
        # For z^-1 variable, we need to handle the derivative properly
        result = np.zeros((self.n, self.m), dtype=complex)
        
        for i in range(self.n):
            for j in range(self.m):
                b_coeffs = b[i, j, :]
                a_coeffs = a[i, j, :]
                
                # Remove leading zeros
                b_coeffs = np.trim_zeros(b_coeffs, 'f')
                a_coeffs = np.trim_zeros(a_coeffs, 'f')
                
                if len(b_coeffs) == 0:
                    b_coeffs = np.array([0.0])
                if len(a_coeffs) == 0:
                    a_coeffs = np.array([1.0])
                
                # For z^-1 polynomials: B(z) = b0 + b1*z^-1 + b2*z^-2 + ...
                # Derivative with respect to z: B'(z) = -b1*z^-2 - 2*b2*z^-3 - ...
                
                # Compute B(z) and A(z)
                z_inv = 1.0 / z
                powers = np.arange(len(b_coeffs))
                z_powers_b = z_inv ** powers
                powers_a = np.arange(len(a_coeffs))
                z_powers_a = z_inv ** powers_a
                
                B_z = np.sum(b_coeffs * z_powers_b)
                A_z = np.sum(a_coeffs * z_powers_a)
                
                # Compute B'(z) and A'(z)
                # For z^-1 variable: d/dz(z^-k) = -k * z^(-k-1)
                if len(b_coeffs) > 1:
                    db_coeffs = -np.arange(1, len(b_coeffs)) * b_coeffs[1:]
                    z_powers_db = z_inv ** (np.arange(1, len(b_coeffs)) + 1)
                    B_prime_z = np.sum(db_coeffs * z_powers_db)
                else:
                    B_prime_z = 0.0
                
                if len(a_coeffs) > 1:
                    da_coeffs = -np.arange(1, len(a_coeffs)) * a_coeffs[1:]
                    z_powers_da = z_inv ** (np.arange(1, len(a_coeffs)) + 1)
                    A_prime_z = np.sum(da_coeffs * z_powers_da)
                else:
                    A_prime_z = 0.0
                
                # Apply quotient rule
                if A_z != 0:
                    result[i, j] = (B_prime_z * A_z - B_z * A_prime_z) / (A_z ** 2)
                else:
                    result[i, j] = 0.0
        
        return result
    
    def inverse(self) -> 'ZTF':
        """Get the inverse filter."""
        return ZTF(self.matrix.denominator, self.matrix.numerator, isDiagonal=self.is_diagonal)
    
    def dfilt_type(self) -> str:
        """Get the corresponding dfilt filter type."""
        return 'df2'
    
    def dfilt_parameter(self, n: int, m: int) -> dict:
        """Get parameters in dfilt format."""
        b = np.transpose(self.matrix.numerator[n, m, :], (0,))
        a = np.transpose(self.matrix.denominator[n, m, :], (0,))
        return {'b': b, 'a': a}