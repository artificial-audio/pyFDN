"""Transfer function matrix implementation."""
from __future__ import annotations
import numpy as np

from pyFDN.auxiliary.matrix_convolution import matrix_convolution
from pyFDN.auxiliary.matrix_polyder import matrix_polyder
from pyFDN.auxiliary.matrix_polyval import matrix_polyval


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