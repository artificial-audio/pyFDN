"""z-Domain Filter structure classes."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from pyFDN.auxiliary.negpolyder import negpolyder

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