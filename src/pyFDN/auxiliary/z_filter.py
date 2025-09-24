from abc import ABC, abstractmethod
import numpy as np

class ZFilter(ABC):
    """
    z-Domain Filter structure abstract class.
    From this, multiple classes such as zFIR, zTF, zSOS are derived.

    Author: Sebastian J. Schlecht
    """
    def __init__(self):
        self.number_of_delay_units = 0
        self.is_diagonal = False
        self.n = None
        self.m = None

    def at(self, z):
        """
        Evaluate filter at z.
        """
        val = self.at_(z)
        if self.is_diagonal:
            return np.diag(val)
        return val

    def der(self, z):
        """
        Evaluate derivative at z.
        """
        val = self.der_(z)
        if self.is_diagonal:
            return np.diag(val)
        return val

    def parse_arguments(self, *args, **kwargs):
        """
        Set optional arguments. Currently supports is_diagonal.
        """
        self.is_diagonal = kwargs.get('is_diagonal', self.default_shape())

    def check_shape(self, m):
        """
        Ensure shape is compatible with diagonal filters.
        """
        if self.is_diagonal and m != 1:
            raise ValueError("For a diagonal filter matrix, provide a vector of filters.")

    @property
    def size(self):
        """
        Return the (n, m) size of the filter.
        """
        if self.n is None or self.m is None:
            raise ValueError("Size is not defined")
        return self.n, self.m

    def default_shape(self):
        """
        Default shape for the filter (non-diagonal)
        """
        return False

    # ---- Abstract methods ----
    @abstractmethod
    def inverse(self):
        """
        Inverse filter
        """
        pass

    @abstractmethod
    def at_(self, z):
        """
        Raw shape-independent values
        """
        pass

    @abstractmethod
    def der_(self, z):
        """
        Raw shape-independent derivative values
        """
        pass

    @abstractmethod
    def dfilt_type(self):
        """
        Corresponding dfilt filter type
        """
        pass

    @abstractmethod
    def dfilt_parameter(self, n, m):
        """
        Corresponding dfilt parameter format
        """
        pass
