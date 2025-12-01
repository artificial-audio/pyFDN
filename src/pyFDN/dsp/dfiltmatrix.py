import numpy as np

from pyFDN.auxiliary.filters import ZFilter, ZScalar

class DFiltMatrix:
    """
    Wrapper for a matrix of filter objects (FIR, IIR, Scalar).

    Each entry can be:
    - ZScalar: numeric matrix
    - ZFIR: FIR filter
    - ZFilter: generic filter
    """

    def __init__(self, zF):
        # If numeric matrix, wrap as ZScalar
        if isinstance(zF, np.ndarray) and zF.ndim == 2:
            zF = ZScalar(zF)

        # Get dimensions
        self.n = getattr(zF, 'n', 1)
        self.m = getattr(zF, 'm', 1)
        self.is_diagonal = getattr(zF, 'isDiagonal', False)

        # Initialize filters
        if isinstance(zF, ZScalar):
            self.filters = zF.mat  # numeric matrix
        elif isinstance(zF, ZFilter):
            # Assume zF.filters is a 2D list of ZFilter objects
            self.filters = np.empty((self.n, self.m), dtype=object)
            for i in range(self.n):
                for j in range(self.m):
                    try:
                        val = zF.dfiltParameter(i, j)  # placeholder method
                        self.filters[i, j] = val
                    except AttributeError:
                        self.filters[i, j] = zF  # fallback
        else:
            raise TypeError("Provide a ZFilter or scalar gains")

    def filter(self, x):
        """
        Apply the filter matrix to input x.

        x: shape [samples, m]
        Returns: [samples, n]
        """
        num_samples = x.shape[0]
        out = np.zeros((num_samples, self.n))

        if isinstance(self.filters, np.ndarray) and np.issubdtype(self.filters.dtype, np.number):
            # Numeric matrix
            if self.is_diagonal:
                out = x * self.filters
            else:
                out = x @ self.filters.T
        else:
            # Object filters (ZFilter)
            if self.is_diagonal:
                for i in range(self.n):
                    out[:, i] += self.filters[i, 0].filter(x[:, i])
            else:
                for i in range(self.n):
                    for j in range(self.m):
                        out[:, i] += self.filters[i, j].filter(x[:, j])
        return out
