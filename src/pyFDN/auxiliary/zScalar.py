import numpy as np
from pyFDN.auxiliary.zfilter import ZFilter

class ZScalar(ZFilter):
    """
    Scalar Matrix in zFilter and its derivative (= zeros).
    Converted from MATLAB version (Sebastian J. Schlecht, 2019).
    """

    def __init__(self, matrix, **kwargs):
        super().__init__()
        self.n, self.m = matrix.shape
        self.parseArguments(kwargs)
        self.checkShape(self.m)

        if not isinstance(matrix, np.ndarray):
            raise ValueError("Needs a numpy array matrix")

        self.matrix = matrix
        self.matrixDer = np.zeros_like(matrix)
        self.numberOfDelayUnits = 0

    # Equivalent to MATLAB at_(z)
    def at_(self, z):
        return self.matrix

    # Equivalent to MATLAB der_(z)
    def der_(self, z):
        return self.matrixDer

    def inverse(self):
        if self.isDiagonal:
            return ZScalar(1.0 / self.matrix, isDiagonal=self.isDiagonal)
        else:
            return ZScalar(np.linalg.inv(self.matrix), isDiagonal=self.isDiagonal)

    def dfiltType(self):
        return "none"

    def dfiltParameter(self, n, m):
        return self.matrix[n, m]
