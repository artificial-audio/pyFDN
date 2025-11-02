import numpy as np

from pyFDN.auxiliary.det_polynomial import det_polynomial
from pyFDN.auxiliary.poly_degree import poly_degree
from pyFDN.auxiliary.polydiag import polydiag
from pyFDN.auxiliary.tf_matrix import TFMatrix
from pyFDN.auxiliary.ztf import ZTF
from pyFDN.auxiliary.zfilter import ZFilter


class ZFIR(ZFilter):
    """
    FIR filter in z-domain derived from zFilter.
    """

    def __init__(self, b, **kwargs):
        super().__init__(**kwargs)
        b = np.array(b)
        self.n, self.m = b.shape[:2]

        self.parseArguments(kwargs)
        self.checkShape(self.m)

        # FIR has denominator = ones
        denominator = np.ones((self.n, self.m))
        self.matrix = self.tfMatrix(b, denominator)

        # Derivative
        self.matrixDer = self.derive(self.matrix)

        # Number of delay units
        self.numberOfDelayUnits = self.getDelays(b)

    # -------------------------
    # Helper methods (stubs)
    # -------------------------
    def tfMatrix(self, numerator, denominator):
        """Convert numerator and denominator to a TFMatrix"""
        return TFMatrix(numerator, denominator)

    def derive(self, matrix):
        """Compute derivative of TFMatrix"""
        return matrix.derive()

    def getDelays(self, numerator):
        """Compute number of delay units"""
        if self.isDiagonal:
            numerator_full = polydiag(np.transpose(numerator, (0, 2, 1)))
        else:
            numerator_full = numerator
        return poly_degree(det_polynomial(numerator_full), variable='z^-1')

    # -------------------------
    # Shape-independent access
    # -------------------------
    def at_(self, z):
        return self.matrix.at(z)

    def der_(self, z):
        return self.matrixDer.at(z)

    def inverse(self):
        return ZTF(
            self.matrix.denominator,
            self.matrix.numerator,
            isDiagonal=self.isDiagonal
        )

    def dfiltType(self):
        return "dffir"

    def dfiltParameter(self, n, m):
        # MATLAB slicing: b(n,m,:) → Python slicing: b[n,m,:]
        return self.matrix.numerator[n, m, :]
