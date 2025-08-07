import numpy as np

from pyFDN.auxiliary.matrix_convolution import matrix_convolution
from pyFDN.auxiliary.matrix_polyder import matrix_polyder
from pyFDN.auxiliary.matrix_polyval import matrix_polyval


class TFMatrix:
    """
    Implementation of transfer function matrix (in z-Domain).
    - var: 'z^1' or 'z^-1'
    - numerator, denominator: numpy arrays of shape (n, m, len)
    """

    def __init__(self, numerator, denominator=None, var='z^-1'):
        if isinstance(numerator, TFMatrix):  # Copy constructor
            self.numerator = numerator.numerator.copy()
            self.denominator = numerator.denominator.copy()
            self.var = numerator.var
        else:
            self.numerator = numerator
            if denominator is not None:
                self.denominator = denominator
                self.var = var
            else:
                self.denominator = np.ones_like(numerator[..., :1])
                self.var = 'z^-1'
        # Precompute flipped versions for 'z^-1' case
        self.flip_numerator = np.flip(self.numerator, axis=2)
        self.flip_denominator = np.flip(self.denominator, axis=2)

    def derive(self):
        B = np.transpose(self.numerator, (2, 0, 1))
        A = np.transpose(self.denominator, (2, 0, 1))
        if self.var == 'z^1':
            num, den = matrix_polyder(B, A)
            num = np.transpose(num, (1, 2, 0))
            den = np.transpose(den, (1, 2, 0))
            return TFMatrix(num, den, self.var)
        elif self.var == 'z^-1':
            num, den = matrix_polyder(B, A, self.var)
            num = np.transpose(num, (1, 2, 0))
            den = np.transpose(den, (1, 2, 0))
            return TFMatrix(num, den, self.var)
        else:
            raise ValueError("Unknown var type")

    def at(self, z):
        if self.var == 'z^1':
            num = matrix_polyval(self.numerator, z)
            den = matrix_polyval(self.denominator, z)
            return num / den
        elif self.var == 'z^-1':
            iz = 1.0 / z
            num = matrix_polyval(self.flip_numerator, iz)
            den = matrix_polyval(self.flip_denominator, iz)
            return num / den
        else:
            raise ValueError("Unknown var type")

    def __mul__(self, other):
        # Matrix multiplication of two TFMatrix objects
        obj1 = TFMatrix(self)
        obj2 = TFMatrix(other)
        num = matrix_convolution(obj1.numerator, obj2.numerator)
        den = matrix_convolution(obj1.denominator, obj2.denominator)
        return TFMatrix(num, den)

    def poles(self):
        # Collect all unique poles from the denominator polynomials
        R = []
        n, m, length = self.denominator.shape
        for nn in range(n):
            for mm in range(m):
                coeffs = self.denominator[nn, mm, :]
                roots = np.roots(coeffs)
                R.extend(roots)
        return np.unique(R)
