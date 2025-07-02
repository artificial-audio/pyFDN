import numpy as np

from auxiliary import det_polynomial, poly_degree, polydiag
from tf_matrix import TFMatrix

class ZTF:
    def __init__(self, b, a, is_diagonal=True):
        # b, a: expected to be numpy arrays with shape (n, m, len)
        self.b = b
        self.a = a
        self.is_diagonal = is_diagonal

        # Check shapes
        bn, bm, blen = b.shape
        an, am, alen = a.shape
        assert bn == an, "Filter sizes need to match"
        assert bm == am, "Filter sizes need to match"
        self.n = an
        self.m = am

        self.check_shape(self.m)

        # Create transfer function matrix
        self.matrix = TFMatrix(b, a, var='z^-1')
        self.matrix_der = self.matrix.derive()
        self.number_of_delay_units = self.get_delays(b)

    def check_shape(self, m):
        if self.is_diagonal and m != 1:
            raise ValueError("For a diagonal filter matrix, provide a vector of filters.")

    def get_delays(self, numerator):
        if self.is_diagonal:
            numerator_full = polydiag(np.transpose(numerator, (0, 2, 1)))
        else:
            numerator_full = numerator
        return poly_degree(det_polynomial(numerator_full), var='z^-1')

    def at_(self, z):
        return self.matrix.at(z)

    def der_(self, z):
        return self.matrix_der.at(z)

    def inverse(self):
        # Swap numerator and denominator
        return ZTF(self.matrix.denominator, self.matrix.numerator, is_diagonal=self.is_diagonal)

    def dfilt_type(self):
        return 'df2'

    def dfilt_parameter(self, n, m):
        b = np.transpose(self.matrix.numerator[n, m, :], (2, 0, 1))
        a = np.transpose(self.matrix.denominator[n, m, :], (2, 0, 1))
        return b, a
