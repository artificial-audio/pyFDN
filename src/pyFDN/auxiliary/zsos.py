from __future__ import annotations
import numpy as np

from pyFDN.auxiliary.negpolyder import negpolyder
from pyFDN.auxiliary.zfilter import ZFilter


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
