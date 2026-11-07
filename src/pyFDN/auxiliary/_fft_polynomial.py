"""FFT/Taylor polynomial evaluations near concentric circles, with bounded work.

Euler derivatives D**j p, D=z*d/dz, are FFT samples. A local expansion in
x=degree*log(z/z_grid) evaluates off-grid points and both ordinary derivatives.
Coefficients are scaled per circle; derivative ratios and backward residuals
are invariant to that scaling. Small point sets retain direct Horner evaluation.
"""

from math import factorial
from time import monotonic

import numpy as np


class FFTPolynomial:
    """Internal scaled evaluations for independent Laguerre polishing."""

    def __init__(self, coefficients: np.ndarray, nfft: int, deadline: float):
        self.coefficients = coefficients
        self.degree = coefficients.size - 1
        self.nfft = nfft
        self.deadline = deadline
        powers = np.arange(coefficients.size)
        self.fractions = powers / self.degree
        self.derivative = coefficients[1:] * powers[1:]
        self.second = self.derivative[1:] * powers[1:-1]

    def check_time(self) -> None:
        if monotonic() >= self.deadline:
            raise RuntimeError("Lindsey-Fox search exceeded its time budget")

    def evaluate(
        self, points: np.ndarray, *, derivatives: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return scaled p,p',p'', absolute allowance, and a coefficient-norm lower bound.

        The last two arrays are used for final backward-residual verification.
        Ordinary IEEE floating-point allowances are not interval arithmetic.
        """
        n = self.degree
        count = points.size
        value = np.full(count, np.nan + 0j)
        first, second = value.copy(), value.copy()
        allowance = np.full(count, np.inf)
        scale = np.zeros(count)
        radii = abs(points)
        finite = np.isfinite(points) & np.isfinite(radii)
        fft_mask = finite & (radii > 0)
        if n < 2048 or count < max(2048, n // 8):
            fft_mask[:] = False
        direct = np.flatnonzero(~fft_mask & finite)
        # Chunking also keeps deadlines interruptible for escaped candidates.
        for start in range(0, direct.size, 256):
            selected = direct[start : start + 256]
            z = points[selected]
            self.check_time()
            value[selected] = np.polynomial.polynomial.polyval(z, self.coefficients)
            self.check_time()
            if derivatives:
                first[selected] = np.polynomial.polynomial.polyval(z, self.derivative)
                self.check_time()
                second[selected] = np.polynomial.polynomial.polyval(z, self.second)
            else:
                scale[selected] = np.polynomial.polynomial.polyval(
                    abs(z), abs(self.coefficients)
                ).real
            allowance[selected] = 0

        selected_fft = np.flatnonzero(fft_mask)
        if selected_fft.size == 0:
            return value, first, second, allowance, scale
        z = points[selected_fft]
        logarithms = np.log(abs(z))
        bins = np.rint(n * logarithms).astype(int)
        # Exact quarter turns avoid subtracting nearly equal angles near 2*pi.
        real_dominates = abs(z.real) >= abs(z.imag)
        quadrant = np.where(
            real_dominates, np.where(z.real >= 0, 0, 2), np.where(z.imag >= 0, 1, 3)
        )
        local = z * np.array([1, -1j, -1, 1j])[quadrant]
        theta = np.angle(local)
        offset = np.rint(theta * self.nfft / (2 * np.pi)).astype(int)
        grid = (offset + quadrant * (self.nfft // 4)) % self.nfft
        angular_offset = theta - offset * (2 * np.pi / self.nfft)
        order = 18
        for anchor in np.unique(bins):
            self.check_time()
            group = np.flatnonzero(bins == anchor)
            target = selected_fft[group]
            delta = n * logarithms[group] - anchor + 1j * n * angular_offset[group]
            weighted = self.coefficients * np.exp(
                self.fractions * anchor - max(0, anchor)
            )
            maximum = float(np.max(abs(weighted)))
            if not np.isfinite(maximum) or maximum == 0:
                raise RuntimeError(
                    "Lindsey-Fox FFT evaluation could not scale its circle"
                )
            weighted /= maximum
            coefficient_norm = float(np.sum(abs(weighted)))
            samples = []
            for _ in range(order + (2 if derivatives else 0)):
                self.check_time()
                sampled = np.fft.ifft(weighted, n=self.nfft, norm="forward")
                samples.append(sampled[grid[group]])
                weighted *= self.fractions
            p = np.zeros(group.size, dtype=complex)
            q1, q2 = p.copy(), p.copy()
            term = np.ones(group.size, dtype=complex)
            for j in range(order):
                p += samples[j] * term
                if derivatives:
                    q1 += samples[j + 1] * term
                    q2 += samples[j + 2] * term
                term *= delta / (j + 1)
            value[target] = p
            if derivatives:
                first[target] = n * q1 / z[group]
                second[target] = (n * n * q2 - n * q1) / z[group] ** 2
            else:
                # Log-radius bins give |Re(delta)|<=1/2; angular spacing <=pi/8.
                magnitude = abs(delta)
                eps = np.finfo(float).eps
                truncation = np.exp(magnitude) * magnitude**order / factorial(order)
                fft_roundoff = (
                    eps * (64 * np.log2(self.nfft) + 8 * order) * np.exp(magnitude)
                )
                coordinates = 2 * n * eps * np.exp(np.maximum(0, delta.real))
                allowance[target] = coefficient_norm * (
                    truncation + fft_roundoff + coordinates
                )
                scale[target] = coefficient_norm * np.exp(np.minimum(0, delta.real))
        return value, first, second, allowance, scale
