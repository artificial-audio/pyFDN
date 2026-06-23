"""
Time Varying matrix generator for FDN feedback matrices.

This module provides a Python implementation of a time-varying matrix generator
for Feedback Delay Network (FDN) feedback matrices.
Translation of the MATLAB implementation `timeVaryingMatrix.m` from fdnToolbox.

Original MATLAB code: (c) Sebastian Jiro Schlecht, 2019
Python translation: Alma Hova, 2026
"""

import numpy as np


class TimeVaryingMatrix:
    """
    Time Varying Matrix for Feedback Delay Networks (FDNs).

    This class generates a time-varying matrix for FDN feedback matrices. The
    matrix varies over time based on sinusoidal modulation, with parameters
    controlling the speed, amplitude, and randomness of the variation.

    Parameters
    ----------
    N : int
        Number of channels (size of the matrix is N x N). Must be a positive even integer.
    cycles_per_second : float
        Frequency of the time variation in Hz (controls oscillation speed).
    amplitude : float
        Maximum angle deflection in radians (strength of modulation).
    fs : float
        Sampling rate in Hz.
    spread : float
        Randomness factor (controls how differently each eigenmode behaves).
    """

    def __init__(
        self,
        N: int,
        cycles_per_second: float,
        amplitude: float,
        fs: float,
        spread: float,
    ) -> None:
        """
        Initialize the TimeVaryingMatrix object.

        Attributes
        ----------
        N : int
            Number of channels.
        cycles_per_second : float
            Frequency of the time variation in Hz.
        amplitude : float
            Maximum angle deflection in radians.
        fs : float
            Sampling rate in Hz.
        spread : float
            Randomness factor.
        num_pairs : int
            Number of eigenmode pairs (N // 2).
        phase : ndarray
            Random initial phases for each eigenmode pair.
        frequency : ndarray
            Frequencies of oscillation for each eigenmode pair.
        angle_amplitude : ndarray
            Amplitudes of oscillation for each eigenmode pair.
        sample_index : int
            Current sample index for time tracking.
        """
        # Enforce N to be a positive, even integer.
        N = int(N)
        if N <= 0:
            raise ValueError("N must be a positive integer")
        if N % 2 != 0:
            raise ValueError("N must be even")

        self.N = N
        self.cycles_per_second = cycles_per_second
        self.amplitude = amplitude
        self.fs = fs
        self.spread = spread

        # Calculate the number of independent 2D rotation planes (conjugate eigenvalue pairs)
        self.num_pairs = N // 2

        # Assign a random initial phase between 0 and 2*pi for each 2D plane
        self.phase = 2 * np.pi * np.random.rand(self.num_pairs)

        # Calculate a unique modulation frequency for each pair using the spread factor
        self.frequency = self.cycles_per_second * (
            1 + self.spread * (2 * np.random.rand(self.num_pairs) - 1)
        )

        # Modulation Amplitude
        self.angle_amplitude = self.amplitude * (
            1 + self.spread * (2 * np.random.rand(self.num_pairs) - 1)
        )

        # Global time tracker index, initialized to 0
        self.sample_index = 0

    def filter(self, x_in: np.ndarray) -> np.ndarray:
        """
        Applies a time-varying orthogonal transformation to the input signal.

        Each adjacent channel pair is rotated by a sinusoidally modulated
        angle. The operation is equivalent to constructing the block-diagonal
        rotation matrix from ``rotation_matrix_from_angles`` at every sample,
        but applies the 2-D rotations directly to the whole input block.

        Parameters
        ----------
        x_in : ndarray
            Input signal of shape (length, N), where `length` is the number of
            samples and `N` is the number of channels.

        Returns
        -------
        out : ndarray
            Output signal of the same shape as `x_in`.
        """
        x = np.asarray(x_in)
        if x.ndim != 2 or x.shape[1] != self.N:
            raise ValueError(f"x_in must have shape (length, {self.N})")

        length = x.shape[0]
        sample_indices = self.sample_index + np.arange(length)
        time = sample_indices[:, np.newaxis] / self.fs
        angles = self.angle_amplitude * np.sin(
            2 * np.pi * self.frequency * time + self.phase
        )
        cos = np.cos(angles)
        sin = np.sin(angles)

        x_pairs = x.reshape(length, self.num_pairs, 2)
        out = np.empty(x.shape, dtype=np.result_type(x, self.angle_amplitude, float))
        out_pairs = out.reshape(length, self.num_pairs, 2)
        out_pairs[..., 0] = cos * x_pairs[..., 0] - sin * x_pairs[..., 1]
        out_pairs[..., 1] = sin * x_pairs[..., 0] + cos * x_pairs[..., 1]

        self.sample_index += length

        return out
