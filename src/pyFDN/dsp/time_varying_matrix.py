"""
Time Varying matrix generator for FDN feedback matrices.

This module provides a Python implementation of a time-varying matrix generator
for Feedback Delay Network (FDN) feedback matrices.
Translation of the MATLAB implementation `timeVaryingMatrix.m` from fdnToolbox.

Original MATLAB code: (c) Sebastian Jiro Schlecht, 2019
Python translation: Alma Hova, 2026
"""

import numpy as np

from pyFDN.auxiliary.tiny_rotation_matrix import rotation_matrix_from_angles


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

        For each time step, a matrix Q(n) is constructed from the current
        modulation state and applied to the signal.

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

        # Get the number of audio samples in the incoming block
        length = x_in.shape[0]

        # Output array matching the size of the input
        out = np.empty_like(x_in)

        # Loop through every single audio sample instance in the block
        for n in range(length):
            # Calculate the absolute time 't' in seconds from the start of processing
            t = (self.sample_index + n) / self.fs

            # Compute the rotation angle for each 2D plane at time instance 't'
            # using independent sinusoidal oscillators
            angles = self.angle_amplitude * np.sin(
                2 * np.pi * self.frequency * t + self.phase
            )

            # generate the combined N x N real orthogonal feedback matrix Q(n) for this sample
            Q = rotation_matrix_from_angles(
                angles,
                n=self.N,
            )

            # Matrix-vector multiplication: Apply the orthogonal transformation matrix Q
            # to the multi-channel input sample vector at index [n]
            out[n] = Q @ x_in[n]

        # Accumulate the total processed sample length
        self.sample_index += length

        return out
