"""
ParallelOnePole filter implementation using FLAMO's parallelBiquad infrastructure.

This module provides a workaround for FLAMO's limitation that biquad filters
cannot have coefficients set directly, only through parameters like cutoff frequency.

Created in response to Sebastian Schlecht's feedback on pyFDN one-pole absorption filters.

by Facundo Franchino, early October 2025

"""

import torch
import numpy as np
import flamo.processor.dsp as dsp


class ParallelOnePole(dsp.parallelBiquad):
    """Parallel one-pole filters using FLAMO's biquad infrastructure.
    
    This class implements one-pole filters by overriding the get_poly_coeff 
    method of parallelBiquad to set coefficients directly.
    
    This is a workaround for FLAMO's limitation that biquad filters expect
    parameters (cutoff, gain) rather than allowing direct coefficient assignment.
    
    Args:
        b_coeffs (array-like): Numerator coefficients, shape (N, 1, 1)
        a_coeffs (array-like): Denominator coefficients, shape (N, 1, 2) 
                               where a_coeffs[:, 0, 0] = 1.0 and a_coeffs[:, 0, 1] = a1
        *args: Arguments passed to parallelBiquad
        **kwargs: Keyword arguments passed to parallelBiquad
    """
    
    def __init__(self, b_coeffs, a_coeffs, *args, **kwargs):
        # store coefficients before calling super().__init__ as it may call get_poly_coeff
        self.b_coeffs = torch.tensor(b_coeffs, dtype=torch.float32)  # shape: (N, 1, 1)
        self.a_coeffs = torch.tensor(a_coeffs, dtype=torch.float32)  # shape: (N, 1, 2)
        super().__init__(*args, **kwargs)
        
    def get_poly_coeff(self, param):
        """Override to set one-pole coefficients directly.
        
        Computes the frequency response H(z) = b0 / (1 + a1*z^-1) for each
        one-pole filter and returns it in the format expected by parallelBiquad.
        
        Args:
            param: Ignored - we use stored coefficients instead
            
        Returns:
            tuple: (H, B, A) where:
                - H: Frequency response, shape (nfft//2+1, N)
                - B: Numerator coefficients (for compatibility)
                - A: Denominator coefficients (for compatibility)
        """
        N = self.size[0]
        nfft = self.nfft
        n_sections = self.n_sections
        
        # frequency bins
        omega = torch.linspace(0, np.pi, nfft//2+1, device=self.device)
        z_inv = torch.exp(-1j * omega)  # z^-1
        
        # initialisew frequency response for each channel
        H = torch.zeros(nfft//2+1, N, dtype=torch.complex64, device=self.device)
        
        for i in range(N):
            b0 = float(self.b_coeffs[i, 0, 0])
            a0 = float(self.a_coeffs[i, 0, 0])  # should be 1
            a1 = float(self.a_coeffs[i, 0, 1])
            
            # one-pole transfer function: H(z) = b0 / (1 + a1*z^-1)
            H[:, i] = b0 / (a0 + a1 * z_inv + 1e-10)
        
        # create B and A tensors for compatibility with parallelBiquad
        B = H.unsqueeze(1) * torch.ones(1, n_sections, 1, device=self.device)
        A = torch.ones_like(B)
        
        return H, B, A