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
        sos (array-like): SOS filter coefficients, shape (6, N) with [b0, b1, b2, a0, a1, a2] per column
                          For one-pole filters: b1=b2=a2=0, a0=1
        *args: Arguments passed to parallelBiquad
        **kwargs: Keyword arguments passed to parallelBiquad
    """
    
    def __init__(self, sos, *args, **kwargs):
        # store SOS coefficients before calling super().__init__ as it may call get_poly_coeff
        # sos shape: (6, N) with [b0, b1, b2, a0, a1, a2] per column
        self.sos = torch.tensor(sos, dtype=torch.float32)
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
                - B: Numerator coefficients in frequency domain
                - A: Denominator coefficients in frequency domain
        """
        N = self.size[0]
        nfft = self.nfft
        
        # create polynomial coefficients for each filter
        # one-pole: H(z) = b0 / (1 + a1*z^-1)
        # this is a biquad with: b = [b0, 0, 0], a = [1, a1, 0]
        
        # Extract coefficients from SOS format: [b0, b1, b2, a0, a1, a2] per column
        b_poly = self.sos[:3, :].to(self.device)  # [b0, b1, b2] for each filter
        a_poly = self.sos[3:, :].to(self.device)  # [a0, a1, a2] for each filter
        
        # apply anti-aliasing envelope (following FLAMO convention)
        # create impulse responses for each filter
        impulse_length = nfft // 2
        b_aa = torch.zeros((impulse_length, N), device=self.device)
        a_aa = torch.zeros((impulse_length, N), device=self.device)
        
        # set impulse at t=0,1,2 for b and a coefficients
        b_aa[0, :] = b_poly[0, :]  # b0
        b_aa[1, :] = b_poly[1, :]  # b1
        b_aa[2, :] = b_poly[2, :]  # b2
        
        a_aa[0, :] = a_poly[0, :]  # a0
        a_aa[1, :] = a_poly[1, :]  # a1  
        a_aa[2, :] = a_poly[2, :]  # a2
        
        # fft to get frequency domain
        B = torch.fft.rfft(b_aa, nfft, dim=0)  # shape: (nfft//2+1, N)
        A = torch.fft.rfft(a_aa, nfft, dim=0)  # shape: (nfft//2+1, N)
        
        # compute frequency response
        H = B / (A + 1e-12)  # add small epsilon for numerical stability
        
        return H, B, A