# dss_to_impz.py
import numpy as np

from pyFDN.auxiliary.filters import ZFilter
from pyFDN.process import process_fdn

def dss_to_impz(ir_len, delays, A, B, C, D, input_type='splitInput', extra_matrix=None, absorption_filters=None):
    """
    Compute impulse response from delay state-space (DSS) representation.

    Parameters
    ----------
    ir_len : int
        Length of impulse response in samples
    delays : list or array
        Delay lengths in samples
    A, B, C, D : numeric or ZFilter
        Delay state-space matrices
    input_type : str
        'splitInput' or 'mergeInput'
    extra_matrix : FilterMatrix or None
        Optional time-varying matrix (use FilterMatrix.from_data from pyFDN.dsp.filter_matrix)
    absorption_filters : FilterMatrix or None
        Optional absorption filters (use FilterMatrix.from_data from pyFDN.dsp.filter_matrix)

    Returns
    -------
    impulse_response : ndarray
        Shape [ir_len, num_outputs, num_inputs]
    """
    # Wrap matrices as zFilter if needed
    A = ZFilter.from_any(A)
    B = ZFilter.from_any(B)
    C = ZFilter.from_any(C)

    num_inputs = B.m

    # Dirac pulse input
    input_signal = np.zeros((ir_len, num_inputs))
    input_signal[0, :] = 1

    # Process FDN
    impulse_response = process_fdn(
        input_signal,
        delays,
        A,
        B,
        C,
        D,
        input_type=input_type,
        extra_matrix=extra_matrix,
        absorption_filters=absorption_filters
    )

    return impulse_response

