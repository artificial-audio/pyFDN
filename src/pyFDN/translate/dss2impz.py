# dss2impz.py
import numpy as np

from pyFDN.auxiliary.convert2zfilter import convert2zFilter
from pyFDN.auxiliary.process_fdn import process_fdn

def dss2impz(ir_len, delays, A, B, C, D, input_type='splitInput', extra_matrix=None, absorption_filters=None):
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
    extra_matrix : DFiltMatrix or None
        Optional time-varying matrix
    absorption_filters : DFiltMatrix or None
        Optional absorption filters

    Returns
    -------
    impulse_response : ndarray
        Shape [ir_len, num_outputs, num_inputs]
    """
    # Wrap matrices as zFilter if needed
    A = convert2zFilter(A)
    B = convert2zFilter(B)
    C = convert2zFilter(C)

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

