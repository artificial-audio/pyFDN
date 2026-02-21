# dss_to_impz.py
import numpy as np

from pyFDN.auxiliary.filters import ZFilter
from pyFDN.process import process_fdn


def dss_to_impz(ir_len, delays, A, B, C, D, extra_matrix=None, absorption_filters=None):
    """
    Compute MIMO impulse response from delay state-space (DSS) representation.

    Runs one simulation per input channel (Dirac at t=0 on that channel only)
    and stacks the results into a single array.

    Parameters
    ----------
    ir_len : int
        Length of impulse response in samples
    delays : list or array
        Delay lengths in samples
    A, B, C, D : numeric or ZFilter
        Delay state-space matrices
    extra_matrix : FilterMatrix or None
        Optional time-varying matrix (use FilterMatrix.from_data from pyFDN.dsp.filter_matrix)
    absorption_filters : FilterMatrix or None
        Optional absorption filters (use FilterMatrix.from_data from pyFDN.dsp.filter_matrix)

    Returns
    -------
    impulse_response : ndarray
        Shape [ir_len, num_outputs, num_inputs]
    """
    A = ZFilter.from_any(A)
    B = ZFilter.from_any(B)
    C = ZFilter.from_any(C)

    num_inputs = B.m
    out_list = []

    for j in range(num_inputs):
        input_signal = np.zeros((ir_len, num_inputs))
        input_signal[0, j] = 1.0
        out_j = process_fdn(
            input_signal,
            delays,
            A,
            B,
            C,
            D,
            extra_matrix=extra_matrix,
            absorption_filters=absorption_filters,
        )
        # out_j shape: (ir_len,) or (ir_len, num_outputs)
        if out_j.ndim == 1:
            out_j = out_j[:, np.newaxis]
        out_list.append(out_j)

    # Stack as (ir_len, num_outputs, num_inputs)
    impulse_response = np.stack(out_list, axis=-1)
    return impulse_response

