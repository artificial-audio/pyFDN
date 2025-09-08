import numpy as np

from pyFDN.auxiliary.convert2zfilter import convert2zFilter
from pyFDN.dsp.dfiltmatrix import DFiltMatrix
from pyFDN.dsp.feedback_delay import FeedbackDelay

# ---------------------------------------------------
# Assumes you have already defined:
# ZFilter, ZScalar, ZFIR
# convert2zFilter
# FeedbackDelay
# DFiltMatrix
# ---------------------------------------------------

def processFDN(input_signal, delays, A, B, C, D, input_type='mergeInput', extra_matrix=None, absorption_filters=None):
    """
    Time-domain processing of a Feedback Delay Network (FDN).

    Parameters
    ----------
    input_signal : ndarray
        Shape [num_samples, num_inputs]
    delays : list or ndarray
        Delays in samples, length N
    A : ZFilter or numeric
        Feedback matrix of shape [N,N]
    B : ZFilter or numeric
        Input gains [N, in]
    C : ZFilter or numeric
        Output gains [out, N]
    D : numeric
        Direct gains [out, in]
    input_type : str
        'splitInput' or 'mergeInput'
    extra_matrix : DFiltMatrix or None
        Optional time-varying matrix
    absorption_filters : DFiltMatrix or None
        Optional absorption filters per delay line

    Returns
    -------
    output : ndarray
        Output signal. Shape depends on input_type.
    """
    N = len(delays)
    num_samples = input_signal.shape[0]
    num_inputs = input_signal.shape[1] if input_signal.ndim > 1 else 1

    # Default absorption filters: identity
    if absorption_filters is None:
        absorption_filters = DFiltMatrix(np.eye(N))

    # Wrap numeric input into ZFilter
    A = convert2zFilter(A)
    B = convert2zFilter(B)
    C = convert2zFilter(C)

    # Convert to DFiltMatrix
    feedback_matrix = DFiltMatrix(A)
    input_gains = DFiltMatrix(B)
    output_gains = DFiltMatrix(C)

    if input_type == 'splitInput':
        output = np.zeros((num_samples, C.n, num_inputs))
        for it_in in range(num_inputs):
            # Select input channel
            in_block = np.zeros_like(input_signal)
            in_block[:, it_in] = input_signal[:, it_in]
            output[:, :, it_in] = computeFDNloop(in_block, delays, feedback_matrix, input_gains, output_gains, extra_matrix, absorption_filters)
        # Add direct path
        D = np.array(D).reshape(output.shape[1], output.shape[2])  # [out, in]
        output += np.expand_dims(input_signal, axis=1) * np.expand_dims(D, axis=0)

    elif input_type == 'mergeInput':
        output = computeFDNloop(input_signal, delays, feedback_matrix, input_gains, output_gains, extra_matrix, absorption_filters)
        output += input_signal @ D.T  # direct path
    else:
        raise ValueError("input_type must be 'splitInput' or 'mergeInput'")

    return output


def computeFDNloop(input_signal, delays, feedback_matrix, input_gains, output_gains, extra_matrix, absorption_filters):
    """
    Compute the FDN loop in the time domain.
    """
    max_block_size = 2**12
    blkSz = min(min(delays), max_block_size)
    input_len = input_signal.shape[0]
    output = np.zeros((input_len, output_gains.n))

    # Initialize delay lines
    delay_filters = FeedbackDelay(max_block_size, delays)

    block_start = 0
    while block_start < input_len:
        # Determine block indices
        blkSz_current = min(blkSz, input_len - block_start)
        blk_idx = slice(block_start, block_start + blkSz_current)
        block = input_signal[blk_idx, :]

        # Delays + absorption
        delay_output = delay_filters.get_values(blkSz_current)
        if absorption_filters is not None:
            delay_output = absorption_filters.filter(delay_output)

        # Feedback
        feedback = feedback_matrix.filter(delay_output)
        if extra_matrix is not None:
            feedback = extra_matrix.filter(feedback)

        # Input + feedback
        delay_line_input = input_gains.filter(block) + feedback
        delay_filters.set_values(delay_line_input)

        # Compute output
        output[blk_idx, :] = output_gains.filter(delay_output)

        # Advance pointers
        delay_filters.next(blkSz_current)
        block_start += blkSz_current

    return output
