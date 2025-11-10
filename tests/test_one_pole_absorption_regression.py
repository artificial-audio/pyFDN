"""
regression tests for one-pole absorption filters against MATLAB FDN Toolbox 

tests to double check that the Python implementation matches the MATLAB reference.

created by Facundo Franchino, early October 2025
"""

import numpy as np
import os
import pytest

from pyFDN.auxiliary.one_pole_absorption import one_pole_absorption
from pyFDN.generate.random_orthogonal import random_orthogonal


# path to MATLAB reference file
REFERENCE_MAT_FILE = os.path.join(
    os.path.dirname(__file__),
    'reference', 
    'example_onePoleAbsorption.mat'
)


def test_one_pole_absorption_coefficients(loadmat):
    """test that pyFDN generates same absorption coefficients as MATLAB."""
    
    # load MATLAB ref
    ref = loadmat(REFERENCE_MAT_FILE)
    
    # extract reference parameters
    RT_DC = ref['RT_DC']
    RT_NY = ref['RT_NY']
    delays = ref['delays']
    fs = ref['fs']
    
    # extract MATLAB coefficient results
    b_matlab = ref['absorption']['b']
    a_matlab = ref['absorption']['a']
    
    # generate coefficients in Python
    b_python, a_python = one_pole_absorption(RT_DC, RT_NY, delays, fs)
    
    # reshape Python outputs to match MATLAB format if needed
    if b_python.shape == (4, 1, 1) and b_matlab.shape == (4, 1):
        b_python = b_python.squeeze(-1)
    if a_python.shape == (4, 1, 2) and a_matlab.shape == (4, 2):
        a_python = a_python.squeeze(1)
    elif a_matlab.shape == (4, 1, 2) and a_python.shape == (4, 2):
        a_matlab = a_matlab.squeeze(1)
    
    # compare coefficients
    np.testing.assert_allclose(b_python, b_matlab, rtol=1e-14, atol=1e-16,
                              err_msg="b coefficients don't match MATLAB reference")
    np.testing.assert_allclose(a_python, a_matlab, rtol=1e-14, atol=1e-16,
                              err_msg="a coefficients don't match MATLAB reference")


def test_random_orthogonal_matrix():
    """test that the random orthogonal matrix is indeed orthogonal."""
    
    # load MATLAB reference for comparison
    ref = load_matlab_reference()
    
    # extract the feedback matrix used in MATLAB
    feedback_matrix_matlab = ref['feedbackMatrix']
    
    # generate our own orthogonal matrix (will be different due to RNG)
    np.random.seed(1)
    feedback_matrix_python = random_orthogonal(4)
    
    # test orthogonality for both matrices
    identity = np.eye(4)
    
    # Python matrix
    product_python = feedback_matrix_python.T @ feedback_matrix_python
    
    # MATLAB matrix
    product_matlab = feedback_matrix_matlab.T @ feedback_matrix_matlab
    
    # both should be orthogonal (product with transpose = identity)
    np.testing.assert_allclose(product_python, identity, rtol=1e-12, atol=1e-15,
                              err_msg="Python matrix is not orthogonal")
    np.testing.assert_allclose(product_matlab, identity, rtol=1e-12, atol=1e-15,
                              err_msg="MATLAB matrix is not orthogonal")


def test_impulse_response_comparison():
    """Test that pyFDN with FLAMO generates similar impulse response to MATLAB FDN."""
    # Import FLAMO components
    try:
        from flamo.processor import dsp, system
        from collections import OrderedDict
        import torch
    except ImportError:
        pytest.skip("FLAMO not available")
        
    # import pyFDN FLAMO integration
    try:
        from pyFDN.dsp.parallel_one_pole import ParallelOnePole
    except ImportError:
        pytest.skip("ParallelOnePole not available")
    
    # load MATLAB reference
    ref = load_matlab_reference()
    
    # Extract parameters
    delays = ref['delays'].flatten()
    feedback_matrix = ref['feedbackMatrix']
    RT_DC = float(ref['RT_DC'].item())
    RT_NY = float(ref['RT_NY'].item())
    fs = int(ref['fs'].item())
    N = int(ref['N'].item())
    ir_matlab = ref['irTimeDomain'].flatten()
    
    # generate absorption filters
    b, a = one_pole_absorption(RT_DC, RT_NY, delays, fs)
    
    # build FLAMO FDN
    nfft = 16384
    device = 'cpu'
    
    # convert to torch tensors
    delays_torch = torch.tensor(delays, dtype=torch.float32)
    feedback_matrix_torch = torch.tensor(feedback_matrix, dtype=torch.float32)
    
    # build components
    input_gain = dsp.Gain(size=(N, 1), nfft=nfft, device=device)
    input_gain.assign_value(torch.ones(N, 1))
    
    output_gain = dsp.Gain(size=(1, N), nfft=nfft, device=device)
    output_gain.assign_value(torch.ones(1, N))
    
    delay_module = dsp.parallelDelay(
        size=(N,),
        max_len=int(delays_torch.max()),
        nfft=nfft,
        isint=True,
        device=device
    )
    delay_module.assign_value(delay_module.sample2s(delays_torch.int()))
    
    mixing_matrix = dsp.Matrix(
        size=(N, N),
        nfft=nfft,
        matrix_type="random",
        device=device
    )
    mixing_matrix.assign_value(feedback_matrix_torch)
    
    absorption = ParallelOnePole(
        b_coeffs=b,
        a_coeffs=a,
        size=(N,),
        n_sections=1,
        nfft=nfft,
        device=device
    )
    
    # build system
    feedback = system.Series(OrderedDict({
        "mixing_matrix": mixing_matrix,
        "absorption": absorption
    }))
    
    feedback_loop = system.Recursion(fF=delay_module, fB=feedback)
    
    fdn = system.Series(OrderedDict({
        "input_gain": input_gain,
        "feedback_loop": feedback_loop,
        "output_gain": output_gain
    }))
    
    # add direct path
    direct_gain = dsp.Gain(size=(1, 1), nfft=nfft, device=device)
    direct_gain.assign_value(torch.ones(1, 1))
    
    complete_system = system.Parallel(
        brA=direct_gain,
        brB=fdn,
        sum_output=True
    )
    
    model = system.Shell(
        core=complete_system,
        input_layer=dsp.FFT(nfft),
        output_layer=dsp.iFFT(nfft)
    )
    
    # generate impulse response
    with torch.no_grad():
        impulse = torch.zeros(1, nfft, 1)
        impulse[0, 0, 0] = 1.0
        ir_flamo = model(impulse).squeeze().cpu().numpy()
    
    # compare impulse responses
    min_len = min(len(ir_flamo), len(ir_matlab))
    ir_f = ir_flamo[:min_len]
    ir_m = ir_matlab[:min_len]
    
    # PLOT the two impulse responses for visual inspection
    import matplotlib.pyplot as plt
    t = np.arange(min_len) / fs
    plt.figure(figsize=(12, 6))
    plt.plot(t, ir_m, label="MATLAB Reference", alpha=0.7)
    plt.plot(t, ir_f, label="FLAMO pyFDN Output", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Impulse Response Comparison (MATLAB vs FLAMO/pyFDN)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # check correlation (FLAMO uses FFT-based processing so exact match not expected)
    correlation = np.corrcoef(ir_f, ir_m)[0, 1]
    
    # we expect reasonable correlation but not perfect match due to different implementations
    # MATLAB uses time-domain processing while FLAMO uses FFT-based processing
    assert correlation > 0.95, f"Impulse responses should be correlated, got {correlation:.3f}"
    
    # check that both have similar energy
    energy_flamo = np.sum(ir_f**2)
    energy_matlab = np.sum(ir_m**2)
    energy_ratio = energy_flamo / energy_matlab
    
    assert 0.95 < energy_ratio < 1.05, f"Energy should be similar, ratio={energy_ratio:.3f}"


def test_without_matlab_reference():
    """test basic functionality even without MATLAB reference file."""
    
    # test parameters
    fs = 48000
    RT_DC = 3.0
    RT_NY = 0.1
    delays = np.array([1320, 1650, 2790, 550])
    
    # test coefficient generation
    b, a = one_pole_absorption(RT_DC, RT_NY, delays, fs)
    
    # basic sanity checks
    assert b.shape[0] == len(delays), "b coefficients should match number of delays"
    assert a.shape[0] == len(delays), "a coefficients should match number of delays"
    assert np.all(np.isfinite(b)), "b coefficients should be finite"
    assert np.all(np.isfinite(a)), "a coefficients should be finite"
    
    # test orthogonal matrix generation
    np.random.seed(1)
    matrix = random_orthogonal(4)
    
    # test orthogonality
    product = matrix.T @ matrix
    identity = np.eye(4)
    np.testing.assert_allclose(product, identity, rtol=1e-12, atol=1e-15,
                              err_msg="generated matrix is not orthogonal")


if __name__ == "__main__":
    # run tests when executed directly
    pytest.main([__file__, "-v"])