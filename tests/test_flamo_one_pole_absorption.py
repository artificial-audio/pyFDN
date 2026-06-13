"""
FLAMO import tests for one-pole absorption filters against MATLAB FDN Toolbox

tests to double check that the Python implementation matches the MATLAB reference.

created by Facundo Franchino, early October 2025
"""

import os
from collections import OrderedDict

import numpy as np
import pytest

from pyFDN.auxiliary.acoustics import one_pole_absorption
from pyFDN.auxiliary.flamo import flamo_time_response


@pytest.fixture(scope="module")
def one_pole_absorption_reference(loadmat):
    """Load MATLAB reference data and generate Python coefficients/impulse response."""

    # path to MATLAB reference file
    REFERENCE_MAT_FILE = os.path.join(
        os.path.dirname(__file__),
        "reference",
        "example_onePoleAbsorption.mat",
    )

    ref = loadmat(REFERENCE_MAT_FILE, upcast=True)

    # Extract reference parameters
    RT_DC = ref["RT_DC"]
    RT_NY = ref["RT_NY"]
    delays = ref["delays"].flatten()
    fs = int(ref["fs"])
    feedback_matrix = ref["feedbackMatrix"]
    ir_matlab = ref["irTimeDomain"].flatten()
    N = int(ref["N"])

    # Extract MATLAB coefficient results
    sos_matlab = ref["absorption"]

    # Generate coefficients in Python (SOS bank, shape (1, 6, N))
    sos_python = one_pole_absorption(RT_DC, RT_NY, delays, fs)

    # Generate impulse response via FLAMO integration (if available)
    try:
        import torch  # type: ignore
        from flamo.processor import dsp, system  # type: ignore
    except ImportError:
        ir_python = None
    else:
        nfft = 2**16
        device = "cpu"

        delays_torch = torch.tensor(delays, dtype=torch.float32)
        feedback_matrix_torch = torch.tensor(feedback_matrix, dtype=torch.float32)

        absorption_coeff = torch.tensor(sos_python, dtype=torch.float32)

        input_gain = dsp.Gain(size=(N, 1), nfft=nfft, device=device)
        input_gain.assign_value(torch.ones(N, 1))

        output_gain = dsp.Gain(size=(1, N), nfft=nfft, device=device)
        output_gain.assign_value(torch.ones(1, N))

        delay = dsp.parallelDelay(
            size=(N,),
            max_len=int(delays_torch.max()),
            nfft=nfft,
            isint=True,
            unit=1,
            device=device,
        )
        delay.assign_value(delay.sample2s(delays_torch))

        mixing_matrix = dsp.Matrix(
            size=(N, N),
            nfft=nfft,
            matrix_type="random",
            device=device,
        )
        mixing_matrix.assign_value(feedback_matrix_torch)

        absorption = dsp.parallelSOSFilter(
            size=(N,), n_sections=1, nfft=nfft, device=device
        )
        absorption.assign_value(absorption_coeff)

        attenuated_delay = system.Series(
            OrderedDict(
                {
                    "delay": delay,
                    "absorption": absorption,
                }
            )
        )

        feedback_loop = system.Recursion(fF=attenuated_delay, fB=mixing_matrix)

        fdn = system.Series(
            OrderedDict(
                {
                    "input_gain": input_gain,
                    "feedback_loop": feedback_loop,
                    "output_gain": output_gain,
                }
            )
        )

        direct_gain = dsp.Gain(size=(1, 1), nfft=nfft, device=device)
        direct_gain.assign_value(torch.ones(1, 1))

        complete_system = system.Parallel(
            brA=direct_gain,
            brB=fdn,
            sum_output=True,
        )

        model = system.Shell(
            core=complete_system,
            input_layer=dsp.FFT(nfft),
            output_layer=dsp.iFFT(nfft),
        )

        # with torch.no_grad():
        #     impulse = torch.zeros(1, nfft, 1)
        #     impulse[0, 0, 0] = 1.0
        #     ir_python = model(impulse).squeeze().cpu().numpy()

        ir_python = flamo_time_response(model).flatten()

    return {
        "sos_matlab": sos_matlab,
        "sos_python": sos_python,
        "ir_matlab": ir_matlab,
        "ir_python": ir_python,
    }


def test_coefficients(one_pole_absorption_reference):
    """test that pyFDN generates same absorption coefficients as MATLAB."""

    sos_matlab = one_pole_absorption_reference["sos_matlab"]
    sos_python = one_pole_absorption_reference["sos_python"]

    b_matlab = sos_matlab["b"].flatten(order="F")
    a_matlab = sos_matlab["a"].flatten(order="F")

    b_python = sos_python[0, 0, :].flatten()
    a_python = sos_python[0, 3:5, :].flatten()

    # compare coefficients
    np.testing.assert_allclose(
        b_python,
        b_matlab,
        rtol=1e-14,
        atol=1e-16,
        err_msg="b coefficients don't match MATLAB reference",
    )
    np.testing.assert_allclose(
        a_python,
        a_matlab,
        rtol=1e-14,
        atol=1e-16,
        err_msg="a coefficients don't match MATLAB reference",
    )


def test_impulse_response(one_pole_absorption_reference):
    """Test that pyFDN with FLAMO generates similar impulse response to MATLAB FDN."""

    ir_matlab = one_pole_absorption_reference["ir_matlab"]
    ir_python = one_pole_absorption_reference["ir_python"]

    if ir_python is None:
        pytest.skip("FLAMO integration not available")

    min_len = min(len(ir_python), len(ir_matlab))
    ir_python = ir_python[:min_len]
    ir_matlab = ir_matlab[:min_len]

    np.testing.assert_almost_equal(  # probably not exact match due to implementation differences
        ir_python, ir_matlab, decimal=3, err_msg="Impulse responses don't match"
    )


if __name__ == "__main__":
    # run tests when executed directly
    pytest.main([__file__, "-v"])
