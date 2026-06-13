# gallery_category: Translation Examples

import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Delay state-space to state-space

    This example converts an FDN in **delay state-space** form (separate delay lengths and feedback matrix) into a single **state-space** system, and checks that the impulse response matches the delay-state-space implementation.

    **What it does:**
    - Builds a small lossless FDN: random orthogonal matrix, diagonal of gains `g^m`, and random input/output vectors.
    - Uses `pyFDN.dss_to_ss` to get the equivalent state-space matrices `(A, b, c, d)`.
    - Computes the impulse response both via `scipy.signal` (from the state-space) and via `pyFDN.dss2impz` (from the delay state-space).
    - Plots both (mu-law encoded) and asserts they match within tolerance.
    """)
    return


@app.cell
def _():
    import numpy as np
    from scipy.signal import dimpulse, dlti, ss2tf

    import pyFDN

    np.random.seed(1)
    # Impulse response length for comparison
    impulse_response_length = 100

    # Lossless FDN: N delay lines, delay lengths m, gain g, random orthogonal feedback
    N = 3
    m = np.array([13, 19, 23])
    g = 0.9
    A = pyFDN.random_orthogonal(N) @ np.diag(g**m)
    b = np.random.randn(N, 1)
    c = np.random.randn(1, N)
    d = np.random.randn(1, 1)

    # Convert delay state-space to single state-space system
    aa, bb, cc, dd = pyFDN.dss_to_ss(m, A, b, c, d)
    # Via state-space (scipy): transfer function then impulse
    num, den = ss2tf(aa, bb, cc, dd)
    system = dlti(num, den)
    _, ir_state_space = dimpulse(system, n=impulse_response_length)
    ir_state_space = np.squeeze(ir_state_space)

    # Via delay state-space (pyFDN); shape (ir_len, n_out, n_in)
    ir_delay_state_space = pyFDN.dss_to_impz(impulse_response_length, m, A, b, c, d)
    ir_delay_state_space = np.asarray(ir_delay_state_space).squeeze()

    # Sanity check: both implementations match
    assert pyFDN.is_almost_zero(ir_state_space - ir_delay_state_space, tol=0.001)

    pyFDN.plot_impulse_response(
        ir_state_space,
        ir_delay_state_space,
        labels=["State space", "Delay state space"],
    )
    return


if __name__ == "__main__":
    app.run()
