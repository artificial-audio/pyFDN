# gallery_category: Translation Examples

import marimo

__generated_with = "0.23.9"
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
    - Plots the FDN parameters (feedback matrix, delays, I/O vectors, magnitude response) with `pyFDN.plot_FDN_build`.
    """)
    return


@app.cell
def _():
    import dataclasses

    import numpy as np
    from scipy.signal import dimpulse, dlti, ss2tf

    import pyFDN

    np.random.seed(1)
    # Impulse response length for comparison
    impulse_response_length = 1000

    m = np.array([13, 19, 23])
    build = pyFDN.fdn_build_gallery(
        delays=m,
        io_type="random",
        direct_gain=None,
        rt=0.02,
        rng=1,
    )
    # Bake attenuation into A matrix for easier comparison.
    build = dataclasses.replace(build, A=np.diag(build.filters[0, 0, :]) @ build.A)
    A, b, c, d = build.A, build.B, build.C, build.D

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
    return aa, bb, build, cc, dd, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FDN parameter overview
    """)
    return


@app.cell
def _(build, pyFDN):
    pyFDN.plot_FDN_build(build, title="FDN parameters")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Equivalent state-space system

    The single state-space matrices `(A, b, c, d)` returned by `pyFDN.dss_to_ss`,
    expanding the delay lines into unit-delay states.
    """)
    return


@app.cell
def _(aa, bb, cc, dd, pyFDN):
    pyFDN.plot_system_matrix(aa, bb, cc, dd, title="State-space system")
    return


if __name__ == "__main__":
    app.run()
