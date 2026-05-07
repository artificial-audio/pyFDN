import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Delay state-space to transfer function

    Convert the delay state-space form of an FDN into matrix transfer function form (`dss_to_tf`). Verify by comparing the impulse response from the TF with the one from direct delay state-space simulation (`dss_to_impz`).

    **What it does:**
    - Defines an FDN with random orthogonal feedback, random delays, and MIMO B, C, D.
    - Converts to transfer function: `(tfB, tfA) = dss_to_tf(delays, A, B, C, D)`.
    - Computes IR via TF: `mtf_to_impz(tfB, tfA, ir_len)`.
    - Computes IR via delay state-space: `dss_to_impz(ir_len, delays, A, B, C, D)`.
    - Plots poles (from `roots(tfA)`) and one channel IR (TF vs DSS), and asserts the two IRs match.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import pyFDN

    np.random.seed(5)
    fs = 48000
    impulse_response_length = fs // 100

    # Define FDN
    N = 4
    num_input = 3
    num_output = 2
    B = np.eye(N, num_input)
    C = np.eye(num_output, N)
    D = np.random.randn(num_output, num_input)
    delays = np.random.randint(50, 101, size=N)
    A = pyFDN.random_orthogonal(N)
    return A, B, C, D, delays, impulse_response_length, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transfer function path
    """)
    return


@app.cell
def _(A, B, C, D, delays, impulse_response_length, pyFDN):
    # Delay state-space -> transfer function -> impulse response
    tfB, tfA = pyFDN.dss_to_tf(delays, A, B, C, D)
    ir_tf = pyFDN.mtf_to_impz(tfB, tfA, impulse_response_length)
    # poles_tf = np.roots(np.flip(tfA))  # tfA is z^{-1} ordering; roots() expects descending powers
    return (ir_tf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Delay state-space path
    """)
    return


@app.cell
def _(A, B, C, D, delays, impulse_response_length, pyFDN):
    # Delay state-space -> direct impulse response (time-domain simulation)
    ir_dss = pyFDN.dss_to_impz(impulse_response_length, delays, A, B, C, D)
    # ir_dss shape: (ir_len, num_output, num_input)
    return (ir_dss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots
    """)
    return


@app.cell
def _(ir_dss, ir_tf, mo, pyFDN):
    fig1 = pyFDN.plot_impulse_response_matrix(t=None,ir=ir_tf)
    fig2 = pyFDN.plot_impulse_response_matrix(t=None,ir=ir_dss)

    mo.vstack([fig1, fig2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verification
    """)
    return


@app.cell
def _(ir_dss, ir_tf, pyFDN):
    # Test: impulse response from TF matches delay state-space simulation
    assert pyFDN.is_almost_zero(ir_dss - ir_tf, tol=1e-10), "IR from TF and DSS should match"
    return


if __name__ == "__main__":
    app.run()
