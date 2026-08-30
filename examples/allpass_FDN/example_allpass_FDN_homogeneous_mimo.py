# gallery_category: Allpass FDNs
# gallery_title: Homogeneous allpass FDN (MIMO)
# gallery_description: Construct and verify a multi-input, multi-output homogeneous allpass FDN from delay-line gains and an orthogonal mixing matrix.

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo, pyFDN):
    mo.md(f"""
    # Homogeneous allpass FDN (MIMO)

    Example for an allpass FDN with **homogeneous decay** so that all poles have the same decay rate. Compared to the SISO case, the MIMO has considerably more degrees of freedom.

    See {pyFDN.paper_link("Allpass_Feedback_Delay_Networks")}.

    """)
    return


@app.cell
def _():
    import numpy as np

    import pyFDN

    np.random.seed(1)
    return np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build homogeneous allpass FDN

    Random delays, gain matrix **G**, random mixing matrix **U**, combined to the feedback matrix **A**. The remaining coefficients are reconstructed by completing the allpass FDN.
    """)
    return


@app.cell
def _(np, pyFDN):
    fs = 48000
    N = 8
    numio = N

    delays = np.random.randint(800, 1800, size=N)
    g = pyFDN.rt_to_gain_per_sample(0.6, fs)
    G = np.diag(g**delays)
    U = pyFDN.random_orthogonal(N)
    A = G @ U

    B, C, D, X = pyFDN.complete_fdn(A, N, str(numio))
    return A, B, C, D, fs, delays


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: uniallpass

    Check that the system is uniallpass, i.e., allpass for any delays.
    """)
    return


@app.cell
def _(A, B, C, D, pyFDN):
    is_a, _ = pyFDN.is_uniallpass(A, B, C, D, tol=1e-7)
    assert is_a, "Expected allpass for homogeneous FDN with these delays"
    print("is_allpass: OK")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The system matrix

    `[A, B; C, D]` as one block heatmap. Losslessness is a property of the whole matrix, not of `A` alone, which is why it is worth seeing the four blocks together.
    """)
    return


@app.cell
def _(A, B, C, D, pyFDN):
    pyFDN.plot_system_matrix(A, B, C, D)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impulse response

    One second of the response, taken from output 3 driven by input 2. Every input/output pair of a MIMO allpass FDN is a different filter; the allpass property belongs to the matrix as a whole, so no single pair has to look special.
    """)
    return


@app.cell
def _(A, B, C, D, delays, fs, mo, pyFDN):
    impulse_response = pyFDN.dss_to_impz(fs, delays, A, B, C, D)
    ir_channel = impulse_response[:, 2, 1]

    _fig = pyFDN.plot_impulse_response(
        ir_channel,
        fs=fs,
        title="Homogeneous allpass FDN — impulse response",
    )

    mo.vstack([_fig, mo.audio(ir_channel, fs)])
    return


if __name__ == "__main__":
    app.run()
