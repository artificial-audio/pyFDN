# gallery_category: Allpass FDNs
# gallery_title: Homogeneous allpass FDN (SISO)
# gallery_description: Build a single-input, single-output homogeneous allpass FDN and validate its allpass response.

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
    # Homogeneous allpass FDN (SISO)

    Example for an allpass FDN with **homogeneous decay** so that all poles have the same decay rate.

    See {pyFDN.paper_link("Allpass_Feedback_Delay_Networks")}.

    """)
    return


@app.cell
def _():
    import numpy as np

    import pyFDN

    np.random.seed(1)
    fs = 48000
    return fs, np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build homogeneous allpass FDN

    Random delays, gain matrix **G**, random admissible diagonal **X**, then **homogeneous_allpass_fdn(G, X)**.
    """)
    return


@app.cell
def _(fs, np, pyFDN):
    N = 6
    delays = np.random.randint(300, 700, size=N)
    g = pyFDN.rt_to_gain_per_sample(0.5, fs)
    G = np.diag(g**delays)

    X = pyFDN.rand_admissible_homogeneous_allpass(G, (0.7, 0.99))
    A, b, c, d, _U = pyFDN.homogeneous_allpass_fdn(G, X, verbose=False)
    return A, b, c, d, delays


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: uniallpass

    Check that the system is uniallpass, i.e., allpass for any delays.
    """)
    return


@app.cell
def _(A, b, c, d, pyFDN):
    is_a, _ = pyFDN.is_uniallpass(A, b, c, d, tol=1e-7)
    assert is_a, "Expected allpass for homogeneous FDN with these delays"
    print("is_allpass: OK")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot system matrix

    Visualize **[A, b; c, d]** as 2×2 block heatmaps.
    """)
    return


@app.cell
def _(A, b, c, d, pyFDN):
    pyFDN.plot_system_matrix(A, b, c, d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impulse response

    Compute the impulse response with **dss_to_impz** (SISO: single input/output).
    """)
    return


@app.cell
def _(A, b, c, d, delays, fs, pyFDN):
    impulse_response = pyFDN.dss_to_impz(2 * fs, delays, A, b, c, d).squeeze()
    return (impulse_response,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time domain and play

    Plot the impulse response in the time domain and use the audio widget to play it.
    """)
    return


@app.cell
def _(fs, impulse_response, mo, pyFDN):
    _fig = pyFDN.plot_impulse_response(
        impulse_response,
        fs=fs,
        title="Homogeneous allpass FDN — impulse response",
    )

    mo.vstack([_fig, mo.audio(impulse_response, fs)])
    return


if __name__ == "__main__":
    app.run()
