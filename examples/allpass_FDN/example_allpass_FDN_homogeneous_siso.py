# gallery_category: Allpass FDN Examples

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
    # Homogeneous allpass FDN (SISO)

    Example for an allpass FDN with **homogeneous decay** so that all poles have the same decay rate.

    See *Allpass Feedback Delay Networks*, Sebastian J. Schlecht (IEEE Trans. Signal Processing).

    — Original MATLAB: Sebastian J. Schlecht, 10 June 2020
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

    import pyFDN

    np.random.seed(1)
    return np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build homogeneous allpass FDN

    Random delays, gain matrix **G**, random admissible diagonal **X**, then **homogeneous_allpass_fdn(G, X)**.
    """)
    return


@app.cell
def _(np, pyFDN):
    Fs = 48000
    N = 6
    delays = np.random.randint(300, 700, size=N)  # delays in samples, 1..30
    g = pyFDN.rt_to_gain_per_sample(0.5, Fs)
    G = np.diag(g**delays)  # gain matrix

    X = pyFDN.rand_admissible_homogeneous_allpass(G, (0.7, 0.99))
    X @ G @ G

    A, b, c, d, U = pyFDN.homogeneous_allpass_fdn(G, X, verbose=False)
    return A, Fs, b, c, d, delays


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
    _fig = pyFDN.plot_system_matrix(A, b, c, d)
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impulse response

    Compute the impulse response with **dss_to_impz** (SISO: single input/output).
    """)
    return


@app.cell
def _(A, Fs, b, c, d, delays, pyFDN):
    ir_len = 2 * Fs  # 2 seconds
    impulse_response = pyFDN.dss_to_impz(ir_len, delays, A, b, c, d).squeeze()
    return (impulse_response,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time domain and play

    Plot the impulse response in the time domain and use the audio widget to play it.
    """)
    return


@app.cell
def _(Fs, impulse_response, mo, pyFDN):
    _fig = pyFDN.plot_impulse_response(
        impulse_response,
        fs=Fs,
        title="Homogeneous allpass FDN — impulse response (time domain)",
    )
    _fig.show()

    mo.vstack([mo.audio(impulse_response, Fs)])
    return


if __name__ == "__main__":
    app.run()
