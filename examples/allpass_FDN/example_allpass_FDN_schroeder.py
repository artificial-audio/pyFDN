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
    # Schroeder's Series Allpass FDN

    Example for Schroeder's series (cascade) allpass: a cascade of first-order allpass sections realized as an FDN with diagonal feedback matrix. SISO.

    **Reference:** Schroeder, M. R. & Logan, B. F. (1961). *"Colorless" artificial reverberation.* IRE Trans. Audio AU-9, 209–214.

    See also: *Allpass Feedback Delay Networks*, Sebastian J. Schlecht (IEEE Trans. Signal Processing).

    — Original MATLAB: Sebastian J. Schlecht, 7 June 2020
    """)
    return


@app.cell
def _():
    # TODO: add a dsp diagram of the schroeder allpass FDN to the intro
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

    np.random.seed(42)
    return np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build series allpass FDN

    Use a vector of gains **g** (one per section).
    """)
    return


@app.cell
def _(np, pyFDN):
    N = 6
    g = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    Fs = 48000
    delays = np.random.randint(200, 1000, size=N)

    A, B, C, D = pyFDN.series_allpass(g)
    return A, B, C, D, Fs, delays


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the system matrix
    """)
    return


@app.cell
def _(A, B, C, D, pyFDN):
    _fig = pyFDN.plot_system_matrix(A, B, C, D)
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: uniallpass

    Check that the FDN is uniallpass (lossless with diagonal Lyapunov matrix).
    """)
    return


@app.cell
def _(A, B, C, D, pyFDN):
    is_a, P = pyFDN.is_uniallpass(A, B, C, D)
    assert is_a, "Expected uniallpass"
    print("Uniallpass: OK")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impulse response

    Render the IR and plot (SISO: single channel).
    """)
    return


@app.cell
def _(A, B, C, D, Fs, delays, pyFDN):
    ir_len = 2 * Fs
    impulse_response = pyFDN.dss_to_impz(ir_len, delays, A, B, C, D).squeeze()
    return impulse_response, ir_len


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impulse response plot
    """)
    return


@app.cell
def _(Fs, impulse_response, pyFDN):
    pyFDN.plot_impulse_response(
        impulse_response,
        fs=Fs,
        title="Schroeder series allpass — impulse response",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spectrogram and play
    """)
    return


@app.cell
def _(Fs, impulse_response, mo, np, pyFDN):
    channel_ir = np.asarray(impulse_response).squeeze()
    _fig = pyFDN.plot_spectrogram(
        channel_ir, Fs, title="Schroeder series allpass — spectrogram"
    )
    _fig.show()

    mo.vstack([mo.audio(channel_ir, Fs)])
    return


if __name__ == "__main__":
    app.run()
