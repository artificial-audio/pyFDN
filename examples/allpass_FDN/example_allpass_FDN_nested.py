# gallery_category: Allpass FDN Examples

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
    # Gardner's Nested Allpass FDN

    Example for the nested allpass structure: an FDN built by iteratively nesting a feedforward/back allpass around the previous system. SISO (single input, single output).

    **Reference:** Gardner, W. G. (1992). *A real-time multichannel room simulator.* J. Acoust. Soc. Am. 92, 1–23.

    See also: *Allpass Feedback Delay Networks*, Sebastian J. Schlecht (IEEE Trans. Signal Processing).

    — Original MATLAB: Sebastian J. Schlecht, 7 June 2020
    """)
    return


@app.cell
def _():
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
    ## Build nested allpass FDN

    Use a vector of gains **g** (one per nesting stage). Delays are powers of two: **m = 2^(0:N-1)**.
    """)
    return


@app.cell
def _(np, pyFDN):
    N = 6
    g = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    delays = np.random.randint(200, 1000, size=N)

    A, B, C, D = pyFDN.nested_allpass(g)
    print("A shape:", A.shape)
    print("B shape:", B.shape)
    print("C shape:", C.shape)
    print("D shape:", D.shape)
    return A, B, C, D, delays


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
def _(A, B, C, D, delays, pyFDN):
    Fs = 48000
    ir_len = 4 * Fs  # 4 seconds
    impulse_response = pyFDN.dss_to_impz(ir_len, delays, A, B, C, D).squeeze()
    # Shape: (ir_len, n_out, n_in) -> (ir_len, ) for SISO
    return Fs, impulse_response, ir_len


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
        title="Nested allpass FDN — impulse response",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spectrogram (one channel)
    """)
    return


@app.cell
def _(Fs, impulse_response, mo, np, pyFDN):
    channel_ir = np.asarray(impulse_response).squeeze()
    _fig = pyFDN.plot_spectrogram(
        channel_ir, Fs, title="Nested allpass FDN — spectrogram"
    )
    _fig.show()

    mo.vstack([mo.audio(channel_ir, Fs)])
    return


if __name__ == "__main__":
    app.run()
