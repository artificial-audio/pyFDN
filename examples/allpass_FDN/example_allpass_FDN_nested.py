# gallery_category: Allpass FDNs
# gallery_title: Gardner's nested allpass FDN
# gallery_description: Recreate Gardner's SISO reverberator by iteratively nesting feedforward and feedback allpass sections.

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
    # Gardner's Nested Allpass FDN

    Example for the nested allpass structure: an FDN built by iteratively nesting a feedforward/back allpass around the previous system. SISO (single input, single output).

    **Reference:** *{pyFDN.paper_link("Gardner1992RealtimeMultichannelRoom")}*.

    See also: {pyFDN.paper_link("Allpass_Feedback_Delay_Networks")}.
    """)
    return


@app.cell
def _():
    import numpy as np

    import pyFDN

    np.random.seed(42)
    fs = 48000
    return fs, np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build nested allpass FDN

    `nested_allpass` takes one gain per nesting stage and returns the delay state-space system that realises the whole nest. The delays are drawn at random — nesting does not constrain them, since the structure is uniallpass.
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
    ## The system matrix

    `[A, B; C, D]` as one block heatmap. Nesting shows up as structure in `A`: each section's feedback sits on the diagonal, and the coupling into the section it wraps sits just off it.
    """)
    return


@app.cell
def _(A, B, C, D, pyFDN):
    pyFDN.plot_system_matrix(A, B, C, D)
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
    is_uniallpass, _P = pyFDN.is_uniallpass(A, B, C, D)
    assert is_uniallpass, "Expected uniallpass"
    print("Uniallpass: OK")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impulse response

    Four seconds of the nested cascade. Nesting an allpass inside another one multiplies their echo patterns rather than concatenating them, which is what buys Gardner's structure its density from so few sections.
    """)
    return


@app.cell
def _(A, B, C, D, delays, fs, pyFDN):
    impulse_response = pyFDN.dss_to_impz(4 * fs, delays, A, B, C, D).squeeze()

    pyFDN.plot_impulse_response(
        impulse_response,
        fs=fs,
        title="Nested allpass FDN — impulse response",
    )
    return (impulse_response,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spectrogram

    Flat on average — it is allpass — but the time structure the ear hears as ringing is visible here and nowhere in the magnitude response.
    """)
    return


@app.cell
def _(fs, impulse_response, mo, pyFDN):
    _fig = pyFDN.plot_spectrogram(
        impulse_response, fs, title="Nested allpass FDN — spectrogram"
    )

    mo.vstack([_fig, mo.audio(impulse_response, fs)])
    return


if __name__ == "__main__":
    app.run()
