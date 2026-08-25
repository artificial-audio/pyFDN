# gallery_category: Allpass FDNs
# gallery_title: Schroeder allpass reverberator
# gallery_description: Build the classic Schroeder series-allpass reverberator and verify that it is uniallpass -- allpass whatever the delays.

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
    # Schroeder's Series Allpass FDN

    Example for Schroeder's series (cascade) allpass: a cascade of first-order allpass sections realized as an FDN with diagonal feedback matrix. SISO.

    **Reference:** *{pyFDN.paper_link("Schroeder1961ColorlessArtificialReverberation")}*.

    See also: {pyFDN.paper_link("Allpass_Feedback_Delay_Networks")}.

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
    fs = 48000
    delays = np.random.randint(200, 1000, size=N)

    A, B, C, D = pyFDN.series_allpass(g)
    return A, B, C, D, delays, fs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The system matrix

    `[A, B; C, D]` as one block heatmap. `series_allpass` puts the gains on the diagonal of `A` and the feedforward taps in the strictly triangular part — the cascade structure read off directly.
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

    Two seconds of the cascade's response. Each section recirculates at its own delay, and because the six delays are unequal their echo patterns interleave instead of landing on top of each other, so the response fills in over time.
    """)
    return


@app.cell
def _(A, B, C, D, delays, fs, pyFDN):
    impulse_response = pyFDN.dss_to_impz(2 * fs, delays, A, B, C, D).squeeze()

    pyFDN.plot_impulse_response(
        impulse_response,
        fs=fs,
        title="Schroeder series allpass — impulse response",
    )
    return (impulse_response,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spectrogram

    The flat magnitude response is the allpass property; what the ear objects to is the structure *in time*, which the spectrogram shows and the magnitude response cannot.
    """)
    return


@app.cell
def _(fs, impulse_response, mo, pyFDN):
    _fig = pyFDN.plot_spectrogram(
        impulse_response, fs, title="Schroeder series allpass — spectrogram"
    )

    mo.vstack([_fig, mo.audio(impulse_response, fs)])
    return


if __name__ == "__main__":
    app.run()
