# gallery_category: Absorption & Filters

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
    # First-order absorption FDN (FLAMO integration)

    This example builds an FDN with **first-order shelving absorption filters** (designed with pyFDN) and runs it in **FLAMO** for the impulse response.

    **What it does:**
    - Designs first-order shelving absorption coefficients with `pyFDN.first_order_absorption` (T60 at DC and Nyquist, shelf crossover at fs/8 by default), following Jot (2015), *Proportional parametric equalizers*.
    - Converts the delay state-space (A, B, C, D, m) to a FLAMO model with `pyFDN.dss_to_flamo` (absorption SOS in the loop) and computes the IR.
    - Plots the FDN parameter overview, IR (mu-law compressed), and spectrogram.
    """)
    return


@app.cell
def _():
    import numpy as np

    import pyFDN

    return np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FDN parameters

    Delay lengths, reverberation time targets, a random orthogonal feedback matrix, and input/output/direct gains.
    """)
    return


@app.cell
def _(np, pyFDN):
    np.random.seed(1)

    fs = 48000
    delays = np.array([797.0, 839.0, 1051.0, 1409.0, 1693.0, 1867.0, 2243.0, 2657.0])
    rt_dc = 2.0  # T60 at DC (seconds)
    rt_ny = 0.5  # T60 at Nyquist (seconds)
    build = pyFDN.fdn_build_gallery(
        fs=fs,
        delays=delays,
        io_type="ones",
        direct_gain=1.0,
        rt60=None,
        rng=1,
    )
    feedback_matrix = build.A
    B, C, D = build.B, build.C, build.D
    delays = build.delays

    print(f"N={delays.size}, fs={fs}, RT60 DC={rt_dc}s Nyquist={rt_ny}s")
    return B, C, D, delays, feedback_matrix, fs, rt_dc, rt_ny


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First-order absorption coefficients (pyFDN)

    Design SOS coefficients for each delay line from target T60 at DC and Nyquist. The shelf crossover frequency defaults to fs/8 (the midpoint of the warped frequency axis) and can be set explicitly via `crossover_frequency`.
    """)
    return


@app.cell
def _(delays, fs, pyFDN, rt_dc, rt_ny):
    sos = pyFDN.first_order_absorption(rt_dc, rt_ny, delays, fs)
    print("SOS shape (1, 6, N):", sos.shape)
    return (sos,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build FDN in FLAMO

    `pyFDN.dss_to_flamo` converts the delay state-space (A, B, C, D, m) to a FLAMO model, with the absorption SOS in the feedback loop (delay -> filter -> A) and the direct path D in parallel.
    """)
    return


@app.cell
def _(B, C, D, delays, feedback_matrix, fs, pyFDN, sos):
    nfft = 2**17

    model = pyFDN.dss_to_flamo(
        feedback_matrix,
        B,
        C,
        D,
        delays,
        fs,
        nfft=nfft,
        sos_filter=sos,  # canonical (1, 6, N) bank
    )
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FDN parameter overview
    """)
    return


@app.cell
def _(B, C, D, delays, feedback_matrix, fs, pyFDN, sos):
    pyFDN.plot_fdn_parameter(
        delays,
        feedback_matrix,
        B,
        C,
        D,
        attenuation_sos=sos,
        fs=fs,
        title="First-order absorption FDN",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate IR
    """)
    return


@app.cell
def _(model, pyFDN):
    ir_python = pyFDN.flamo_time_response(model).flatten()
    return (ir_python,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots

    IR (mu-law compressed), spectrogram, and audio playback.
    """)
    return


@app.cell
def _(fs, ir_python, pyFDN):
    pyFDN.plot_impulse_response(
        ir_python,
        fs=fs,
        labels=["FLAMO (Python)"],
    )
    return


@app.cell
def _(fs, ir_python, pyFDN):
    pyFDN.plot_spectrogram(ir_python, fs, title="Spectrogram (FLAMO)")
    return


@app.cell
def _(fs, ir_python, mo, np):
    mo.vstack(
        [
            mo.Html("Impulse response").style({"font-size": "2.0em"}),
            mo.audio(np.asarray(ir_python), rate=fs),
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
