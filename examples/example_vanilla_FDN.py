# gallery_category: Getting Started
# gallery_description: Build a basic FLAMO FDN, inspect its response, and process a dry audio signal through it.

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
    # Vanilla FDN (FLAMO)

    The shortest path from nothing to a reverberator you can listen to. `pyFDN.fdn_build_gallery` picks a complete set of FDN parameters — delays, an orthogonal feedback matrix, input/output gains and frequency-dependent absorption — and `pyFDN.dss_to_flamo` turns them into a FLAMO model, which is a differentiable torch module that also happens to render audio.

    Two things come out of that model here: its impulse response, which is the reverb on its own, and a dry recording pushed through it.
    """)
    return


@app.cell
def _():
    import numpy as np
    import torch

    import pyFDN

    return np, pyFDN, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build the model

    Eight delay lines at 48 kHz, decaying over 2 s at DC and half that at Nyquist — the frequency-dependent absorption every real room has. `dss_to_flamo` takes the build's matrices, delays and per-line filters and returns the FLAMO model; `flamo_time_response` renders its impulse response.
    """)
    return


@app.cell
def _(pyFDN, torch):
    torch.manual_seed(42)
    n = 8
    fs = 48000

    build = pyFDN.fdn_build_gallery(
        n,
        fs=fs,
        io_type="ones",
        direct_gain=1.0,
        rt=2.0,
        rt_nyquist=0.5,
        output_gain_db=0.0,
        output_gain_db_nyquist=-6.0,
        rng=42,
    )
    model = pyFDN.dss_to_flamo(
        build.A,
        build.B,
        build.C,
        build.D,
        build.delays,
        build.fs,
        nfft=2**18,
        post_delay=build.post_delay,
        post_output=build.post_output,
    )
    ir = pyFDN.flamo_time_response(model).flatten()
    return build, fs, ir, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the model is made of

    Feedback matrix, delays, input and output gains, and the magnitude response the absorption filters impose on each delay line.
    """)
    return


@app.cell
def _(build, pyFDN):
    pyFDN.plot_FDN_build(build, title="Vanilla FDN parameters")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The impulse response

    A dense exponential decay whose high end dies first, which is the audible signature of the Nyquist reverberation time being shorter than the one at DC.
    """)
    return


@app.cell
def _(fs, ir, mo, np, pyFDN):
    _fig = pyFDN.plot_impulse_response(
        ir,
        fs=fs,
        title="Vanilla FDN impulse response",
    )

    mo.vstack([_fig, mo.audio(np.asanyarray(ir), fs)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run audio through it

    The same model, driven by a signal instead of an impulse. `flamo_process` renders the model's frequency response once and convolves; `tail_seconds` appends enough silence for the tail to finish rather than wrapping back onto the start.
    """)
    return


@app.cell
def _(fs, mo, model, np, pyFDN):
    dry, _ = pyFDN.load_audio("synth_dry", fs=fs)
    # Reserve 2 s of trailing silence so the reverb tail does not wrap around.
    wet = pyFDN.flamo_process(model, dry, fs=fs, tail_seconds=2.0)

    mo.hstack(
        [
            mo.md("Dry:"),
            mo.audio(np.asanyarray(dry), fs),
            mo.md("Wet:"),
            mo.audio(np.asarray(wet), fs),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
