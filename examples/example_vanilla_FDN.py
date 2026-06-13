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

    Build a vanilla FDN with `pyFDN.dss_to_flamo`, optionally alter delays and feedforward (e.g. diagonal gain, no absorption), plot IRs, and run a dry signal through the model.
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
    ## Parameters
    """)
    return


@app.cell
def _(torch):
    torch.manual_seed(42)
    n = 8
    fs = 48000
    print(f"n={n}, fs={fs}")
    return fs, n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build model and get original IR
    """)
    return


@app.cell
def _(fs, n, pyFDN):
    build = pyFDN.fdn_build_gallery(
        n,
        fs=fs,
        io_type="ones",
        direct_gain=1.0,
        rt60=2.0,
        rt60_nyquist=0.5,
        post_eq_db_dc=0.0,
        post_eq_db_nyquist=-6.0,
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
        sos_filter=build.filters,
        output_filter=build.post_eq,
    )
    ir_original = pyFDN.flamo_time_response(model).flatten()
    return build, ir_original, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FDN parameter overview
    """)
    return


@app.cell
def _(build, pyFDN):
    pyFDN.plot_FDN_build(build, title="Vanilla FDN parameters")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot and listen to IRs
    """)
    return


@app.cell
def _(fs, ir_original, mo, np, pyFDN):
    _fig = pyFDN.plot_impulse_response(
        ir_original,
        fs=fs,
        labels=["Original"],
        title="Vanilla FDN impulse response",
    )

    mo.vstack(
        [
            _fig,
            mo.md("Original:"),
            mo.audio(np.asanyarray(ir_original), fs),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Process dry audio

    Load packaged synth dry, trim to one block of length `n_fft`, run through the model, listen to dry and wet.
    """)
    return


@app.cell
def _(fs, mo, model, np, pyFDN):
    dry, _ = pyFDN.load_audio("synth_dry.wav", fs=fs)
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
