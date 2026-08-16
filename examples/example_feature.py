# gallery_category: Getting Started
# gallery_description: Build a basic FLAMO FDN, inspect its response, and process a dry audio signal through it.

import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Feature
    """)
    return


@app.cell
def _():
    import numpy as np
    import torch

    import pyFDN

    return pyFDN, torch


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
    n_fft = 2048
    hop_length = 512
    print(f"n={n}, fs={fs} Hz, n_fft={n_fft}, hop_length={hop_length}")
    return fs, hop_length, n, n_fft


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
        rt=2.0,
        rt_nyquist=0.5,
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
    ).to("cpu")
    ir = pyFDN.flamo_time_response(model).squeeze()
    return (ir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature 1: STFT Magnitude
    """)
    return


@app.cell
def _(hop_length, ir, n_fft, pyFDN, torch):
    stft_mag = pyFDN.stft_magnitude(
        torch.from_numpy(ir),
        n_fft=n_fft,
        hop_length=hop_length
    )
    print(f"STFT magnitude shape: {stft_mag.shape}")
    print(f"Value range: [{stft_mag.min():.4f}, {stft_mag.max():.4f}]")

    return (stft_mag,)


@app.cell
def _(mo, stft_mag, torch):
    import matplotlib.pyplot as plt
    # Plot STFT magnitude in dB
    fig, ax = plt.subplots(figsize=(12, 5))

    db = 20 * torch.log10(torch.clamp(stft_mag, min=1e-10))
    im = ax.pcolormesh(db.cpu().numpy(), shading="auto", cmap="viridis")

    ax.set_xlabel("Time frame")
    ax.set_ylabel("Frequency bin")
    ax.set_title("STFT Magnitude Spectrogram (dB)")
    fig.colorbar(im, ax=ax, label="dB")
    plt.tight_layout()

    mo.vstack([
        mo.md("### STFT Magnitude Visualization"),
        fig
    ])
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature 2: STFT Phase
    """)
    return


@app.cell
def _(hop_length, ir, n_fft, pyFDN, torch):
    stft_phase = pyFDN.stft_phase(
        torch.from_numpy(ir),
        n_fft=n_fft,
        hop_length=hop_length
    )

    print(f"STFT phase shape: {stft_phase.shape}")
    print(f"Value range (radians): [{stft_phase.min():.4f}, {stft_phase.max():.4f}]")

    return (stft_phase,)


@app.cell
def _(mo, plt, stft_phase):

    fig1, ax1 = plt.subplots(figsize=(12, 5))

    im1 = ax1.pcolormesh(stft_phase.cpu().numpy(), shading="auto", cmap="hsv")
    ax1.set_xlabel("Time frame")
    ax1.set_ylabel("Frequency bin")
    ax1.set_title("STFT Phase (radians)")
    fig1.colorbar(im1, ax=ax1, label="Radians")
    plt.tight_layout()

    mo.vstack([
        mo.md("### STFT Phase Visualization"),
        fig1
    ])
    return


if __name__ == "__main__":
    app.run()
