# gallery_category: Getting Started
# gallery_description: Build a basic FLAMO FDN, inspect its response, and process a dry audio signal through it.

import marimo

__generated_with = "0.24.0"
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature 3: Mel-Spectrogram
    """)
    return


@app.cell
def _(fs, hop_length, ir, n_fft, pyFDN, torch):
    mel_spectrogram = pyFDN.MelSpectrogramFeature(
        sample_rate=fs,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128,
        f_min=0,
        f_max=fs / 2,
        log=True
    )

    mel_spec = mel_spectrogram(torch.from_numpy(ir))
    print(f"Mel-spectrogram shape: {mel_spec.shape}")
    print(f"Value range (log): [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")
    return (mel_spec,)


@app.cell
def _(mel_spec, mo, plt):
    fig2, ax2 = plt.subplots(figsize=(12, 5))

    im2 = ax2.pcolormesh(mel_spec.cpu().numpy(), shading="auto", cmap="magma")
    ax2.set_xlabel("Time frame")
    ax2.set_ylabel("Mel frequency bin")
    ax2.set_title("Mel-Spectrogram (Log-scale)")
    fig2.colorbar(im2, ax=ax2, label="Log Power")
    plt.tight_layout()

    mo.vstack([
        mo.md("### Mel-Spectrogram Visualization"),
        fig2
    ])
    return


@app.cell
def _(hop_length, ir, mo, n_fft, plt, pyFDN, torch):
    # Spectral flatness feature (library implementation)
    spec_flat =  pyFDN.spectral_flatness(
        torch.from_numpy(ir),
        n_fft=n_fft,
        hop_length=hop_length,
        eps=1e-10,
    )


    print(f"Spectral flatness shape: {spec_flat.shape}")
    print(f"Mean spectral flatness: {spec_flat.mean().item():.4f}")
    print(f"Range: [{spec_flat.min().item():.4f}, {spec_flat.max().item():.4f}]")

    fig_sf, ax_sf = plt.subplots(figsize=(12, 4))
    ax_sf.plot(spec_flat.detach().cpu().numpy(), linewidth=2, color="tab:green")
    ax_sf.set_xlabel("Time frame")
    ax_sf.set_ylabel("Spectral flatness")
    ax_sf.set_title("Spectral Flatness over Time")
    ax_sf.set_ylim(0, 1.05)
    ax_sf.grid(alpha=0.3)
    plt.tight_layout()

    mo.vstack([
        mo.md("### Spectral Flatness Visualization"),
        fig_sf
    ])

    return


@app.cell
def _(hop_length, ir, mo, n_fft, plt, pyFDN, torch):

    edr_db = pyFDN.energy_decay_relief(
        torch.from_numpy(ir),
        n_fft=n_fft,
        hop_length=hop_length,
        eps=1e-10,
    )

    print(f"Energy decay relief shape: {edr_db.shape}")
    print(f"Value range: [{edr_db.min().item():.4f}, {edr_db.max().item():.4f}]")
    print(f"Mean energy decay relief: {edr_db.mean().item():.4f}")



    fig_edr, ax_edr = plt.subplots(figsize=(12, 5))
    im_edr = ax_edr.pcolormesh(edr_db.detach().cpu().numpy(), shading="auto", cmap="viridis")
    ax_edr.set_xlabel("Time frame")
    ax_edr.set_ylabel("Decay band")
    ax_edr.set_title("Energy Decay Relief (dB)")
    fig_edr.colorbar(im_edr, ax=ax_edr, label="dB")
    plt.tight_layout()

    mo.vstack([
        mo.md("### Energy Decay Relief Visualization"),
        fig_edr
    ])
    return


if __name__ == "__main__":
    app.run()
