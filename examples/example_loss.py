import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch

    import pyFDN

    return mo, pyFDN, torch


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
    ir1 = pyFDN.flamo_time_response(model).squeeze()
    build1 = pyFDN.fdn_build_gallery(
        n,
        fs=fs,
        io_type="ones",
        direct_gain=1.0,
        rt=2.0,
        rt_nyquist=0.5,
        post_eq_db_dc=0.0,
        post_eq_db_nyquist=-6.0,
        rng=41,
    )
    model1 = pyFDN.dss_to_flamo(
        build1.A,
        build1.B,
        build1.C,
        build1.D,
        build1.delays,
        build1.fs,
        nfft=2**18,
        sos_filter=build1.filters,
        output_filter=build1.post_eq,
    ).to("cpu")
    ir2 = pyFDN.flamo_time_response(model1).squeeze()
    return ir1, ir2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MSE with STFT
    """)
    return


@app.cell
def _(hop_length, ir1, ir2, n_fft, pyFDN, torch):
    # Compare STFT magnitudes of the two impulse responses created above
    stft_ir1 = pyFDN.stft_magnitude(
        torch.from_numpy(ir1),
        n_fft=n_fft,
        hop_length=hop_length,
    )
    # Compare STFT magnitudes of the two impulse responses created above
    stft_ir2 = pyFDN.stft_magnitude(
        torch.from_numpy(ir2),
        n_fft=n_fft,
        hop_length=hop_length,
    )

    mse_loss = pyFDN.mse.MSELoss(reduction="mean")
    loss_stft = mse_loss(stft_ir1, stft_ir2)

    print(f"STFT magnitude MSE: {loss_stft.item():.8f}")
    return stft_ir1, stft_ir2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## L1 Loss function
    """)
    return


@app.cell
def _(pyFDN, stft_ir1, stft_ir2):
    # Compare STFT magnitudes of the two impulse responses created above

    l1_loss = pyFDN.l1.L1Loss(reduction="mean")
    loss_l1 = l1_loss(stft_ir1, stft_ir2)

    print(f"l1 magnitude MSE: {loss_l1.item():.8f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Asymmetric Loss
    """)
    return


@app.cell
def _(pyFDN, stft_ir1):
    asym_loss = pyFDN.asymetricLoss.AsymmetricLoss()
    loss_asym = asym_loss(stft_ir1)
    print(f"Asymmetric loss: {loss_asym.item():.8f}")
    return


if __name__ == "__main__":
    app.run()
