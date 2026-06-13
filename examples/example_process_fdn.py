# gallery_category: Getting Started

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
    # process_fdn — Pure DSS Simulation

    Demonstrates `pyFDN.process_fdn` for time-domain simulation of a feedback delay network with static matrices.
    A dry audio signal is run through the FDN to produce reverberation.

    For FDNs with absorption filters or learnable parameters, use the FLAMO path (`dss_to_flamo`).
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
    ## Load dry audio
    """)
    return


@app.cell
def _(mo, pyFDN):
    dry, fs = pyFDN.load_audio("synth_dry.wav")

    print(f"Loaded {len(dry)} samples at {fs} Hz ({len(dry) / fs:.2f} s)")
    mo.vstack([mo.audio(dry, fs)])
    return dry, fs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define FDN parameters
    """)
    return


@app.cell
def _(fs, np, pyFDN):
    delays = pyFDN.ms_to_smp(np.array([20, 27, 31, 37, 43, 53, 61, 71]), fs)
    build = pyFDN.fdn_build_gallery(
        fs=fs,
        delays=delays,
        io_type="normalized",
        direct_gain=0.0,
        rt=None,
        rng=0,
    )
    # Bake delay-proportional broadband decay into the lossless feedback matrix.
    g = pyFDN.rt_to_gain_per_sample(2.0, fs)
    delays = build.delays
    A = np.diag(g**delays) @ build.A
    B, C, D = build.B, build.C, build.D

    print(f"Delays: {delays} samples, gain per sample: {g:.6f}")
    return A, B, C, D, delays


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run dry signal through the FDN
    """)
    return


@app.cell
def _(A, B, C, D, delays, dry, fs, mo, pyFDN):
    wet = pyFDN.process_fdn(dry, delays, A, B, C, D)
    wet = pyFDN.peak_normalize(wet)

    print(f"Output shape: {wet.shape}")
    mo.vstack([mo.audio(wet, fs)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot impulse response
    """)
    return


@app.cell
def _(A, B, C, D, delays, fs, pyFDN):
    ir = pyFDN.dss_to_impz(4 * fs, delays, A, B, C, D)[:, 0, 0]
    pyFDN.plot_impulse_response(ir, fs=fs)
    return (ir,)


@app.cell
def _(fs, ir, pyFDN):
    pyFDN.plot_edc(ir, fs=fs)
    return


if __name__ == "__main__":
    app.run()
