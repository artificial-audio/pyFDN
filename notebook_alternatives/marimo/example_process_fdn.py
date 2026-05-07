import marimo

__generated_with = "0.23.4"
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

    For FDNs with absorption filters or learnable parameters, use the FLAMO path (`vanilla_FDN`, `dss_to_flamo`).
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from importlib.resources import files
    import soundfile as sf
    from IPython.display import Audio, display

    import pyFDN

    return Audio, display, files, np, plt, pyFDN, sf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load dry audio
    """)
    return


@app.cell
def _(Audio, display, files, sf):
    path = files("pyFDN.audio") / "synth_dry.wav"
    with path.open("rb") as f:
        dry, fs = sf.read(f, dtype="float64")
    dry = dry[:, 0] if dry.ndim > 1 else dry

    print(f"Loaded {len(dry)} samples at {fs} Hz ({len(dry)/fs:.2f} s)")
    display(Audio(dry, rate=fs))
    return dry, fs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define FDN parameters
    """)
    return


@app.cell
def _(fs, np, pyFDN):
    N = 8
    delays = pyFDN.ms_to_smp(np.array([20, 27, 31, 37, 43, 53, 61, 71]), fs)
    rt60 = 2.0  # reverberation time in seconds

    U = pyFDN.random_orthogonal(N)              # N×N orthogonal mixing matrix
    g = pyFDN.rt_to_gain_per_sample(rt60, fs)  # broadband gain per sample for RT60
    G = np.diag(g ** delays)                   # delay-proportional attenuation
    A = G @ U                                  # feedback matrix: decay × mixing

    B = np.ones((N, 1)) / np.sqrt(N)    # N×1 input gain
    C = np.ones((1, N)) / np.sqrt(N)    # 1×N output gain
    D = np.zeros((1, 1))                # no direct path

    print(f"Delays: {delays} samples, gain per sample: {g:.6f}")
    return A, B, C, D, delays


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run dry signal through the FDN
    """)
    return


@app.cell
def _(A, Audio, B, C, D, delays, display, dry, fs, pyFDN):
    wet = pyFDN.process_fdn(dry, delays, A, B, C, D)
    wet = pyFDN.peak_normalize(wet)

    print(f"Output shape: {wet.shape}")
    display(Audio(wet, rate=fs))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot impulse response
    """)
    return


@app.cell
def _(A, B, C, D, delays, fs, np, plt, pyFDN):
    ir = pyFDN.dss_to_impz(4 * fs, delays, A, B, C, D)[:, 0, 0]
    t = np.arange(len(ir)) / fs

    fig, axes = plt.subplots(2, 1, figsize=(10, 5))

    axes[0].plot(t, pyFDN.mulaw_encode(ir), lw=0.5)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (μ-law)")
    axes[0].set_title("Impulse Response")

    axes[1].plot(t, pyFDN.edc(ir))
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("EDC (dB)")
    axes[1].set_title("Energy Decay Curve")

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
