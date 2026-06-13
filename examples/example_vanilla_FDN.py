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
    # Vanilla FDN (FLAMO)

    Build a vanilla FDN with `pyFDN.dss_to_flamo`, optionally alter delays and feedforward (e.g. diagonal gain, no absorption), plot IRs, and run a dry signal through the model.
    """)
    return


@app.cell
def _():
    from collections import OrderedDict

    import numpy as np
    import torch
    from flamo.processor import dsp, system

    import pyFDN

    return OrderedDict, dsp, np, pyFDN, system, torch


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
def _(fs, n, np, pyFDN):
    # Vanilla FDN: random delays + orthogonal feedback matrix + first-order absorption.
    _delays = np.random.randint(400, 1200, size=n).astype(np.float64)
    _sos = pyFDN.first_order_absorption(2.0, 0.5, _delays, fs=fs)
    model = pyFDN.dss_to_flamo(
        pyFDN.random_orthogonal(n),
        np.ones((n, 1)),
        np.ones((1, n)),
        np.ones((1, 1)),
        _delays,
        fs,
        nfft=2**18,
        sos_filter=_sos,
    )
    ir_original = model.get_time_response().flatten()
    return ir_original, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Alter model (optional)

    Change delays, replace feedforward with delay + diagonal gain (no absorption), then recompute IR.
    """)
    return


@app.cell
def _(OrderedDict, dsp, model, n, np, system, torch):
    core = model.get_core()
    fdn = core.branchA
    feedback_loop = fdn.feedback_loop
    delay_module = feedback_loop.feedforward.delay

    new_delays = np.linspace(1100, 1150, n, dtype=np.int64)
    new_delays_t = torch.tensor(
        new_delays, dtype=torch.float32, device=next(delay_module.parameters()).device
    )
    delay_module.assign_value(delay_module.sample2s(new_delays_t))

    _n_fft = model.get_inputLayer().nfft
    device = next(delay_module.parameters()).device
    diagonal_gain = dsp.Gain(size=(n, n), nfft=_n_fft, device=device)
    diagonal_gain.assign_value(torch.diag(0.9999**new_delays_t))
    feedback_loop.feedforward = system.Series(
        OrderedDict({"delay": delay_module, "diagonal_gain": diagonal_gain})
    )

    ir_altered = model.get_time_response().flatten()
    return (ir_altered,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot and listen to IRs
    """)
    return


@app.cell
def _(fs, ir_altered, ir_original, mo, np, pyFDN):
    _fig = pyFDN.plot_impulse_response(
        ir_original,
        ir_altered,
        fs=fs,
        labels=["Original", "Altered"],
        title="Vanilla FDN impulse response",
    )

    mo.vstack(
        [
            _fig,
            mo.md("Original:"),
            mo.audio(np.asanyarray(ir_original), fs),
            mo.md("Altered:"),
            mo.audio(np.asarray(ir_altered), fs),
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
def _(fs, mo, model, np, torch):
    from importlib.resources import files

    import soundfile as sf

    try:
        path = files("pyFDN.audio") / "synth_dry.wav"
        with path.open("rb") as f:
            dry, file_fs = sf.read(f, dtype="float64")
    except Exception:
        raise FileNotFoundError("pyFDN.audio/synth_dry.wav not found.") from None
    dry = dry[:, 0] if dry.ndim > 1 else dry
    if file_fs != fs:
        from scipy.signal import resample

        dry = resample(dry, int(len(dry) * fs / file_fs))

    _n_fft = model.get_inputLayer().nfft
    dry = dry[: int(_n_fft)]
    dry[-(fs * 2) :] = 0
    x = torch.tensor(dry, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():  # pad with zeros to avoid wrap around
        wet = model(x).squeeze().cpu().numpy()

    mo.vstack(
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
