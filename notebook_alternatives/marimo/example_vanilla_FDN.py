import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Vanilla FDN (FLAMO)

    Build a vanilla FDN with `pyFDN.vanilla_FDN`, optionally alter delays and feedforward (e.g. diagonal gain, no absorption), plot IRs, and run a dry signal through the model.
    """)
    return


@app.cell
def _():
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from collections import OrderedDict
    from IPython.display import Audio, display

    import pyFDN
    from flamo.processor import system, dsp

    return Audio, OrderedDict, display, dsp, np, plt, pyFDN, system, torch


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
    model = pyFDN.vanilla_FDN(n, fs=fs, n_fft=2**18)
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
    fdn = core.branchB
    feedback_loop = fdn.feedback_loop
    delay_module = feedback_loop.feedforward.delay

    new_delays = np.linspace(1100, 1150, n, dtype=np.int64)
    new_delays_t = torch.tensor(new_delays, dtype=torch.float32, device=next(delay_module.parameters()).device)
    delay_module.assign_value(delay_module.sample2s(new_delays_t))

    _n_fft = getattr(model.get_inputLayer(), 'nfft')
    device = next(delay_module.parameters()).device
    diagonal_gain = dsp.Gain(size=(n, n), nfft=_n_fft, device=device)
    diagonal_gain.assign_value(torch.diag(0.9999 ** new_delays_t))
    feedback_loop.feedforward = system.Series(OrderedDict({'delay': delay_module, 'diagonal_gain': diagonal_gain}))

    ir_altered = model.get_time_response().flatten()
    return (ir_altered,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot and listen to IRs
    """)
    return


@app.cell
def _(Audio, display, fs, ir_altered, ir_original, np, plt, pyFDN):
    t = np.arange(len(ir_original)) / fs
    plt.figure(figsize=(10, 3))
    plt.plot(t, pyFDN.mulaw_encode(ir_original), alpha=0.8, lw=0.6, label="Original")
    if "ir_altered" in globals():
        plt.plot(t, pyFDN.mulaw_encode(ir_altered), alpha=0.8, lw=0.6, label="Altered")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (μ-law)")
    plt.title("Vanilla FDN impulse response")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 2])
    plt.tight_layout()
    plt.show()

    display(Audio(ir_original, rate=fs))
    display(Audio(ir_altered, rate=fs))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Process dry audio

    Load packaged synth dry, trim to one block of length `n_fft`, run through the model, listen to dry and wet.
    """)
    return


@app.cell
def _(Audio, display, fs, model, torch):
    from importlib.resources import files
    import soundfile as sf

    try:
        path = files('pyFDN.audio') / 'synth_dry.wav'
        with path.open('rb') as f:
            dry, file_fs = sf.read(f, dtype='float64')
    except Exception:
        raise FileNotFoundError('pyFDN.audio/synth_dry.wav not found.')
    dry = dry[:, 0] if dry.ndim > 1 else dry
    if file_fs != fs:
        from scipy.signal import resample
        dry = resample(dry, int(len(dry) * fs / file_fs))
    
    _n_fft = getattr(model.get_inputLayer(), 'nfft')
    dry = dry[:int(_n_fft)]
    dry[-(fs * 2):] = 0
    x = torch.tensor(dry, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():  # pad with zeros to avoid wrap around
        wet = model(x).squeeze().cpu().numpy()

    display(Audio(dry, rate=fs))
    display(Audio(wet, rate=fs))
    return


if __name__ == "__main__":
    app.run()
