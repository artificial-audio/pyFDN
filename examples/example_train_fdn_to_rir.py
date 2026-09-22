# gallery_category: Getting Started
# gallery_title: Train an FDN to match a measured room
# gallery_description: Fit an FDN to a measured room impulse response, inspect and export its trained filters, and validate octave-band decay and level.
# references: Concert_Hall_Impulse_Responses, Learning_Filters_In_FDNs_From_Noisy_RIRs

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo, pyFDN):
    # TODO: Replace the arXiv reference with the published version when available.
    mo.md(f"""
    # Train an FDN to match a measured room

    Fit an FDN to a concert-hall impulse response using gradient descent.
    Start with a flat 1 s decay, train its absorption and output EQ together with
    the feedback matrix and gains, then export the result and validate its decay
    and level in octave bands.

    Related work: {pyFDN.paper_link("Learning_Filters_In_FDNs_From_Noisy_RIRs")}.
    The paper studies fitting attenuation filters to noisy RIRs; this example
    uses a cumulative-energy loss without an explicit noise model.
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.express as px
    import torch

    import pyFDN

    return np, px, pyFDN, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Settings

    Choose CPU or CUDA GPU below; both use float32. CUDA is selected when available.
    Absorption and output EQ are first-order shelves: two parameters and one
    biquad per filter.
    """)
    return


@app.cell
def _(mo, torch):
    fs = 48000
    _runtimes = {"CPU": "cpu"}
    _default_runtime = "CPU"
    if torch.cuda.is_available():
        _runtimes["GPU (CUDA)"] = "cuda"
        _default_runtime = "GPU (CUDA)"
    runtime_choice = mo.ui.dropdown(
        options=_runtimes, value=_default_runtime, label="Run on"
    )
    mo.hstack([runtime_choice], justify="start")
    return fs, runtime_choice


@app.cell
def _(runtime_choice):
    device = runtime_choice.value
    return (device,)


@app.cell(hide_code=True)
def _(mo, pyFDN):
    mo.md(f"""
    ## Step 1 — the target

    The Promenadikeskus concert hall in Pori, Finland, published at
    {pyFDN.paper_link("Concert_Hall_Impulse_Responses")}. Trimmed to the direct
    sound and normalized to unit energy, exactly as in **Convert a room impulse
    response into an FDN**, so the two notebooks are comparable.
    """)
    return


@app.cell
def _(fs, np, pyFDN):
    rir, _file_fs = pyFDN.load_audio("s3_r4_o", fs=fs)
    rir = rir[int(np.argmax(np.abs(rir))) :]  # trim to the direct sound
    rir = rir / np.linalg.norm(rir)  # unit energy
    rir_len = len(rir)

    print(f"target RIR: {rir_len} samples ({rir_len / fs:.2f} s) at {fs} Hz")

    return rir, rir_len


@app.cell
def _(fs, mo, pyFDN, rir):
    mo.vstack(
        [
            pyFDN.plot_spectrogram(rir, fs, title="The target: a measured hall"),
            pyFDN.labeled_audio("the room", pyFDN.peak_normalize(rir), fs=fs),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 — build the initial FDN

    Use sixteen geometrically spaced, coprime delays, a random orthogonal feedback
    matrix, normalized input/output gains, and a flat 1 s decay.
    A fixed seed makes the starting point reproducible.
    """)
    return


@app.cell
def _(fs, np, pyFDN):
    # This seed gives a feedback matrix in SO(N).
    init_build = pyFDN.fdn_build_gallery(
        N=16,
        fs=fs,
        delay_range=(700, 2500),
        delay_distribution="geometric",
        coprime=True,
        sort_delays=True,
        io_type="normalized",
        direct_gain=0.0,
        rt=1.0,  # flat 1 s decay
        rt_nyquist=1.0,
        rng=2,
    )
    assert np.linalg.det(init_build.A) > 0, "feedback matrix is not in SO(N)"

    print(f"delays (samples): {init_build.delays}")
    print(f"absorption:       {init_build.post_delay.shape} — flat 1 s")

    return (init_build,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 — make it trainable

    `trainable_from_build` creates a FLAMO model with these parameters:

    | Component | Parameter |
    |---|---|
    | Absorption (`post_delay`) | Reverberation time in seconds |
    | Output EQ (`post_output`) | Gain in dB |
    | Feedback matrix | Orthogonal matrix on $SO(N)$ |
    | Input, output and direct gains | Linear gain |

    The integer delays stay fixed. `AttenuationFilter` maps positive, smoothly
    floored RT values to per-delay attenuation, $-60d_i/(\mathrm{RT}\,f_s)$ dB,
    and designs the absorption filters. `OutputEQ` shapes the output spectrum.

    Train at `2**16` samples (1.37 s at 48 kHz) to reduce the cost per step.
    Render at `2**17` (2.73 s) to give the final decay estimators a longer window.
    Use `set_nfft` to switch grids while retaining the parameters.
    Match the initial model's energy to the target on the render window.
    """)
    return


@app.cell
def _(device, fs, init_build, np, pyFDN, rir, rir_len, torch):
    from copy import deepcopy

    train_nfft = 2**16  # 1.37 s training window
    render_nfft = 2**17  # 2.73 s validation window

    model = pyFDN.trainable_from_build(
        init_build,
        # every gain with a gradient: A, b, c and D
        trainable=pyFDN.Trainable(direct=True),
        # absorption parametrized by reverberation time
        post_delay=pyFDN.AttenuationFilter(
            1.0,
            init_build.delays,
            fs,
            design="first_order_shelf",
            nfft=render_nfft,
            device=device,
            dtype=torch.float32,
        ),
        # the output EQ, starting flat, as a gain in dB
        post_output=pyFDN.OutputEQ(
            0.0,
            1,
            fs,
            design="first_order_shelf",
            nfft=render_nfft,
            device=device,
            dtype=torch.float32,
        ),
        nfft=render_nfft,
        device=device,
        dtype=torch.float32,
    )

    # Match energy on the validation window; output gain scales the entire IR
    # because the initial direct path is zero.
    ir_init = pyFDN.flamo_time_response(model, fs=fs).squeeze().astype(float)[:rir_len]
    energy_gain = np.linalg.norm(rir[:render_nfft]) / np.linalg.norm(ir_init)
    with torch.no_grad():
        pyFDN.param(model, "output_gain").raw().mul_(energy_gain)
    ir_init *= energy_gain
    initial_build = deepcopy(pyFDN.extract_build(model))

    for _p in pyFDN.params(model):
        print(_p)
    print(f"\nenergy match: output gain x {energy_gain:.2f}")
    return initial_build, ir_init, model, render_nfft, train_nfft


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 — define the loss

    `MatchCumulativeEnergy` compares integrated short-time energy to capture decay
    and spectral shape. For the descending frequency direction:

    $$E[f,t] = \sum_{t'\ge t}\sum_{f'\ge f}|S[f',t']|^2.$$

    `frequency="both"` averages ascending and descending frequency comparisons to
    balance their weighting across the spectrum. `power=0.5` compresses the
    normalized energy to give the quieter tail more weight. The STFT window is
    1024 samples.
    """)
    return


@app.cell
def _(pyFDN, rir):
    loss = pyFDN.MatchCumulativeEnergy(rir, window=1024, power=0.5, frequency="both")

    return (loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5 — train

    Run up to 300 Adam steps at `lr=3e-2` on the training grid, then render the
    trained response on the longer grid for validation.
    """)
    return


@app.cell
def _(device, fs, loss, model, pyFDN, render_nfft, rir_len, train_nfft):
    model.set_nfft(train_nfft)
    log = pyFDN.train_fdn(
        model,
        loss,
        max_steps=300,
        lr=3e-2,
        patience=100,
        device=device,
        rng=0,
    )
    model.set_nfft(render_nfft)
    ir_trained = pyFDN.flamo_time_response(model, fs=fs).squeeze()[:rir_len]
    print(
        f"{log.steps_run} steps: loss "
        f"{log.train_loss[0]:.4g} -> {log.train_loss[-1]:.4g}"
    )
    return ir_trained, log


@app.cell
def _(log, px):
    px.line(
        y=log.train_loss,
        log_y=True,
        labels={"index": "Step", "y": "Cumulative-energy RMS error"},
        title="Training loss",
        template="plotly_white",
    ).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6 — export and inspect the fitted FDN

    `extract_build` returns a NumPy `FDNBuild` with absorption and output EQ stored
    as SOS banks. Use it with `build_to_impz`, `build_to_flamo`, or `process_fdn`.
    `param(...).raw()` gives the trained RT values in seconds and EQ gains in dB.

    `plot_FDN_build` shows the matrix, gains, delays, absorption in dB per sample,
    and output EQ in dB. Compare the initial and trained builds below.
    """)
    return


@app.cell
def _(ir_trained, model, np, pyFDN):
    # Depend on the trained response so marimo extracts after training.
    assert np.all(np.isfinite(ir_trained)), "trained FDN did not render"
    trained_build = pyFDN.extract_build(model)
    for _name, _unit in (("post_delay", "s"), ("post_output", "dB")):
        _parameter = pyFDN.param(model, _name)
        print(
            f"{_name} ({_unit}): "
            f"{_parameter.raw().detach().cpu().numpy().ravel().round(2)}"
        )
        np.testing.assert_allclose(
            getattr(trained_build, _name),
            _parameter.value().detach().cpu().numpy(),
            atol=1e-12,
        )
    return (trained_build,)


@app.cell
def _(initial_build, mo, pyFDN, trained_build):
    mo.hstack(
        [
            pyFDN.plot_FDN_build(initial_build, title="FDN, untrained"),
            pyFDN.plot_FDN_build(trained_build, title="FDN, trained"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inspect the responses

    Compare the full-band energy decay and spectrograms of the target and FDNs.
    """)
    return


@app.cell
def _(fs, ir_init, ir_trained, pyFDN, rir):
    pyFDN.plot_edc(
        rir,
        ir_init,
        ir_trained,
        fs=fs,
        labels=["Target RIR", "FDN, untrained", "FDN, trained"],
        title="Energy decay curve",
    ).show()
    return


@app.cell
def _(fs, ir_init, pyFDN):
    pyFDN.plot_spectrogram(ir_init, fs, title="FDN, untrained")
    return


@app.cell
def _(fs, ir_trained, pyFDN):
    pyFDN.plot_spectrogram(ir_trained, fs, title="FDN, trained")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Listen

    Peak-normalize all three responses for listening.
    """)
    return


@app.cell
def _(fs, ir_init, ir_trained, mo, pyFDN, rir):
    mo.hstack(
        [
            pyFDN.labeled_audio("Target RIR", pyFDN.peak_normalize(rir), fs=fs),
            pyFDN.labeled_audio("Untrained", pyFDN.peak_normalize(ir_init), fs=fs),
            pyFDN.labeled_audio("Trained", pyFDN.peak_normalize(ir_trained), fs=fs),
        ],
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 7 — validate

    Measure octave-band reverberation time and initial level in the target and
    both rendered FDNs using Schroeder integration. Compare relative RT error,
    mean level offset, and level shape error (mean absolute error after removing
    the offset). These measurements assess the fitted response independently of
    the training loss.
    """)
    return


@app.cell
def _(fs, pyFDN, rir):
    est_rt, f_centre = pyFDN.estimate_rt_bands(rir, fs)
    est_level, _ = pyFDN.estimate_initial_level_bands(rir, est_rt, fs)

    print(f"bands (Hz):   {f_centre.round(0)}")
    print(f"measured RT:  {est_rt.round(2)}")
    return est_level, est_rt, f_centre


@app.cell
def _(fs, ir_init, ir_trained, pyFDN):
    rt_init, _ = pyFDN.estimate_rt_bands(ir_init, fs)
    rt_trained, _ = pyFDN.estimate_rt_bands(ir_trained, fs)
    level_init, _ = pyFDN.estimate_initial_level_bands(ir_init, rt_init, fs)
    level_trained, _ = pyFDN.estimate_initial_level_bands(ir_trained, rt_trained, fs)
    return level_init, level_trained, rt_init, rt_trained


@app.cell
def _(
    est_level,
    est_rt,
    f_centre,
    level_init,
    level_trained,
    mo,
    np,
    px,
    pyFDN,
    rt_init,
    rt_trained,
):
    _labels = ["Target RIR", "FDN, untrained", "FDN, trained"]
    _plots = []
    for _title, _values in (
        ("RT (s)", (est_rt, rt_init, rt_trained)),
        ("Initial level (dB)", pyFDN.lin_to_db([est_level, level_init, level_trained])),
    ):
        _plots.append(
            px.line(
                {
                    "Frequency (Hz)": f_centre,
                    **dict(zip(_labels, _values, strict=True)),
                },
                x="Frequency (Hz)",
                y=_labels,
                log_x=True,
                markers=True,
                labels={"value": _title, "variable": ""},
                title=_title,
                template="plotly_white",
            )
        )

    for _name, _rt, _level in (
        ("FDN, untrained", rt_init, level_init),
        ("FDN, trained", rt_trained, level_trained),
    ):
        _err = pyFDN.lin_to_db(_level) - pyFDN.lin_to_db(est_level)
        print(
            f"{_name:16s} RT error {100 * np.abs(_rt / est_rt - 1).mean():4.1f}%   "
            f"level offset {_err.mean():+5.1f} dB   "
            f"level shape {np.abs(_err - _err.mean()).mean():4.2f} dB"
        )
    mo.hstack(_plots)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Validation checks

    Check that the measured RTs are finite and the sampled absorption curves
    are contractive for every delay line.
    Require mean RT error below 15%, worst-band error below 35%, and a mean error
    less than 30% of the initial model's.
    """)
    return


@app.cell
def _(est_rt, np, pyFDN, rt_init, rt_trained, trained_build):
    assert np.all(np.isfinite(rt_trained)), "trained RT estimates are not finite"
    _, _gain = pyFDN.sos_gain_per_sample_curves(
        trained_build.post_delay, trained_build.delays
    )
    assert np.all(np.isfinite(_gain)) and np.all((_gain > 0) & (_gain < 1)), (
        "the decay filters are not contractive at every sampled frequency"
    )

    _err_init = np.abs(rt_init / est_rt - 1)
    _err = np.abs(rt_trained / est_rt - 1)
    print(f"RT error per band, untrained: {_err_init.round(3)}")
    print(f"RT error per band, trained:   {_err.round(3)}")
    assert _err.mean() < 0.3 * _err_init.mean(), "the fit barely moved the decay"
    assert _err.mean() < 0.15, "the trained decay is not close to the measurement"
    assert _err.max() < 0.35, "one band's decay is far off the measurement"
    return


if __name__ == "__main__":
    app.run()
