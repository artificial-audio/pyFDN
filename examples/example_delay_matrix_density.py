import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Denser Reverberation with Delay Feedback Matrix

    This example compares three FDN topologies and their **echo density** (Abel & Huang 2006):

    1. **Vanilla FDN** — Build with `pyFDN.vanilla_FDN`, then overwrite delays and feedback matrix and replace absorption with a **broadband gain** (diagonal gain per delay).
    2. **Delay+matrix+delay in feedback** — Copy the model and replace the feedback path with **delay_in → matrix → delay_out** to increase echo density.
    3. **Swapped feedforward/feedback** — Copy again and swap the two paths (feedforward ↔ feedback).

    Reference: *Schlecht, S., Habets, E. (2019). Dense Reverberation with Delay Feedback Matrices.* Proc. IEEE Workshop Applicat. Signal Process. Audio Acoust. (WASPAA).
    """)
    return


@app.cell
def _():
    import copy

    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    import torch

    pio.renderers.default = "sphinx_gallery"  # interactive in Jupyter + docs HTML
    from collections import OrderedDict

    from flamo.processor import dsp, system

    import pyFDN

    return OrderedDict, copy, dsp, go, np, pyFDN, system, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameters

    Set RNG seed, sampling rate, number of delay lines **N**, broadband gain per sample, and base delays plus extra **delays_in** / **delays_out** for the delay+matrix+delay chain.
    """)
    return


@app.cell
def _(np, pyFDN, torch):
    rng = np.random.default_rng(7)
    torch.manual_seed(42)

    fs = 48000
    N = 8
    gain_per_sample = 0.99995
    feedback_matrix = pyFDN.random_orthogonal(N)

    delays = rng.integers(1000, 5000, size=N).astype(np.int64)

    delays_in = rng.integers(0, 200, size=N).astype(np.int64)
    delays_out = rng.integers(0, 200, size=N).astype(np.int64)

    total_delay = delays + delays_in + delays_out
    return (
        N,
        delays,
        delays_in,
        delays_out,
        feedback_matrix,
        fs,
        gain_per_sample,
        rng,
        total_delay,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build vanilla FDN

    Create the model with `pyFDN.vanilla_FDN` (FLAMO). Delays and absorption will be overwritten in the next step.
    """)
    return


@app.cell
def _(N, fs, pyFDN):
    model = pyFDN.vanilla_FDN(
        N,
        fs=fs,
        n_fft=2**18,
    )
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overwrite standard values and absorption

    1. Set **delays** to the chosen lengths (3000–8000).
    2. Set **feedback matrix** to random orthogonal.
    3. Replace **absorption filters** with a **broadband gain** (diagonal matrix `gain^delays`); feedforward becomes delay + diagonal gain.
    """)
    return


@app.cell
def _(
    N,
    OrderedDict,
    dsp,
    feedback_matrix,
    gain_per_sample,
    model,
    pyFDN,
    system,
    torch,
    total_delay,
):
    core = model.get_core()
    fdn = core.branchB
    feedback_loop = fdn.feedback_loop
    delay_module = feedback_loop.feedforward.delay
    _mixing_matrix = feedback_loop.feedback
    device = next(delay_module.parameters()).device
    n_fft = model.get_inputLayer().nfft

    total_delay_t = torch.tensor(total_delay, dtype=torch.float32, device=device)
    delay_module.assign_value(delay_module.sample2s(total_delay_t))

    feedback_matrix_t = torch.tensor(
        feedback_matrix, dtype=torch.float32, device=device
    )
    _mixing_matrix.assign_value(feedback_matrix_t)

    broadband_gain = dsp.Gain(size=(N, N), nfft=n_fft, device=device)
    broadband_gain.assign_value(torch.diag(gain_per_sample**total_delay_t))
    feedback_loop.feedforward = system.Series(
        OrderedDict({"delay": delay_module, "broadband_gain": broadband_gain})
    )

    ir_vanilla = model.get_time_response().flatten()

    pyFDN.plot_flamo_graph(model)
    return delay_module, device, ir_vanilla, n_fft


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Copy model and insert delay+matrix+delay in feedback

    Deep-copy the overwritten model, then replace the feedback path with **delay_in → matrix → delay_out** (using `delays_in` and `delays_out`). Generate a second IR from this topology.
    """)
    return


@app.cell
def _(
    N,
    OrderedDict,
    copy,
    delay_module,
    delays,
    delays_in,
    delays_out,
    device,
    dsp,
    model,
    n_fft,
    np,
    pyFDN,
    system,
    torch,
):
    model_delay_matrix = copy.deepcopy(model)
    core_delay_matrix = model_delay_matrix.get_core()
    fdn_delay_matrix = core_delay_matrix.branchB
    feedback_loop_delay_matrix = fdn_delay_matrix.feedback_loop
    _mixing_matrix = feedback_loop_delay_matrix.feedback

    delays_t = torch.tensor(delays, dtype=torch.float32, device=device)
    feedback_loop_delay_matrix.feedforward.delay.assign_value(
        delay_module.sample2s(delays_t)
    )

    max_in = int(np.max(delays_in))
    max_out = int(np.max(delays_out))
    extra_delay_in = dsp.parallelDelay(
        size=(N,), max_len=max(1, max_in), nfft=n_fft, isint=True, unit=1, device=device
    )
    extra_delay_out = dsp.parallelDelay(
        size=(N,),
        max_len=max(1, max_out),
        nfft=n_fft,
        isint=True,
        unit=1,
        device=device,
    )
    extra_delay_in.assign_value(
        extra_delay_in.sample2s(
            torch.tensor(delays_in, dtype=torch.float32, device=device)
        )
    )
    extra_delay_out.assign_value(
        extra_delay_out.sample2s(
            torch.tensor(delays_out, dtype=torch.float32, device=device)
        )
    )

    delay_matrix_chain = system.Series(
        OrderedDict(
            [
                ("delay_in", extra_delay_in),
                ("matrix", _mixing_matrix),
                ("delay_out", extra_delay_out),
            ]
        )
    )
    feedback_loop_delay_matrix.feedback = delay_matrix_chain

    ir_delay_matrix = model_delay_matrix.get_time_response().flatten()

    pyFDN.plot_flamo_graph(model_delay_matrix)
    return ir_delay_matrix, model_delay_matrix


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Copy model and swap feedforward and feedback

    Make another copy of the delay+matrix+delay model and **swap** the two paths: feedforward becomes the former feedback (delay_in → matrix → delay_out), and feedback becomes the former feedforward (delay + broadband gain). Generate an IR from this swapped topology.
    """)
    return


@app.cell
def _(
    N,
    copy,
    delay_module,
    device,
    model_delay_matrix,
    np,
    pyFDN,
    rng,
    torch,
):
    # generate new delays for swapped model
    delays_swapped = rng.integers(333, 1333, size=N).astype(np.int64)
    delays_in_swapped = rng.integers(333, 2333, size=N).astype(np.int64)
    delays_out_swapped = rng.integers(333, 2333, size=N).astype(np.int64)

    # swap feedforward and feedback
    model_swapped = copy.deepcopy(model_delay_matrix)
    feedback_loop_swapped = model_swapped.get_core().branchB.feedback_loop
    old_feedforward = feedback_loop_swapped.feedforward
    old_feedback = feedback_loop_swapped.feedback
    feedback_loop_swapped.feedforward = old_feedback
    feedback_loop_swapped.feedback = old_feedforward

    feedback_loop_swapped.feedback.delay.assign_value(
        delay_module.sample2s(
            torch.tensor(delays_swapped, dtype=torch.float32, device=device)
        )
    )
    feedback_loop_swapped.feedforward.delay_in.assign_value(
        delay_module.sample2s(
            torch.tensor(delays_in_swapped, dtype=torch.float32, device=device)
        )
    )
    feedback_loop_swapped.feedforward.delay_out.assign_value(
        delay_module.sample2s(
            torch.tensor(delays_out_swapped, dtype=torch.float32, device=device)
        )
    )

    ir_swapped = model_swapped.get_time_response().flatten()

    pyFDN.plot_flamo_graph(model_swapped)
    return (ir_swapped,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot and listen

    Impulse responses (μ-law) and **echo density** (Abel & Huang 2006) for the three topologies. Echo density is computed with `pyFDN.echo_density`; the horizontal dashed line marks the mixing-time threshold (1.0).
    """)
    return


@app.cell
def _(fs, go, ir_delay_matrix, ir_swapped, ir_vanilla, mo, np, pyFDN):
    ir_v = np.asarray(ir_vanilla).ravel()
    ir_dm = np.asarray(ir_delay_matrix).ravel()
    ir_sw = np.asarray(ir_swapped).ravel()
    t = np.arange(len(ir_v)) / fs

    _, echo_vanilla = pyFDN.echo_density(ir_v, n=1024, fs=fs)
    _, echo_delay_matrix = pyFDN.echo_density(ir_dm, n=1024, fs=fs)
    _, echo_swapped = pyFDN.echo_density(ir_sw, n=1024, fs=fs)

    fig = pyFDN.plot_impulse_response(
        ir_v,
        ir_dm,
        ir_sw,
        fs=fs,
        labels=[
            "Vanilla FDN",
            "Delay+matrix+delay in feedback",
            "Swapped feedforward/feedback",
        ],
        title="Delay feedback matrix density",
    )

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=t,
            y=echo_vanilla,
            mode="lines",
            name="Vanilla FDN",
            line={"width": 0.8},
            opacity=0.8,
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=t,
            y=echo_delay_matrix,
            mode="lines",
            name="Delay+matrix+delay in feedback",
            line={"width": 0.8},
            opacity=0.8,
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=t,
            y=echo_swapped,
            mode="lines",
            name="Swapped feedforward/feedback",
            line={"width": 0.8},
            opacity=0.8,
        )
    )
    fig2.add_hline(
        y=1.0, line_dash="dash", line_color="gray", annotation_text="mixing thresh"
    )
    fig2.update_layout(
        title="Echo density (Abel & Huang 2006)",
        xaxis_title="Time [s]",
        yaxis_title="Echo density",
        xaxis={"range": [0, 0.8], "showgrid": True, "gridwidth": 1, "griddash": "dot"},
        yaxis={"showgrid": True, "gridwidth": 1, "griddash": "dot"},
        legend={"yanchor": "bottom", "y": 0.99, "xanchor": "right", "x": 0.99},
        height=350,
        margin={"l": 60, "r": 40, "t": 50, "b": 50},
    )

    plots = mo.vstack([fig, fig2])

    audio_blocks = mo.vstack(
        [
            mo.md("Vanilla:"),
            mo.audio(np.asarray(ir_vanilla), rate=fs),
            mo.md("Delay matrix:"),
            mo.audio(np.asarray(ir_delay_matrix), rate=fs),
            mo.md("Swapped:"),
            mo.audio(np.asarray(ir_swapped), rate=fs),
        ]
    )

    mo.vstack([plots, audio_blocks])
    return


if __name__ == "__main__":
    app.run()
