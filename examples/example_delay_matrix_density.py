# gallery_category: FDN Design & Analysis

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    from docs.references import paper_link

    return mo, paper_link


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Denser Reverberation with Delay Feedback Matrix

    This example compares three FDN topologies and their **echo density** (Abel & Huang 2006):

    1. **Vanilla FDN** — Build a complete FDN with `pyFDN.fdn_build_gallery`, bake in broadband decay, and render it with `pyFDN.dss_to_flamo`.
    2. **Delay+matrix+delay in feedback** — Copy the model and replace the feedback path with **delay_in → matrix → delay_out** to increase echo density.
    3. **Swapped feedforward/feedback** — Copy again and swap the base-delay and delay-matrix paths.

    Reference: *Schlecht, S., Habets, E. (2019). Dense Reverberation with Delay Feedback Matrices.* Proc. IEEE Workshop Applicat. Signal Process. Audio Acoust. (WASPAA).
    """)
    return


@app.cell
def _(mo, paper_link):
    mo.md(f"""
    # Denser Reverberation with Delay Feedback Matrix

    This example compares three FDN topologies and their **echo density** (Abel & Huang 2006):

    1. **Vanilla FDN** — Build a complete FDN with `pyFDN.fdn_build_gallery`, bake in broadband decay, and render it with `pyFDN.dss_to_flamo`.
    2. **Delay+matrix+delay in feedback** — Copy the model and replace the feedback path with **delay_in → matrix → delay_out** to increase echo density.
    3. **Swapped feedforward/feedback** — Copy again and swap the base-delay and delay-matrix paths.

    Reference: *{paper_link("Schlecht2019DenseReverberationDelay")}*.
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.renderers.default = "sphinx_gallery"  # interactive in Jupyter + docs HTML

    import pyFDN

    return go, np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameters

    Set RNG seed, sampling rate, number of delay lines **N**, broadband RT,
    and base delays plus extra **delays_in** / **delays_out** for the
    delay+matrix+delay chain.
    """)
    return


@app.cell
def _(np, pyFDN):
    rng = np.random.default_rng(7)

    fs = 48000
    nfft = 2**18
    N = 8
    rt = 3.0
    gain_per_sample = pyFDN.rt_to_gain_per_sample(rt, fs)
    delays = rng.integers(1000, 5000, size=N).astype(np.int64)
    delays_in = rng.integers(0, 200, size=N).astype(np.int64)
    delays_out = rng.integers(0, 200, size=N).astype(np.int64)
    total_delay = delays + delays_in + delays_out
    return (
        delays,
        delays_in,
        delays_out,
        fs,
        gain_per_sample,
        nfft,
        total_delay,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build vanilla FDN

    Create the complete broadband FDN with `pyFDN.fdn_build_gallery`, then
    render it with `pyFDN.dss_to_flamo`.
    """)
    return


@app.cell
def _(fs, gain_per_sample, nfft, np, pyFDN, total_delay):
    build = pyFDN.fdn_build_gallery(
        fs=fs,
        delays=total_delay,
        io_type="ones",
        direct_gain=1.0,
        rt=None,
        rng=42,
    )
    # Bake delay-proportional broadband decay into the lossless feedback matrix.
    A = np.diag(gain_per_sample**build.delays) @ build.A
    model = pyFDN.dss_to_flamo(
        A,
        build.B,
        build.C,
        build.D,
        build.delays,
        build.fs,
        nfft=nfft,
        sos_filter=build.filters,
        output_filter=build.post_eq,
    )
    ir_vanilla = pyFDN.flamo_time_response(model).flatten()
    pyFDN.plot_flamo_graph(model)
    return ir_vanilla, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Copy model and insert delay+matrix+delay in feedback

    Deep-copy the vanilla model, split its total delays into a base delay and
    **delay_in → matrix → delay_out**, then move the latter path into the
    feedback branch.
    """)
    return


@app.cell
def _(delays, delays_in, delays_out, model, pyFDN):
    model_delay_matrix = pyFDN.flamo_delay_feedback_matrix(
        model,
        delays,
        delays_in,
        delays_out,
    )

    ir_delay_matrix = pyFDN.flamo_time_response(model_delay_matrix).flatten()

    pyFDN.plot_flamo_graph(model_delay_matrix)
    return ir_delay_matrix, model_delay_matrix


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Copy model and swap feedforward and feedback

    Make another copy of the delay-matrix model and **swap** its two recursion
    paths without changing any modules or parameter values.
    """)
    return


@app.cell
def _(model_delay_matrix, pyFDN):
    model_swapped = pyFDN.swap_flamo_recursion_paths(model_delay_matrix)

    ir_swapped = pyFDN.flamo_time_response(model_swapped).flatten()

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
    t = np.arange(len(ir_vanilla)) / fs

    _, echo_vanilla = pyFDN.echo_density(ir_vanilla, n=1024, fs=fs)
    _, echo_delay_matrix = pyFDN.echo_density(ir_delay_matrix, n=1024, fs=fs)
    _, echo_swapped = pyFDN.echo_density(ir_swapped, n=1024, fs=fs)

    fig = pyFDN.plot_impulse_response(
        ir_vanilla,
        ir_delay_matrix,
        ir_swapped,
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
        pyFDN.downsampled_scatter(
            x=t,
            y=echo_vanilla,
            mode="lines",
            name="Vanilla FDN",
            line={"width": 0.8},
            opacity=0.8,
        )
    )
    fig2.add_trace(
        pyFDN.downsampled_scatter(
            x=t,
            y=echo_delay_matrix,
            mode="lines",
            name="Delay+matrix+delay in feedback",
            line={"width": 0.8},
            opacity=0.8,
        )
    )
    fig2.add_trace(
        pyFDN.downsampled_scatter(
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
            mo.audio(ir_vanilla, rate=fs),
            mo.md("Delay matrix:"),
            mo.audio(ir_delay_matrix, rate=fs),
            mo.md("Swapped:"),
            mo.audio(ir_swapped, rate=fs),
        ]
    )

    mo.vstack([plots, audio_blocks])
    return


if __name__ == "__main__":
    app.run()
