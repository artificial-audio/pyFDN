# gallery_category: Absorption & Filters

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    from docs.references import paper_link

    return mo, paper_link


@app.cell
def _(mo, paper_link):
    mo.md(f""" 
    # Absorption GEQ in an FDN

    Demonstrates `pyFDN.absorption_geq`: frequency-dependent absorption designed
    as a 10-band graphic EQ (11 biquad sections) targeting a given reverberation
    time curve.

    The absorption filters are applied per delay line.  Here we:

    1. Design the filters from a target T60 curve.
    2. Run a one-channel FDN using FLAMO.
    3. Estimate T60 from the impulse response and compare with the target.

    Reference: *{paper_link("Scattering_in_Feedback_Delay_Networks")}*

    Reference: *{paper_link("Valimaki2016AllAudioEqualization")}*

    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio

    import pyFDN

    pio.renderers.default = "sphinx_gallery"
    return go, np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FDN parameters
    """)
    return


@app.cell
def _(np, pyFDN):
    np.random.seed(5)
    fs = 48000
    num_delays = 8
    rir_len = 3 * fs  # 3 seconds

    build = pyFDN.fdn_build_gallery(
        num_delays,
        fs=fs,
        delay_range=(500, 2001),
        sort_delays=True,
        io_type="ones",
        input_scale=1 / num_delays,
        direct_gain=0.0,
        rt=None,
        rng=5,
    )
    delays = build.delays
    feedback_matrix = build.A
    B_in, C_out, D_dir = build.B, build.C, build.D

    # Target RT at the 10 GEQ bands (seconds)
    target_rt = np.array([2.0, 2.0, 2.2, 2.3, 2.1, 1.5, 1.1, 0.8, 0.7, 0.7])

    print(f"Delays: {delays}")
    print(f"Target RT: {target_rt}")
    return B_in, C_out, D_dir, delays, feedback_matrix, fs, rir_len, target_rt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Design absorption filters

    `absorption_geq` converts T60 to a per-sample dB slope, fits a GEQ, and returns
    SOS coefficients for each delay line.
    """)
    return


@app.cell
def _(delays, fs, pyFDN, target_rt):
    # absorption_geq uses the 8 interior RT values (bands 1..8)
    # The outer two are the shelf bounds; strip them to match the 10 GEQ bands
    sos_absorption = pyFDN.absorption_geq(target_rt, delays, fs)
    print(f"Absorption SOS shape: {sos_absorption.shape}")
    # shape: (11, 6, num_delays)  -> (n_sections, 6, N)
    return (sos_absorption,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Absorption filter magnitude responses

    Plot the cascaded per-delay absorption filter response for each of the 8 delay
    lines.  The curves should decay toward lower dB at higher frequencies (shorter
    T60 = more attenuation per sample at HF).
    """)
    return


@app.cell
def _(delays, fs, pyFDN, sos_absorption):
    pyFDN.plot_db_per_sample(
        sos_absorption,
        delays,
        fs=fs,
        nfft=2**14,
        title="Per-delay absorption filter magnitude (one application)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute impulse response

    Build a FLAMO FDN with the GEQ absorption filters in the loop via `dss_to_flamo`.
    Signal path: input → B → [delays → SOS → A] → C → output.
    """)
    return


@app.cell
def _(
    B_in,
    C_out,
    D_dir,
    delays,
    feedback_matrix,
    fs,
    np,
    pyFDN,
    rir_len,
    sos_absorption,
):
    nfft = int(2 ** np.ceil(np.log2(rir_len)))
    model = pyFDN.dss_to_flamo(
        feedback_matrix,
        B_in,
        C_out,
        D_dir,
        delays,
        fs,
        nfft=nfft,
        sos_filter=sos_absorption,  # canonical (n_sections, 6, N) bank
        shell=True,
    )
    rir = pyFDN.flamo_time_response(model).squeeze()[:rir_len]
    rir /= np.max(np.abs(rir)) + 1e-300
    print(f"RIR computed: {rir_len} samples, peak at sample {np.argmax(np.abs(rir))}")
    return model, rir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize DSP graph
    """)
    return


@app.cell
def _(model, pyFDN):
    pyFDN.plot_flamo_graph(model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impulse response
    """)
    return


@app.cell
def _(fs, pyFDN, rir):
    pyFDN.plot_spectrogram(
        rir, fs, title="FDN impulse response — time-frequency energy"
    )
    return


@app.cell
def _(fs, mo, np, rir):
    mo.audio(np.asarray(rir), fs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RT estimate vs target

    Estimate RT in octave bands (63–8000 Hz) by Butterworth bandpass filtering
    and compare with the design target.
    """)
    return


@app.cell
def _(fs, go, pyFDN, rir, target_rt):
    rt_est, f_centre = pyFDN.estimate_rt_bands(rir, fs)

    # target_rt[1:9] covers the same 8 octave bands (63–8k Hz)
    fig_rt = go.Figure()
    fig_rt.add_trace(
        go.Scatter(
            x=f_centre,
            y=target_rt[1:9],
            mode="lines+markers",
            name="Target RT",
            line={"dash": "dash"},
        )
    )
    fig_rt.add_trace(
        go.Scatter(
            x=f_centre,
            y=rt_est,
            mode="lines+markers",
            name="Estimated RT",
        )
    )
    fig_rt.update_layout(
        title="RT: estimated vs target",
        xaxis={"title": "Frequency (Hz)", "type": "log"},
        yaxis={"title": "RT (s)"},
        yaxis_range=[0, None],
        template="plotly_white",
        height=380,
    )
    fig_rt.show()
    return


if __name__ == "__main__":
    app.run()
