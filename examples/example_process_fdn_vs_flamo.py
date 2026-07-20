# gallery_category: FDN Design & Analysis

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
    # Time-domain FDN vs FLAMO with GEQ absorption

    The same FDN with frequency-dependent absorption is rendered by two
    independent implementations and the impulse responses are compared:

    1. **`process_fdn`** — block time-domain recursion; the per-delay-line
       SOS cascades run in an `SOSFilterBank` and the FIR feedback matrix in
       an `FIRMatrixFilter`, both with persistent state.
    2. **`dss_to_flamo`** — FLAMO frequency-domain model with the same SOS
       cascades as `parallelSOSFilter` and the FIR feedback matrix as a
       `Filter` module in the loop.

    The feedback matrix is a paraunitary scattering matrix from
    `filter_matrix_gallery`; the absorption is a 10-band graphic EQ
    (`absorption_geq`, 11 biquad sections per delay line) targeting a
    frequency-dependent reverberation time. The two impulse responses must
    match to numerical precision.

    Reference: *{paper_link("Schlecht2017AccurateReverberationTime")}.*
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    import torch

    import pyFDN

    pio.renderers.default = "sphinx_gallery"
    return go, np, pyFDN, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FDN parameters and target reverberation time
    """)
    return


@app.cell
def _(np, pyFDN):
    np.random.seed(5)
    fs = 48000
    num_delays = 4
    ir_len = fs  # 1 second

    delays = np.sort(np.random.randint(500, 2001, num_delays))
    feedback_matrix = pyFDN.filter_matrix_gallery(
        num_delays, "Velvet", num_stages=3, sparsity=3
    )
    input_gain = np.ones((num_delays, 1)) / num_delays
    output_gain = np.ones((1, num_delays))
    direct = np.zeros((1, 1))

    # Target RT at the 10 GEQ bands (seconds), decaying towards high frequencies
    target_rt = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2])

    print(f"Delays: {delays}")
    print(f"Feedback matrix: {feedback_matrix.shape[2]} taps")
    print(f"Target RT: {target_rt}")
    return (
        delays,
        direct,
        feedback_matrix,
        fs,
        input_gain,
        ir_len,
        num_delays,
        output_gain,
        target_rt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Design GEQ absorption filters

    `absorption_geq` converts the target T60 to a per-sample dB slope, fits a
    graphic EQ, and returns one SOS cascade per delay line, shape
    (N, 11, 6).
    """)
    return


@app.cell
def _(delays, fs, go, np, num_delays, pyFDN, target_rt):
    sos_absorption = pyFDN.absorption_geq(target_rt, delays, fs)
    print(f"Absorption SOS shape: {sos_absorption.shape}")

    # constract the SOSFilter
    absorption = pyFDN.SOSFilterBank(sos_absorption, len(delays))

    fig_mag = go.Figure()
    for _i in range(num_delays):
        _, _H_bands, _W_bands = pyFDN.probe_sos(
            sos_absorption[..., _i], np.array([]), fft_len=2**14, fs=fs
        )
        _mag_db = pyFDN.lin_to_db(np.abs(np.prod(_H_bands, axis=1)))
        fig_mag.add_trace(
            go.Scatter(
                x=_W_bands[:, 0],
                y=_mag_db / delays[_i],
                mode="lines",
                name=f"delay={delays[_i]}",
                line={"width": 1.2},
            )
        )
    fig_mag.update_layout(
        title="Per-delay absorption magnitude (one application)",
        xaxis={
            "title": "Frequency (Hz)",
            "type": "log",
            "range": [np.log10(50), np.log10(fs / 2)],
        },
        yaxis={"title": "Magnitude (dB/sample)"},
        template="plotly_white",
        height=400,
    )
    fig_mag.show()
    return absorption, sos_absorption


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Render with both implementations

    `process_fdn` filters the delay outputs block by block (`SOSFilterBank`)
    and runs the FIR feedback matrix in the time-domain recursion
    (`FIRMatrixFilter`); the FLAMO model places the same cascades as a
    `parallelSOSFilter` behind the delays and the FIR matrix as a `Filter`
    feedback module. FLAMO renders circularly with period `nfft`, so `nfft`
    is chosen long enough for the tail to decay below numerical precision.
    """)
    return


@app.cell
def _(
    absorption,
    delays,
    direct,
    feedback_matrix,
    fs,
    input_gain,
    ir_len,
    np,
    output_gain,
    pyFDN,
    sos_absorption,
    torch,
):
    impulse = np.zeros(ir_len)
    impulse[0] = 1.0
    ir_td = pyFDN.process_fdn(
        impulse,
        delays,
        feedback_matrix,
        input_gain,
        output_gain,
        direct,
        absorption=absorption,
    )

    model = pyFDN.dss_to_flamo(
        feedback_matrix,
        input_gain,
        output_gain,
        direct,
        delays,
        fs,
        nfft=2**17,
        sos_filter=sos_absorption,  # canonical (n_sections, 6, N) bank
        shell=True,
        dtype=torch.float64,
    )
    ir_flamo = pyFDN.flamo_time_response(model).squeeze().astype(np.float64)[:ir_len]

    difference = ir_td - ir_flamo
    max_deviation = np.max(np.abs(difference))
    print(f"Max |IR_process - IR_flamo| = {max_deviation:.3e}")
    assert pyFDN.is_almost_zero(difference, tol=1e-9)
    return difference, ir_flamo, ir_td


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impulse responses

    The two impulse responses overlap to numerical precision (mu-law encoded
    for visibility of the tail).
    """)
    return


@app.cell
def _(ir_flamo, ir_td, np, pyFDN):
    t_axis = np.arange(len(ir_td))
    pyFDN.plot_impulse_response(
        ir_td,
        ir_flamo,
        labels=["process_fdn (time domain)", "FLAMO (frequency domain)"],
        title="Impulse response: process_fdn vs FLAMO",
    )
    return (t_axis,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Error over time
    """)
    return


@app.cell
def _(difference, fs, go, pyFDN, t_axis):
    fig_err = go.Figure()
    fig_err.add_trace(
        go.Scatter(
            x=t_axis / fs,
            y=pyFDN.lin_to_db(difference),
            mode="lines",
            name="|IR_process - IR_flamo|",
            line={"width": 0.8},
        )
    )
    fig_err.update_layout(
        title="Difference between the two implementations",
        xaxis={"title": "Time (s)"},
        yaxis={"title": "Error (dB)"},
        template="plotly_white",
        height=360,
    )
    fig_err.show()
    return


if __name__ == "__main__":
    app.run()
