# gallery_category: Analysis & Verification
# gallery_title: Time-domain FDN versus FLAMO
# gallery_description: Render the same paraunitary FDN with GEQ absorption in two independent engines and verify sample-accurate agreement.

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo, pyFDN):
    mo.md(f"""
    # Time-domain FDN vs FLAMO with GEQ absorption

    The same FDN with frequency-dependent absorption is rendered by two independent implementations and the impulse responses are compared:

    1. **`process_fdn`** — block time-domain recursion; the per-delay-line SOS cascades run in a `td.SOSBank` and the FIR feedback matrix in a `td.MatrixFIR`, both with persistent state.
    2. **`dss_to_flamo`** — FLAMO frequency-domain model with the same SOS cascades as `parallelSOSFilter` and the FIR feedback matrix as a `Filter` module in the loop.

    The feedback matrix is a paraunitary scattering matrix from `filter_matrix_gallery`; the absorption is a 10-band graphic EQ (`absorption_geq`, 11 biquad sections per delay line) targeting a frequency-dependent reverberation time. The two impulse responses must match to numerical precision.

    Reference: *{pyFDN.paper_link("Schlecht2017AccurateReverberationTime")}.*
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import torch

    import pyFDN
    from pyFDN import td

    return go, np, pyFDN, td, torch


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
    return delays, direct, feedback_matrix, fs, input_gain, ir_len, output_gain


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Design GEQ absorption filters

    `absorption_geq` converts the target T60 to a per-sample dB slope, fits a graphic EQ, and returns one SOS cascade per delay line, shape (N, 11, 6).
    """)
    return


@app.cell
def _(delays, fs, np, pyFDN):
    # Target RT at the 10 GEQ bands (seconds), decaying towards high frequencies
    target_rt = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2])

    sos_absorption = pyFDN.absorption_geq(target_rt, delays, fs)
    print(f"Absorption SOS shape: {sos_absorption.shape}")
    return (sos_absorption,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Render with both implementations

    `process_fdn` filters the delay outputs block by block (`td.SOSBank`) and runs the FIR feedback matrix in the time-domain recursion (`td.MatrixFIR`); the FLAMO model places the same cascades as a `parallelSOSFilter` behind the delays and the FIR matrix as a `Filter` feedback module. FLAMO renders circularly with period `nfft`, so `nfft` is chosen long enough for the tail to decay below numerical precision.
    """)
    return


@app.cell
def _(
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
    td,
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
        post_delay=td.SOSBank(sos_absorption),
    )

    model = pyFDN.dss_to_flamo(
        feedback_matrix,
        input_gain,
        output_gain,
        direct,
        delays,
        fs,
        nfft=2**17,
        post_delay=sos_absorption,  # canonical (n_sections, 6, N) bank
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

    The two impulse responses overlap to numerical precision (mu-law encoded for visibility of the tail).
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
