# gallery_category: FDN Design & Analysis

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
    # Filter feedback delay network (FFDN) with paraunitary feedback matrix

    An FDN with a *paraunitary* (FIR, lossless) scattering matrix in the loop.
    The example computes the impulse response by time-domain recursion and by
    modal decomposition, and verifies that the system is lossless (all poles
    on the unit circle).

    Reference: *Schlecht, S., Habets, E. (2020). Scattering in Feedback Delay
    Networks. IEEE/ACM Transactions on Audio, Speech, and Language Processing.*
    [doi:10.1109/taslp.2020.3001395](https://dx.doi.org/10.1109/taslp.2020.3001395)

    Original MATLAB: `example_paraunitaryFDN.m`, Sebastian J. Schlecht,
    23 April 2018.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    import torch

    import pyFDN

    pio.renderers.default = "sphinx_gallery"
    return go, np, plt, pyFDN, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define FFDN with paraunitary feedback matrix
    """)
    return


@app.cell
def _(np, pyFDN):
    np.random.seed(2)
    fs = 48000
    ir_len = int(fs / 10)

    num_delays = 4
    num_stages = 3
    delays = np.random.randint(150, 501, num_delays)
    input_gain = np.eye(num_delays, 1)
    output_gain = np.eye(1, num_delays)
    direct = np.zeros((1, 1))

    feedback_matrix = pyFDN.filter_matrix_gallery(
        num_delays, "RandomDense", num_stages=num_stages, stage_matrix_type="random"
    )
    print(f"Delays: {delays} (sum = {delays.sum()})")
    print(f"Feedback matrix: {feedback_matrix.shape[2]} taps")
    return delays, direct, feedback_matrix, fs, input_gain, ir_len, output_gain


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feedback matrix impulse responses

    Each subplot shows the FIR of one matrix entry.
    """)
    return


@app.cell
def _(feedback_matrix, np, plt, pyFDN):
    pyFDN.plot_impulse_response_matrix(
        np.arange(feedback_matrix.shape[2]),
        pyFDN.mulaw_encode(feedback_matrix.transpose(2, 0, 1)),
        xlabel="Time (samples)",
        ylabel="Amplitude (mu)",
        title="Paraunitary feedback matrix",
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verify paraunitarity

    A paraunitary matrix satisfies $A^T(z^{-1})\,A(z) = I$; in the time domain
    the matrix impulse response is lossless.
    """)
    return


@app.cell
def _(feedback_matrix, pyFDN):
    is_pu, _, max_off = pyFDN.is_paraunitary(feedback_matrix.transpose(2, 0, 1))
    print(
        f"Feedback matrix is paraunitary: {bool(is_pu)} (max off-diagonal {max_off:.2e})"
    )
    assert is_pu
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impulse response and modal decomposition

    The FIR feedback matrix runs directly in the time-domain recursion.
    For the modal decomposition the FIR matrix is placed as a FLAMO Filter
    module in the loop (`dss_to_flamo`) and `flamo_to_pr` refines the poles
    with Ehrlich–Aberth iteration. The FIR feedback adds poles beyond the
    delay count, so the number of root seeds is set to the degree of the
    generalized characteristic polynomial.
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
    torch,
):
    ir_time = pyFDN.dss_to_impz(
        ir_len, delays, feedback_matrix, input_gain, output_gain, direct
    )[:, 0, 0]

    # pole count = degree of the generalized characteristic polynomial in w = z^{-1}
    gcp = pyFDN.general_char_poly(delays, feedback_matrix)
    num_poles = int(np.nonzero(np.abs(gcp) > 1e-12 * np.max(np.abs(gcp)))[0][-1])

    model = pyFDN.dss_to_flamo(
        A=feedback_matrix,
        B=input_gain,
        C=output_gain,
        D=direct,
        m=delays,
        Fs=fs,
        shell=False,
        dtype=torch.float64,
    )
    residues, poles, direct_term, is_pair, _meta = pyFDN.flamo_to_pr(
        model,
        quality_threshold=1e-10,
        refinement_tol=1e-10,
        maximum_iterations=80,
        deflation_type="fullDeflation",
        verbose=False,
        num_poles=num_poles,
    )
    ir_modal = pyFDN.pr_to_impz(
        residues, poles, direct_term, is_pair, ir_len, mode="lowMemory"
    )[:, 0, 0]

    difference = ir_time - ir_modal
    max_deviation = np.max(np.abs(difference))
    print(f"Number of poles: {poles.size}")
    print(f"Max |IR_time - IR_modal| = {max_deviation:.3e}")
    assert pyFDN.is_almost_zero(difference, tol=1e-3)
    return difference, ir_modal, ir_time, poles, residues


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time-domain vs modal reconstruction
    """)
    return


@app.cell
def _(difference, ir_modal, ir_time, pyFDN):
    pyFDN.plot_impulse_response(
        difference,
        ir_time,
        ir_modal,
        labels=["Difference", "Time domain", "Poles/residues"],
        title="Impulse response: time-domain recursion vs modal synthesis",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Poles and residues

    The FFDN is lossless: all pole magnitudes are 0 dB. The residue magnitudes
    spread over a wide range.
    """)
    return


@app.cell
def _(go, np, poles, pyFDN, residues):
    fig_pr = go.Figure()
    fig_pr.add_trace(
        go.Scatter(
            x=np.angle(poles),
            y=pyFDN.lin_to_db(np.abs(poles)),
            mode="markers",
            marker={"size": 4},
            name="Poles",
        )
    )
    fig_pr.add_trace(
        go.Scatter(
            x=np.angle(poles),
            y=pyFDN.lin_to_db(np.abs(residues[:, 0, 0])),
            mode="markers",
            marker={"size": 4, "symbol": "x"},
            name="Residues",
        )
    )
    fig_pr.update_layout(
        title="Pole and residue magnitudes over pole angle",
        xaxis={"title": "Pole angle (rad)"},
        yaxis={"title": "Magnitude (dB)"},
        template="plotly_white",
        height=420,
    )
    fig_pr.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Losslessness check
    """)
    return


@app.cell
def _(np, poles, pyFDN):
    max_pole_deviation = np.max(np.abs(np.abs(poles) - 1.0))
    print(f"Max | |pole| - 1 | = {max_pole_deviation:.3e}")
    assert pyFDN.is_almost_zero(np.abs(poles) - 1.0)
    return


if __name__ == "__main__":
    app.run()
