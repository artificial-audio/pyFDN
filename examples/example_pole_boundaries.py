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
    # Frequency-dependent pole boundaries

    FDN with frequency-dependent absorption filters, but *not* with homogeneous
    (delay-proportional) decay. Still, boundaries for the pole magnitudes can be
    computed from the singular values of the loop transfer function and tested
    against the actual poles.

    The loop here is $P(z) = \mathrm{diag}(z^{m}) - A\,\mathrm{diag}(h(z))$ with
    a two-tap FIR absorption filter $h(z) = 0.65 + 0.3 z^{-1}$ on every delay
    line and a non-orthogonal feedback matrix $A = Q/1.5$.
    """)
    return


@app.cell
def _(mo, paper_link):
    mo.md(f"""
    Reference: *{paper_link("Schlecht2019ModalDecompositionFeedback")}.*

    Original MATLAB: `example_poleBoundaries.m`, Sebastian J. Schlecht,
    23 April 2018. Delays are scaled down relative to MATLAB so the
    root-finding stays fast.
    """)
    return


@app.cell
def _():
    from types import SimpleNamespace

    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    import torch

    import pyFDN

    pio.renderers.default = "sphinx_gallery"
    return SimpleNamespace, go, np, pyFDN, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define FDN with FIR absorption
    """)
    return


@app.cell
def _(SimpleNamespace, np, pyFDN):
    np.random.seed(6)
    fs = 48000
    num_delays = 8
    delays = np.random.randint(50, 301, num_delays)
    input_gain = np.eye(num_delays, 1)
    output_gain = np.eye(1, num_delays)
    direct = np.random.randn(1, 1)

    feedback_matrix = pyFDN.random_orthogonal(num_delays) / 1.5

    # two-tap FIR absorption per delay line: h(z) = 0.65 + 0.3 z^{-1}
    absorption = SimpleNamespace(
        b=np.zeros((num_delays, 1, 2)), a=np.zeros((num_delays, 1, 2))
    )
    absorption.a[:, 0, 0] = 1.0
    absorption.b[:, 0, 0] = 0.65
    absorption.b[:, 0, 1] = 0.3

    print(f"Delays: {delays} (sum = {delays.sum()})")
    return (
        absorption,
        delays,
        direct,
        feedback_matrix,
        fs,
        input_gain,
        output_gain,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pole boundaries and modal decomposition

    `pole_boundaries` combines the singular values of the feedback matrix with
    the absorption magnitude responses and group delays. For the poles, the
    FIR absorption is placed as an SOS filter behind the delays in a FLAMO
    model (loop: delay → absorption → $A$) via `dss_to_flamo`, and
    `flamo_to_pr` extracts the poles with Ehrlich–Aberth refinement in the
    $w = z^{-1}$ domain.
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
    np,
    output_gain,
    pyFDN,
    torch,
):
    min_curve, max_curve, f_bounds = pyFDN.pole_boundaries(
        delays, absorption, feedback_matrix[:, :, None], fs
    )

    # absorption FIR as one SOS section per delay line: [b0, b1, b2, a0, a1, a2]
    sos_loop = np.zeros((1, 6, delays.size))
    sos_loop[0, 0, :] = absorption.b[:, 0, 0]
    sos_loop[0, 1, :] = absorption.b[:, 0, 1]
    sos_loop[0, 3, :] = 1.0

    model = pyFDN.dss_to_flamo(
        A=feedback_matrix,
        B=input_gain,
        C=output_gain,
        D=direct,
        m=delays,
        Fs=fs,
        shell=False,
        sos_filter=sos_loop,
        dtype=torch.float64,
    )
    _residues, poles, _direct_term, _is_pair, _meta = pyFDN.flamo_to_pr(
        model,
        quality_threshold=1e-10,
        refinement_tol=1e-10,
        maximum_iterations=80,
        reject_unstable_poles=True,
        deflation_type="fullDeflation",
        verbose=False,
    )
    print(f"Number of FDN poles: {poles.size} (conjugate pairs reduced)")
    return f_bounds, max_curve, min_curve, poles


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Poles between the boundaries

    Pole magnitudes converted to T60 over frequency. All poles lie between the
    minimum and maximum boundary curves.
    """)
    return


@app.cell
def _(f_bounds, fs, go, max_curve, min_curve, np, poles, pyFDN):
    pole_freq = pyFDN.rad_to_hertz(np.angle(poles), fs)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pole_freq,
            y=pyFDN.slope_to_rt(pyFDN.lin_to_db(np.abs(poles)), fs),
            mode="markers",
            marker={"size": 4},
            name="Poles",
        )
    )
    for _curve, _name in [
        (min_curve, "Minimum boundary"),
        (max_curve, "Maximum boundary"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=f_bounds,
                y=pyFDN.slope_to_rt(pyFDN.lin_to_db(_curve), fs),
                mode="lines",
                line={"width": 3},
                name=_name,
            )
        )
    fig.update_layout(
        title="Pole T60 and frequency-dependent boundaries",
        xaxis={"title": "Frequency (Hz)"},
        yaxis={"title": "Pole RT (s)"},
        template="plotly_white",
        height=460,
    )
    fig.show()
    return (pole_freq,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verify bounds
    """)
    return


@app.cell
def _(f_bounds, max_curve, min_curve, np, pole_freq, poles, pyFDN):
    upper_ok, _ = pyFDN.is_bounding_curve(
        pole_freq, np.abs(poles), f_bounds, max_curve, "upper"
    )
    lower_ok, _ = pyFDN.is_bounding_curve(
        pole_freq, np.abs(poles), f_bounds, min_curve, "lower"
    )
    print(f"All poles below the maximum boundary: {bool(upper_ok)}")
    print(f"All poles above the minimum boundary: {bool(lower_ok)}")
    assert upper_ok and lower_ok
    return


if __name__ == "__main__":
    app.run()
