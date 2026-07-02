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
    # Random FDN statistics

    Statistics of the modal decomposition of a random FDN. The pole angles are
    almost equidistributed on the unit circle, while the residue magnitudes are
    spread across a large range.

    The residue of each mode factors into

    $$\rho_i = \underbrace{\frac{1}{l_i^H P'(\lambda_i)\, r_i}}_{\text{undriven}}
      \cdot \underbrace{(c\, r_i)(l_i^H b)}_{\text{input/output drive}},$$

    so we compare the distribution of total residues, undriven residues, and
    the input–output drive.
    """)
    return


@app.cell
def _(mo, paper_link):
    mo.md(f"""
    Reference: *{paper_link("Schlecht2019ModalDecompositionFeedback")}*

    Original MATLAB: `example_randomFDNstatistics.m`, Sebastian J. Schlecht, 23 April 2018.
    Delays are scaled down relative to MATLAB to keep the eigendecomposition fast.
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
    ## Define FDN and modal decomposition
    """)
    return


@app.cell
def _(np, pyFDN):
    np.random.seed(3)
    fs = 48000
    ir_len = fs // 2

    build = pyFDN.fdn_build_gallery(
        8,
        fs=fs,
        delay_range=(100, 401),
        io_type="identity",
        direct_gain=None,
        rt=None,
        rng=3,
    )
    delays = build.delays
    feedback_matrix = build.A
    input_gain, output_gain, direct = build.B, build.C, build.D

    print(f"Delays: {delays} (sum = {delays.sum()})")
    return delays, direct, feedback_matrix, input_gain, ir_len, output_gain


@app.cell
def _(
    delays,
    direct,
    feedback_matrix,
    input_gain,
    ir_len,
    np,
    output_gain,
    pyFDN,
):
    ir_time = pyFDN.dss_to_impz(
        ir_len, delays, feedback_matrix, input_gain, output_gain, direct
    )[:, 0, 0]

    residues, poles, direct_term, is_pair, meta = pyFDN.dss_to_pr(
        delays, feedback_matrix, input_gain, output_gain, direct
    )
    undriven_residues = meta["undrivenResidues"]

    ir_modal = pyFDN.pr_to_impz(
        residues, poles, direct_term, is_pair, ir_len, mode="lowMemory"
    )[:, 0, 0]

    difference = ir_time - ir_modal
    print(f"Number of poles: {poles.size}")
    print(f"Max |IR_time - IR_modal| = {np.max(np.abs(difference)):.3e}")
    return difference, ir_modal, ir_time, poles, residues, undriven_residues


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time-domain vs modal reconstruction
    """)
    return


@app.cell
def _(difference, go, ir_modal, ir_time, np):
    fig_ir = go.Figure()
    t_axis = np.arange(len(ir_time))
    for _sig, _name, _offset in [
        (difference, "Difference", 0.0),
        (ir_time, "Time domain", -2.0),
        (ir_modal, "Poles/residues", -4.0),
    ]:
        fig_ir.add_trace(
            go.Scatter(
                x=t_axis,
                y=_sig + _offset,
                mode="lines",
                name=_name,
                line={"width": 0.8},
            )
        )
    fig_ir.update_layout(
        title="Impulse response: time-domain recursion vs modal synthesis",
        xaxis={"title": "Time (samples)"},
        yaxis={"title": "Amplitude (offset for display)"},
        template="plotly_white",
        height=420,
    )
    fig_ir.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pole angle distribution

    The pole angles of a random lossless FDN are nearly uniformly distributed.
    """)
    return


@app.cell
def _(go, np, poles):
    angle_counts, angle_edges = np.histogram(
        np.angle(poles), bins=np.linspace(0, np.pi, 300), density=True
    )
    fig_angles = go.Figure(
        go.Bar(x=angle_edges[1:], y=angle_counts, marker={"line": {"width": 0}})
    )
    fig_angles.update_layout(
        title="Distribution of pole angles",
        xaxis={"title": "Pole angle (rad)", "range": [0, np.pi]},
        yaxis={"title": "Likelihood of occurrence"},
        template="plotly_white",
        height=380,
        bargap=0,
    )
    fig_angles.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Residue magnitude distribution

    Total residues split into the undriven part (system-intrinsic) and the
    input/output drive. The total residue magnitudes span a wide dB range.
    """)
    return


@app.cell
def _(go, np, pyFDN, residues, undriven_residues):
    db_edges = np.arange(-120.0, 41.0, 1.0)

    def _pdf(values_db):
        counts, edges = np.histogram(values_db, bins=db_edges, density=True)
        return edges[1:], counts

    fig_res = go.Figure()
    _x, _y = _pdf(pyFDN.lin_to_db(np.abs(residues[:, 0, 0] / undriven_residues)))
    fig_res.add_trace(
        go.Scatter(
            x=_x,
            y=_y,
            mode="lines",
            name="Input-output drives",
            line={"dash": "dot", "color": "black"},
        )
    )
    _x, _y = _pdf(pyFDN.lin_to_db(np.abs(residues).ravel()))
    fig_res.add_trace(
        go.Scatter(x=_x, y=_y, mode="lines", name="Total residues", line={"width": 2})
    )
    _x, _y = _pdf(pyFDN.lin_to_db(np.abs(undriven_residues)))
    fig_res.add_trace(
        go.Scatter(
            x=_x, y=_y, mode="lines", name="Undriven residues", line={"width": 2}
        )
    )
    fig_res.update_layout(
        title="Distribution of residue magnitudes",
        xaxis={"title": "Residue magnitude (dB)"},
        yaxis={"title": "Likelihood of occurrence"},
        template="plotly_white",
        height=420,
    )
    fig_res.show()
    return


if __name__ == "__main__":
    app.run()
