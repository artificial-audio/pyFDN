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
    # FDN eigenvectors (mode shapes)

    Demonstrates how to compute the mode shapes of an FDN from the left and
    right eigenvectors of the loop polynomial $P(z) = D_m(z) - A$.

    Each residue factors into the input/output drive and an undriven part:

    $$\rho_i = \frac{(c\, r_i)\,(l_i^H b)}{l_i^H P'(\lambda_i)\, r_i},$$

    where $r_i$ and $l_i$ are the right/left null vectors of $P(\lambda_i)$.
    The eigenvectors live on the delay lines; expanding each entry along its
    delay line with $\lambda_i^k$ gives the mode shape over the full state.
    """)
    return


@app.cell
def _(mo, paper_link):
    mo.md(f"""
    Reference: *{paper_link("Schlecht2024ModalExcitationFeedback")}.*

    Original MATLAB: `example_FDNEigenvectors.m`, Sebastian J. Schlecht, 27 February 2024.
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
    np.random.seed(1)
    delays = np.array([13, 19, 23])
    build = pyFDN.fdn_build_gallery(
        delays=delays,
        io_type="ones",
        direct_gain=0.0,
        rt=None,
        rng=1,
    )
    # Bake delay-proportional broadband decay into the lossless feedback matrix.
    delays = build.delays
    A = np.diag(0.98**delays) @ build.A
    b, c, d = build.B, build.C, build.D
    num_delays = delays.size
    return A, b, c, d, delays, num_delays


@app.cell
def _(A, b, c, d, delays, pyFDN):
    residues, poles, direct, is_pair, meta = pyFDN.dss_to_pr(delays, A, b, c, d)
    num_modes = poles.size
    rv = meta["eigenvectors"]["right"]  # (N, num_modes)
    lv = meta["eigenvectors"]["left"]
    undriven = meta["undrivenResidues"]
    print(f"Number of modes (conjugate pairs reduced): {num_modes}")
    return direct, is_pair, lv, num_modes, poles, residues, rv, undriven


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Residues from eigenvectors

    Reassemble the residues from the eigenvectors and the undriven part; the
    result matches the residues returned by the modal decomposition.
    """)
    return


@app.cell
def _(b, c, lv, np, residues, rv, undriven):
    res_compact = undriven * (c @ rv).ravel() * (lv.conj().T @ b).ravel()
    max_residue_error = np.max(np.abs(residues[:, 0, 0] - res_compact))
    print(f"Max |residue - compact residue| = {max_residue_error:.3e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impulse response check

    Compare the time-domain recursion with the modal synthesis.
    """)
    return


@app.cell
def _(A, b, c, d, delays, direct, go, is_pair, np, poles, pyFDN, residues):
    ir_len = 1000
    ir_time = pyFDN.dss_to_impz(ir_len, delays, A, b, c, d)[:, 0, 0]
    ir_modal = pyFDN.pr_to_impz(residues, poles, direct, is_pair, ir_len)[:, 0, 0]

    fig_ir = go.Figure()
    fig_ir.add_trace(
        go.Scatter(y=ir_time, mode="lines", name="Impulse response (time domain)")
    )
    fig_ir.add_trace(
        go.Scatter(y=ir_modal + 1.0, mode="lines", name="IR pole residue (+1 offset)")
    )
    fig_ir.update_layout(
        title=f"Time domain vs modal synthesis (max err = {np.max(np.abs(ir_time - ir_modal)):.2e})",
        xaxis={"title": "Time (samples)"},
        yaxis={"title": "Impulse response value"},
        template="plotly_white",
        height=400,
    )
    fig_ir.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mode shapes over the full state space

    Expand the eigenvectors along each delay line: state $k$ of delay line $j$
    carries $r_{j,i}\,\lambda_i^k$. Horizontal lines mark the delay-line
    boundaries.
    """)
    return


@app.cell
def _(delays, go, np, num_delays, num_modes, poles, rv):
    rv_state_blocks = []
    for _j in range(num_delays):
        _powers = poles[None, :] ** np.arange(delays[_j])[:, None]
        rv_state_blocks.append(rv[_j, :][None, :] * _powers)
    rv_state = np.vstack(rv_state_blocks)  # (sum(delays), num_modes)

    fig_state = go.Figure(
        go.Heatmap(
            z=np.real(rv_state),
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "Re"},
        )
    )
    for _boundary in np.cumsum(delays)[:-1]:
        fig_state.add_hline(y=_boundary - 0.5, line={"color": "black", "width": 2})
    fig_state.update_layout(
        title="Right eigenvectors expanded over the state space",
        xaxis={"title": "Eigenvalue index i", "range": [-0.5, num_modes - 0.5]},
        yaxis={"title": "State space index", "autorange": "reversed"},
        template="plotly_white",
        height=520,
    )
    fig_state.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Right eigenvectors on the delay lines

    The raw right eigenvectors, one row per delay line.
    """)
    return


@app.cell
def _(go, np, num_modes, rv):
    fig_rv = go.Figure(
        go.Heatmap(
            z=np.real(rv),
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "Re", "orientation": "h"},
        )
    )
    fig_rv.update_layout(
        title="Right eigenvectors (per delay line)",
        xaxis={"title": "Eigenvalue index i", "range": [-0.5, num_modes - 0.5]},
        yaxis={"title": "Delay index", "autorange": "reversed"},
        template="plotly_white",
        height=300,
    )
    fig_rv.show()
    return


if __name__ == "__main__":
    app.run()
