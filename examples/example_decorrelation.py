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
    # Decorrelation in feedback delay networks

    Analyses the decorrelation properties of an FDN with a velvet-noise
    scattering feedback matrix.

    The MIMO transfer function of an FDN factorises as
    $H(z) = C \, \mathrm{adj}(P(z)) B \,/\, \det(P(z)) + D$ with the characteristic matrix $P(z) = \mathrm{diag}(z^{m}) - A(z)$.  The adjugate
    matrix $\mathrm{adj}(P(z))$ collects the FIR filters that differentiate
    the input-output paths: the more decorrelated its entries, the more
    decorrelated the FDN outputs.  Here we compute the adjugate, then the
    pairwise maximum cross-correlation between all of its entries.
    """)
    return


@app.cell
def _(mo, paper_link):
    mo.md(f""" 
    Reference: *{paper_link("Decorrelation_in_Feedback_Delay_Networks")}*

    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio

    import pyFDN

    pio.renderers.default = "sphinx_gallery"
    return go, np, plt, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define FDN

    A small FDN ($N = 4$) with random delays and a sparse velvet-noise
    paraunitary feedback matrix (3 cascaded stages, sparsity 3).
    """)
    return


@app.cell
def _(np, pyFDN):
    np.random.seed(5)

    num_delays = 4
    delays = np.random.randint(300, 1001, num_delays)

    num_stages = 3
    sparsity = 3
    feedback_matrix, _ = pyFDN.construct_velvet_feedback_matrix(
        num_delays, num_stages, sparsity
    )

    print(f"Delays: {delays}")
    print(f"Feedback matrix: {feedback_matrix.shape[2]} taps")
    return delays, feedback_matrix, num_delays


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adjugate of the characteristic matrix

    `loop_tf` constructs the polynomial matrix $P(z)$; `adj_poly` computes its
    adjugate by evaluating $P$ on a DFT grid, taking the scalar adjugate at
    every bin, and transforming back.
    """)
    return


@app.cell
def _(delays, feedback_matrix, pyFDN):
    P = pyFDN.loop_tf(delays, feedback_matrix)
    adj_mat = pyFDN.adj_poly(P, "z^1")
    print(f"Loop transfer function P: {P.shape}")
    print(f"Adjugate matrix: {adj_mat.shape}")
    return P, adj_mat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot impulse response matrix

    Each subplot is one FIR entry of
    - $P(z)$ - characteristic matrix
    - $\mathrm{adj}(P(z))$ — the path filter from delay-line input $j$ to delay-line output $i$ (up to the common denominator $\det P(z)$).
    """)
    return


@app.cell
def _(P, adj_mat, plt, pyFDN):
    pyFDN.plot_impulse_response_matrix(
        None,
        P.transpose(2, 0, 1),
        xlabel="Time (samples)",
        ylabel="Sample value",
        title="Characteristic matrix",
        linewidth=0.6,
    )
    plt.show()

    pyFDN.plot_impulse_response_matrix(
        None,
        adj_mat.transpose(2, 0, 1),
        xlabel="Time (samples)",
        ylabel="Sample value",
        title="Adjugate of the characteristic matrix",
        linewidth=0.6,
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlation analysis

    `max_corr` computes the maximum normalized cross-correlation over all lags
    between every pair of adjugate entries (16 signals for $N = 4$, i.e. a
    $16 \times 16$ matrix).  The median and interquartile range of the
    off-diagonal correlations summarise how decorrelated the paths are.
    """)
    return


@app.cell
def _(adj_mat, np, pyFDN):
    max_correlation = pyFDN.max_corr(adj_mat)

    # statistics over the upper triangle (each pair counted once)
    _upper = max_correlation[np.triu_indices(max_correlation.shape[0], k=1)]
    _upper = np.abs(_upper)
    corr_values = _upper[_upper >= np.finfo(float).eps]

    median_corr = np.median(corr_values)
    iqr_corr = np.percentile(corr_values, 75) - np.percentile(corr_values, 25)
    print(f"Median correlation metric: {median_corr:.4f}")
    print(f"Interquartile range:       {iqr_corr:.4f}")
    return (max_correlation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inter-channel maximum correlation matrix

    Heatmap of $|\rho_{\max}|$ between all pairs of adjugate entries.  Axis
    label $ij$ denotes the adjugate entry in row $i$, column $j$.  The
    diagonal is the autocorrelation (1 by construction); low off-diagonal
    values indicate good decorrelation.
    """)
    return


@app.cell
def _(go, max_correlation, np, num_delays):
    coord_labels = [
        f"{_k % num_delays + 1}{_k // num_delays + 1}" for _k in range(num_delays**2)
    ]
    fig_heat = go.Figure(
        go.Heatmap(
            z=np.abs(max_correlation),
            x=coord_labels,
            y=coord_labels,
            zmin=0,
            zmax=1,
            colorscale="gray",
            colorbar={"title": "|max corr|"},
        )
    )
    fig_heat.update_layout(
        title="Inter-channel maximum correlation",
        xaxis={"title": "ij", "type": "category"},
        yaxis={"title": "kl", "type": "category", "autorange": "reversed"},
        template="plotly_white",
        height=600,
        width=700,
    )
    fig_heat.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Single input distributed to all delays

    With a single source distributed equally to all delay lines
    ($B = \mathbf{1}$), the numerator of the transfer function collapses to
    the vector $\mathrm{adj}(P(z))\,\mathbf{1}$ — one FIR filter per output
    channel.  The pairwise maximum correlation among these $N$ filters
    indicates the decorrelation among the FDN output channels for a single
    source.
    """)
    return


@app.cell
def _(adj_mat, np, num_delays, plt, pyFDN):
    input_gains = np.ones((num_delays, 1, 1))
    adj_vector = pyFDN.matrix_convolution(adj_mat, input_gains)

    pyFDN.plot_impulse_response_matrix(
        None,
        adj_vector.transpose(2, 0, 1),
        xlabel="Time (samples)",
        ylabel="Sample value",
        title="Adjugate vector adj(P(z)) B for a single input",
        linewidth=0.6,
    )
    plt.show()
    return (adj_vector,)


@app.cell
def _(adj_vector, go, np, num_delays, pyFDN):
    max_correlation_single = pyFDN.max_corr(adj_vector)

    _upper = max_correlation_single[
        np.triu_indices(max_correlation_single.shape[0], k=1)
    ]
    _upper = np.abs(_upper)
    _values = _upper[_upper >= np.finfo(float).eps]
    print(f"Median correlation metric: {np.median(_values):.4f}")
    print(
        "Interquartile range:       "
        f"{np.percentile(_values, 75) - np.percentile(_values, 25):.4f}"
    )

    channel_labels = [str(_k + 1) for _k in range(num_delays)]
    fig_heat_single = go.Figure(
        go.Heatmap(
            z=np.abs(max_correlation_single),
            x=channel_labels,
            y=channel_labels,
            zmin=0,
            zmax=1,
            colorscale="gray",
            colorbar={"title": "|max corr|"},
        )
    )
    fig_heat_single.update_layout(
        title="Inter-channel maximum correlation — single input",
        xaxis={"title": "Output channel", "type": "category"},
        yaxis={
            "title": "Output channel",
            "type": "category",
            "autorange": "reversed",
        },
        template="plotly_white",
        height=450,
        width=520,
    )
    fig_heat_single.show()
    return


if __name__ == "__main__":
    app.run()
