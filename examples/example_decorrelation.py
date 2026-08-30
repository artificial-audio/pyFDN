# gallery_category: Analysis & Verification
# gallery_description: Measure how a velvet-noise scattering feedback matrix decorrelates the input-output paths of an FDN.

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Decorrelation in feedback delay networks

    Analyses the decorrelation properties of an FDN with a velvet-noise scattering feedback matrix.

    The MIMO transfer function of an FDN factorises as $H(z) = C \, \mathrm{adj}(P(z)) B \,/\, \det(P(z)) + D$ with the characteristic matrix $P(z) = \mathrm{diag}(z^{m}) - A(z)$.  The adjugate matrix $\mathrm{adj}(P(z))$ collects the FIR filters that differentiate the input-output paths: the more decorrelated its entries, the more decorrelated the FDN outputs.  Here we compute the adjugate, then the pairwise maximum cross-correlation between all of its entries.
    """)
    return


@app.cell(hide_code=True)
def _(mo, pyFDN):
    mo.md(f"""
    Reference: *{pyFDN.paper_link("Decorrelation_in_Feedback_Delay_Networks")}*

    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go

    import pyFDN

    return go, np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define FDN

    A small FDN ($N = 4$) with random delays and a sparse velvet-noise paraunitary feedback matrix (3 cascaded stages, sparsity 3).
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


@app.cell
def _(go, np):
    def summarise_correlation(matrix):
        """Median and IQR of the off-diagonal maximum correlations."""
        upper = np.abs(matrix[np.triu_indices(matrix.shape[0], k=1)])
        values = upper[upper >= np.finfo(float).eps]
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        return np.median(values), iqr

    def correlation_heatmap(matrix, labels, title, axis_titles, size):
        fig = go.Figure(
            go.Heatmap(
                z=np.abs(matrix),
                x=labels,
                y=labels,
                zmin=0,
                zmax=1,
                colorscale="gray",
                colorbar={"title": "|max corr|"},
            )
        )
        fig.update_layout(
            title=title,
            xaxis={"title": axis_titles[0], "type": "category"},
            yaxis={
                "title": axis_titles[1],
                "type": "category",
                "autorange": "reversed",
            },
            template="plotly_white",
            height=size[0],
            width=size[1],
        )
        return fig

    return correlation_heatmap, summarise_correlation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adjugate of the characteristic matrix

    `loop_tf` constructs the polynomial matrix $P(z)$; `adj_poly` computes its adjugate by evaluating $P$ on a DFT grid, taking the scalar adjugate at every bin, and transforming back.
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
def _(P, adj_mat, mo, pyFDN):
    _fig_char, _, _ = pyFDN.plot_impulse_response_matrix(
        None,
        P.transpose(2, 0, 1),
        xlabel="Time (samples)",
        ylabel="Sample value",
        title="Characteristic matrix",
        linewidth=0.6,
    )
    _fig_adj, _, _ = pyFDN.plot_impulse_response_matrix(
        None,
        adj_mat.transpose(2, 0, 1),
        xlabel="Time (samples)",
        ylabel="Sample value",
        title="Adjugate of the characteristic matrix",
        linewidth=0.6,
    )

    mo.vstack([_fig_char, _fig_adj])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlation analysis

    `max_corr` computes the maximum normalized cross-correlation over all lags between every pair of adjugate entries (16 signals for $N = 4$, i.e. a $16 \times 16$ matrix).  The median and interquartile range of the off-diagonal correlations summarise how decorrelated the paths are.
    """)
    return


@app.cell
def _(adj_mat, pyFDN, summarise_correlation):
    max_correlation = pyFDN.max_corr(adj_mat)

    _median, _iqr = summarise_correlation(max_correlation)
    print(f"Median correlation metric: {_median:.4f}")
    print(f"Interquartile range:       {_iqr:.4f}")
    return (max_correlation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inter-channel maximum correlation matrix

    Heatmap of $|\rho_{\max}|$ between all pairs of adjugate entries. Axis label $ij$ denotes the adjugate entry in row $i$, column $j$. The diagonal is the autocorrelation (1 by construction); low off-diagonal values indicate good decorrelation.
    """)
    return


@app.cell
def _(correlation_heatmap, max_correlation, num_delays):
    _labels = [
        f"{_k % num_delays + 1}{_k // num_delays + 1}" for _k in range(num_delays**2)
    ]
    correlation_heatmap(
        max_correlation,
        _labels,
        "Inter-channel maximum correlation",
        ("ij", "kl"),
        (600, 700),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Single input distributed to all delays

    With a single source distributed equally to all delay lines ($B = \mathbf{1}$), the numerator of the transfer function collapses to the vector $\mathrm{adj}(P(z))\,\mathbf{1}$ — one FIR filter per output channel. The pairwise maximum correlation among these $N$ filters indicates the decorrelation among the FDN output channels for a single source.
    """)
    return


@app.cell
def _(adj_mat, mo, np, num_delays, pyFDN):
    input_gains = np.ones((num_delays, 1, 1))
    adj_vector = pyFDN.matrix_convolution(adj_mat, input_gains)

    _fig, _, _ = pyFDN.plot_impulse_response_matrix(
        None,
        adj_vector.transpose(2, 0, 1),
        xlabel="Time (samples)",
        ylabel="Sample value",
        title="Adjugate vector adj(P(z)) B for a single input",
        linewidth=0.6,
    )
    mo.as_html(_fig)
    return (adj_vector,)


@app.cell
def _(adj_vector, correlation_heatmap, num_delays, pyFDN, summarise_correlation):
    max_correlation_single = pyFDN.max_corr(adj_vector)

    _median, _iqr = summarise_correlation(max_correlation_single)
    print(f"Median correlation metric: {_median:.4f}")
    print(f"Interquartile range:       {_iqr:.4f}")

    correlation_heatmap(
        max_correlation_single,
        [str(_k + 1) for _k in range(num_delays)],
        "Inter-channel maximum correlation — single input",
        ("Output channel", "Output channel"),
        (450, 520),
    )
    return


if __name__ == "__main__":
    app.run()
