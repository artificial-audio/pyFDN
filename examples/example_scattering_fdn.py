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
    # Scattering feedback matrices

    Demonstration of different types of scattering (FIR paraunitary) feedback
    matrices from `filter_matrix_gallery`:

    - **RandomDense** — dense cascaded paraunitary matrix;
    - **Velvet** — sparse velvet-noise feedback matrix;
    - **FromElementals** — cascade of degree-one lossless factors;
    - **NoScatter** — plain static orthogonal matrix (for comparison).

    Validation is performed with the normalized echo density measure
    (Abel & Huang 2006): scattering matrices build up echo density much faster
    than the static matrix.

    Reference: *Schlecht, S., Habets, E. (2020). Scattering in Feedback Delay
    Networks. IEEE/ACM Transactions on Audio, Speech, and Language Processing.*
    [doi:10.1109/taslp.2020.3001395](https://dx.doi.org/10.1109/taslp.2020.3001395)

    Original MATLAB: `example_scatteringFDN.m`, Sebastian J. Schlecht,
    28 December 2019.
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
    ## Define FDN and feedback matrices
    """)
    return


@app.cell
def _(np, pyFDN):
    np.random.seed(1)
    fs = 48000
    ir_len = fs

    num_delays = 4
    delays = np.random.randint(750, 2001, num_delays)
    input_gain = np.ones((num_delays, 1))
    output_gain = np.ones((1, num_delays))
    direct = np.zeros((1, 1))

    num_stages = 3
    sparsity = 3
    feedback_matrices = {
        _name: pyFDN.filter_matrix_gallery(
            num_delays, _name, num_stages=num_stages, sparsity=sparsity
        )
        for _name in pyFDN.filter_matrix_gallery()
    }
    feedback_matrices["NoScatter"] = pyFDN.random_orthogonal(num_delays)

    print(f"Delays: {delays}")
    for _name, _mat in feedback_matrices.items():
        _taps = _mat.shape[2] if _mat.ndim == 3 else 1
        print(f"{_name}: {_taps} taps")
    return (
        delays,
        direct,
        feedback_matrices,
        fs,
        input_gain,
        ir_len,
        output_gain,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impulse responses and echo density

    `process_fdn` handles the FIR feedback matrices directly in the time-domain
    recursion (each matrix entry is an FIR filter with persistent state).
    """)
    return


@app.cell
def _(
    delays,
    direct,
    feedback_matrices,
    fs,
    input_gain,
    ir_len,
    output_gain,
    pyFDN,
):
    irs = {}
    echo_densities = {}
    for _name, _mat in feedback_matrices.items():
        irs[_name] = pyFDN.dss_to_impz(
            ir_len, delays, _mat, input_gain, output_gain, direct
        )[:, 0, 0]
        _t_mix, echo_densities[_name] = pyFDN.echo_density(irs[_name], 1024, fs, 0)
        print(f"{_name}: mixing time = {_t_mix:.0f} ms")
    return echo_densities, irs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot

    Solid: impulse responses (offset per type). Dashed: normalized echo
    density profiles. The scattering matrices reach echo density 1 (Gaussian)
    long before the static matrix.
    """)
    return


@app.cell
def _(echo_densities, fs, go, irs, np, pyFDN):
    fig = go.Figure()
    colors = ["#636efa", "#ef553b", "#00cc96", "#ab63fa"]
    t_axis = np.arange(len(next(iter(irs.values())))) / fs
    for _i, _name in enumerate(irs):
        _offset = 2.0 * (_i + 1)
        fig.add_trace(
            pyFDN.downsampled_scatter(
                x=t_axis,
                y=irs[_name] + _offset,
                mode="lines",
                name=_name,
                line={"color": colors[_i], "width": 0.6},
                legendgroup=_name,
            )
        )
        fig.add_trace(
            pyFDN.downsampled_scatter(
                x=t_axis,
                y=echo_densities[_name] + _offset,
                mode="lines",
                line={"color": colors[_i], "width": 1.8, "dash": "dash"},
                legendgroup=_name,
                showlegend=False,
            )
        )
    fig.update_layout(
        title="Impulse response (solid) and echo density (dashed)",
        xaxis={"title": "Time (s)"},
        yaxis={"title": "Amplitude and echo density (offset per type)"},
        template="plotly_white",
        height=560,
    )
    fig.show()
    return


if __name__ == "__main__":
    app.run()
