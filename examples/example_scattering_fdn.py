# gallery_category: Feedback Matrices
# gallery_title: Scattering feedback matrices
# gallery_description: Compare dense, velvet-noise, elemental, and static feedback matrices by their echo-density buildup.

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo, pyFDN):
    mo.md(f"""
    # Scattering feedback matrices

    Demonstration of different types of scattering (FIR paraunitary) feedback matrices from `filter_matrix_gallery`:

    - **RandomDense** — dense cascaded paraunitary matrix;
    - **Velvet** — sparse velvet-noise feedback matrix;
    - **FromElementals** — cascade of degree-one lossless factors;
    - **NoScatter** — plain static orthogonal matrix (for comparison).

    Validation is performed with the normalized echo density measure (Abel & Huang 2006): scattering matrices build up echo density much faster than the static matrix.

    Reference: *{pyFDN.paper_link("Scattering_in_Feedback_Delay_Networks")}*.

    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    import pyFDN

    return np, plt, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define FDN and feedback matrices
    """)
    return


@app.cell
def _(np, pyFDN):
    _rng = np.random.default_rng(1)
    fs = 48000
    ir_len = fs

    num_delays = 4
    delays = _rng.integers(750, 2001, num_delays)
    input_gain = np.ones((num_delays, 1))
    output_gain = np.ones((1, num_delays))
    direct = np.zeros((1, 1))

    num_stages = 3
    sparsity = 3
    feedback_matrices = {
        _name: pyFDN.filter_matrix_gallery(
            num_delays, _name, num_stages=num_stages, sparsity=sparsity, rng=_rng
        )
        for _name in pyFDN.FILTER_MATRIX_TYPES
    }
    # a static matrix is a one-tap filter matrix
    feedback_matrices["NoScatter"] = pyFDN.random_orthogonal(num_delays, rng=_rng)[
        :, :, None
    ]

    print(f"Delays: {delays}")
    for _name, _mat in feedback_matrices.items():
        print(f"{_name}: {_mat.shape[2]} taps")
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

    Each FIR feedback matrix is a `td.MatrixFIR` on the feedback path of the time-domain graph (each matrix entry is an FIR filter with persistent state), with `A` the identity.
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
    np,
    output_gain,
    pyFDN,
):
    irs = {}
    echo_densities = {}
    for _name, _mat in feedback_matrices.items():
        _graph = pyFDN.dss_to_td(
            delays,
            np.eye(len(delays)),
            input_gain,
            output_gain,
            direct,
            post_matrix=pyFDN.td.MatrixFIR(_mat),
        )
        irs[_name] = _graph.impulse_response(ir_len)[:, 0, 0]
        _t_mix, echo_densities[_name] = pyFDN.echo_density(irs[_name], fs)
        print(f"{_name}: mixing time = {_t_mix:.0f} ms")
    return echo_densities, irs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot

    Solid: impulse responses (offset per type). Dashed: normalized echo density profiles. The scattering matrices reach echo density 1 (Gaussian) long before the static matrix.
    """)
    return


@app.cell
def _(echo_densities, fs, irs, np, plt):
    fig, _ax = plt.subplots(figsize=(8, 5.6))
    colors = ["#636efa", "#ef553b", "#00cc96", "#ab63fa"]
    t_axis = np.arange(len(next(iter(irs.values())))) / fs
    for _i, _name in enumerate(irs):
        _offset = 2.0 * (_i + 1)
        _ax.plot(
            t_axis, irs[_name] + _offset, color=colors[_i], linewidth=0.6, label=_name
        )
        _ax.plot(
            t_axis,
            echo_densities[_name] + _offset,
            color=colors[_i],
            linewidth=1.8,
            linestyle="--",
        )
    _ax.set_title("Impulse response (solid) and echo density (dashed)")
    _ax.set_xlabel("Time (s)")
    _ax.set_ylabel("Amplitude and echo density (offset per type)")
    _ax.legend()
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
