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
    # FDN with spread modal decay

    Demonstrates an FDN *without* homogeneous decay, but with a certain decay
    spread, as is typically observed in shoebox rooms and scattering delay
    networks. The spread is evaluated via the modal decomposition.

    Two feedback matrices with identical gain-per-sample energy:

    - **Proportional**: $A = Q\,\Gamma$ with orthogonal $Q$ and delay-proportional
      gains $\Gamma = \mathrm{diag}(g^{m_i})$ → all modes decay with the same T60.
    - **Spread**: $A = Q_1\,\Gamma\,Q_2$ → the second rotation distributes the
      absorption unevenly over the modes, spreading their T60s.

    Reference: *Schlecht, S., Habets, E. (2019). Modal Decomposition of Feedback
    Delay Networks. IEEE Transactions on Signal Processing 67(20), 5340-5351.*
    [doi:10.1109/tsp.2019.2937286](https://dx.doi.org/10.1109/tsp.2019.2937286)

    Original MATLAB: `example_spreadFDNpoles.m`, Sebastian J. Schlecht, 23 April 2018.
    Delays are scaled down relative to the MATLAB script to keep the
    eigendecomposition fast; the qualitative behaviour is unchanged.
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
    ## Define FDN
    """)
    return


@app.cell
def _(np, pyFDN):
    np.random.seed(1)
    fs = 48000
    types = ["Proportional", "Spread"]

    rt60 = 1.0  # seconds
    ir_len = int(rt60 * fs)
    gain_per_sample = pyFDN.rt_to_gain_per_sample(rt60, fs)

    num_delays = 8
    delays = np.random.randint(100, 301, num_delays)
    input_gain = np.eye(num_delays, 1)
    output_gain = np.eye(1, num_delays)
    direct = np.zeros((1, 1))

    gain_matrix = np.diag(gain_per_sample ** delays.astype(float))
    feedback_matrix = {
        "Proportional": pyFDN.random_orthogonal(num_delays) @ gain_matrix,
        "Spread": pyFDN.random_orthogonal(num_delays)
        @ gain_matrix
        @ pyFDN.random_orthogonal(num_delays),
    }
    print(f"Delays: {delays} (sum = {delays.sum()})")
    return (
        delays,
        direct,
        feedback_matrix,
        fs,
        input_gain,
        ir_len,
        output_gain,
        rt60,
        types,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modal decomposition, impulse response, energy decay curve

    For each matrix type, compute poles/residues with `dss_to_pr_direct`,
    synthesize the impulse response from the modes with `pr_to_impz`, and
    compute the energy decay curve.
    """)
    return


@app.cell
def _(
    delays,
    direct,
    feedback_matrix,
    input_gain,
    ir_len,
    output_gain,
    pyFDN,
    types,
):
    poles = {}
    edcs = {}
    for _type in types:
        _res, _pol, _direct_term, _is_pair, _ = pyFDN.dss_to_pr_direct(
            delays, feedback_matrix[_type], input_gain, output_gain, direct
        )
        poles[_type] = _pol
        _ir = pyFDN.pr_to_impz(
            _res, _pol, _direct_term, _is_pair, ir_len, mode="lowMemory"
        )[:, 0, 0]
        edcs[_type] = pyFDN.sq_to_db(pyFDN.edc(_ir))
        print(f"{_type}: {_pol.size} poles")
    return edcs, poles


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Energy decay curves

    The proportional FDN decays along a straight line; the spread FDN bends,
    because slowly decaying modes dominate the late tail.
    """)
    return


@app.cell
def _(edcs, fs, go, ir_len, np, types):
    fig_edc = go.Figure()
    t_axis = np.arange(ir_len) / fs
    for _type in types:
        fig_edc.add_trace(go.Scatter(x=t_axis, y=edcs[_type], mode="lines", name=_type))
    fig_edc.update_layout(
        title="Energy decay curve",
        xaxis={"title": "Time (s)"},
        yaxis={"title": "Energy Decay Curve (dB)"},
        template="plotly_white",
        height=420,
    )
    fig_edc.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pole T60s

    Per-mode reverberation time over pole angle. The proportional matrix puts
    all poles on a single T60 line; the spread matrix scatters them.
    """)
    return


@app.cell
def _(fs, go, np, poles, pyFDN, rt60, types):
    fig_poles = go.Figure()
    for _type in types:
        _pol = poles[_type]
        fig_poles.add_trace(
            go.Scatter(
                x=np.angle(_pol),
                y=pyFDN.slope_to_rt(pyFDN.lin_to_db(np.abs(_pol)), fs),
                mode="markers",
                marker={"size": 4},
                name=_type,
            )
        )
    fig_poles.update_layout(
        title="Modal reverberation times",
        xaxis={"title": "Pole angle (rad)"},
        yaxis={"title": "Pole T60 (s)"},
        template="plotly_white",
        yaxis_range=[0, rt60 * 1.5],
        height=420,
    )
    fig_poles.show()
    return


if __name__ == "__main__":
    app.run()
