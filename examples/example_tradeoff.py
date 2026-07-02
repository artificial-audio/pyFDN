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
    # FDN design tradeoff

    FDN design typically needs to balance modal and echo density with
    computational complexity:

    - the longer the delays, the more modes, but less echo density;
    - the more delays, the higher modal and echo density, but more expensive.

    We compare a 3×3 grid of settings: FDN size $N \in \{4, 8, 16\}$ and
    short/medium/long delays. Echo density (Abel & Huang 2006) makes the
    tradeoff visible: small $N$ with long delays stays sparse for a long time.
    """)
    return


@app.cell
def _(mo, paper_link):
    mo.md(f"""
    Reference: *{paper_link("Schlecht2020FDNTBFeedbackDelay")}*

    Original MATLAB: `example_tradeoff.m`, Sebastian J. Schlecht, 06 March 2023.
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.io as pio
    from plotly.subplots import make_subplots

    import pyFDN

    pio.renderers.default = "sphinx_gallery"
    return make_subplots, np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameters

    All settings share the same homogeneous decay (RT = 2 s), so the only
    differences are mode count and reflection density.
    """)
    return


@app.cell
def _(np, pyFDN):
    np.random.seed(2)
    fs = 48000
    ir_len = 2 * fs
    rt = 2.0  # seconds
    gain_per_sample = pyFDN.rt_to_gain_per_sample(rt, fs)

    sizes = [4, 8, 16]  # FDN size: small, medium, large
    delay_scales = [300, 1000, 3000]  # delays: short, medium, long
    base_delays = np.random.rand(16) + 0.5  # shared random delay pattern
    all_delays = np.round(
        base_delays[None, :] * np.array(delay_scales)[:, None]
    ).astype(int)
    return all_delays, fs, gain_per_sample, ir_len, sizes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate impulse responses

    For each combination, an orthogonal feedback matrix is scaled by the
    delay-proportional gains (homogeneous decay), and the impulse response is
    computed with `dss_to_impz`.
    """)
    return


@app.cell
def _(all_delays, fs, gain_per_sample, ir_len, np, pyFDN, sizes):
    rirs = {}
    density = {}
    num_modes = {}
    for _n in sizes:
        for _s in range(all_delays.shape[0]):
            _delays = all_delays[_s, :_n]
            _g = np.diag(gain_per_sample ** _delays.astype(float))
            _A = pyFDN.fdn_matrix_gallery(_n, "orthogonal") @ _g
            rirs[(_n, _s)] = pyFDN.dss_to_impz(
                ir_len,
                _delays,
                _A,
                np.ones((_n, 1)),
                np.ones((1, _n)),
                np.zeros((1, 1)),
            )[:, 0, 0]
            _, density[(_n, _s)] = pyFDN.echo_density(rirs[(_n, _s)], 1024, fs, 0)
            num_modes[(_n, _s)] = int(_delays.sum())

    for (_n, _s), _modes in sorted(num_modes.items()):
        _mean_delay = int(np.mean(all_delays[_s, :_n]))
        print(f"N = {_n:2d}, mean delay = {_mean_delay:4d}: {_modes:5d} modes")
    return density, rirs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impulse responses and echo density

    Each panel shows the impulse response (gray) and its normalized echo
    density profile (red). An echo density of 1 means the response is
    indistinguishable from Gaussian noise (fully mixed). Down a column,
    longer delays slow down the echo density buildup; across a row, larger
    $N$ speeds it up.
    """)
    return


@app.cell
def _(all_delays, density, fs, make_subplots, np, pyFDN, rirs, sizes):
    fig = make_subplots(
        rows=3,
        cols=3,
        shared_xaxes=True,
        subplot_titles=[
            f"N = {_n}<br>mean delay = {int(np.mean(all_delays[_s, :_n]))}"
            for _s in range(3)
            for _n in sizes
        ],
        vertical_spacing=0.07,
    )
    t_axis = np.arange(len(next(iter(rirs.values())))) / fs

    for _s in range(3):
        for _col, _n in enumerate(sizes, start=1):
            _rir = rirs[(_n, _s)]
            _dens = density[(_n, _s)]
            fig.add_trace(
                pyFDN.downsampled_scatter(
                    x=t_axis,
                    y=pyFDN.mulaw_encode(pyFDN.peak_normalize(_rir)),
                    max_points=10_000,
                    mode="lines",
                    line={"color": "gray", "width": 0.5},
                    showlegend=False,
                ),
                row=_s + 1,
                col=_col,
            )
            fig.add_trace(
                pyFDN.downsampled_scatter(
                    x=t_axis,
                    y=_dens,
                    max_points=100,
                    mode="lines",
                    line={"color": "crimson", "width": 1.5},
                    showlegend=False,
                ),
                row=_s + 1,
                col=_col,
            )

    fig.update_layout(
        title="Impulse response (gray) and echo density (red)",
        template="plotly_white",
        height=720,
    )
    fig.update_xaxes(title_text="Time (s)", row=3)
    fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Listen

    The perceptual difference is most obvious for the small-$N$ / long-delay
    setting (flutter echoes) versus the large-$N$ / short-delay setting
    (smooth reverberation).
    """)
    return


@app.cell
def _(all_delays, fs, mo, np, pyFDN, rirs, sizes):
    _delay_labels = ["short", "medium", "long"]
    _rows = []
    for _s, _delay_label in enumerate(_delay_labels):
        _players = []
        for _n in sizes:
            _mean_delay = int(np.mean(all_delays[_s, :_n]))
            _players.append(
                mo.vstack(
                    [
                        mo.md(
                            f"**N = {_n}, {_delay_label} delays**  \n"
                            f"mean = {_mean_delay} samples"
                        ),
                        mo.audio(
                            np.asarray(0.5 * pyFDN.peak_normalize(rirs[(_n, _s)])),
                            rate=fs,
                        ).style({"width": "100%", "zoom": "0.62"}),
                    ],
                    gap=0.4,
                ).style({"min-width": "110px", "flex": "1 1 110px"})
            )
        _rows.append(mo.hstack(_players, wrap=True, gap=0.1))

    mo.vstack(_rows, gap=0.1)
    return


if __name__ == "__main__":
    app.run()
