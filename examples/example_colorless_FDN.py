# gallery_category: Optimization
# gallery_title: Colorless FDN presets
# gallery_description: Load optimized colorless FDN builds, add a chosen decay time, and compare their magnitude responses and impulse responses.

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
    # Colorless FDN

    FDN optimized for reduced metallic ringing (perceptually colorless reverberation).
    Original method published in *{pyFDN.paper_link("Differentiable_FDN_For_Colorless_Reverberation")}.*

    Parameters are loaded from readable, versioned JSON `FDNBuild` files converted from the [diff-fdn-colorless](https://github.com/gdalsanto/diff-fdn-colorless) companion material. The impulse response is computed with `pyFDN.dss_to_impz`.
    """)
    return


@app.cell
def _():
    import numpy as np

    import pyFDN

    return np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Listening parameters

    The optimization fixes the lossless part of the FDN; decay is added afterwards, so the reverberation time here is a free choice and not part of what was trained.
    """)
    return


@app.cell
def _():
    fs = 48000
    rt = 3.0
    ir_len = int(rt * fs)
    return fs, ir_len, rt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choose parameter file

    Pick the FDN size $N$ and delay set; the matching `colorless_init_*` JSON build provides the random initialization.
    """)
    return


@app.cell
def _(mo, pyFDN):
    import re

    _pairs = sorted(
        {
            (int(match.group(1)), int(match.group(2)))
            for name in pyFDN.available_fdn_presets()
            if (match := re.fullmatch(r"colorless_N(\d+)_d(\d+)", name))
        }
    )
    _options = {f"N = {n}, delay set {d}": (n, d) for n, d in _pairs}
    _default = "N = 16, delay set 1"
    param_choice = mo.ui.dropdown(
        options=_options,
        value=_default if _default in _options else next(iter(_options)),
        label="Parameter file",
    )
    mo.output.replace(param_choice)
    return (param_choice,)


@app.cell
def _(param_choice):
    N, delay_set = param_choice.value
    print(f"Selected: N={N}, delay set {delay_set}")
    return N, delay_set


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the packaged preset

    `pyFDN.load_fdn_preset` returns the coefficients as an `FDNBuild`. We add the desired decay with `pyFDN.build_set_decay` and render it directly.
    """)
    return


@app.cell
def _(N, delay_set, ir_len, pyFDN, rt):
    _preset = f"colorless_N{N}_d{delay_set}"
    _lossless = pyFDN.load_fdn_preset(_preset)
    _build = pyFDN.build_set_decay(_lossless, rt)
    ir_optim = pyFDN.build_to_impz(_build, ir_len).squeeze()
    A, B, C, D, m = (
        _lossless.A,
        _lossless.B,
        _lossless.C,
        _lossless.D,
        _lossless.delays,
    )
    return A, B, C, D, ir_optim, m


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Against the initialization

    The same delays and the same random draw, before the optimizer touched them. What changed is the feedback matrix and the input/output gains — the difference between a colorless FDN and the arbitrary one it started as.
    """)
    return


@app.cell
def _(N, delay_set, ir_len, pyFDN, rt):
    _preset = f"colorless_init_N{N}_d{delay_set}"
    _lossless = pyFDN.load_fdn_preset(_preset)
    _build = pyFDN.build_set_decay(_lossless, rt)
    ir_init = pyFDN.build_to_impz(_build, ir_len).squeeze()
    A_i, B_i, C_i, D_i, m_i = (
        _lossless.A,
        _lossless.B,
        _lossless.C,
        _lossless.D,
        _lossless.delays,
    )
    return A_i, B_i, C_i, D_i, ir_init, m_i


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FDN parameter overview

    `pyFDN.plot_fdn_parameter` shows the system matrix blocks $A$, $b$, $c$, $d$ as heatmaps and the delays as bars aligned with the columns of the feedback matrix. The optimization changes $A$, $b$, $c$ (the lossless part, before the homogeneous attenuation $\Gamma = \mathrm{diag}(g^m)$ is applied); the delays stay fixed.
    """)
    return


@app.cell
def _(A_i, B_i, C_i, D_i, m_i, pyFDN):
    pyFDN.plot_fdn_parameter(
        m_i,
        A_i,
        B_i,
        C_i,
        D_i,
        title="Random Initialization",
    )
    return


@app.cell
def _(A, B, C, D, m, pyFDN):
    pyFDN.plot_fdn_parameter(
        m,
        A,
        B,
        C,
        D,
        title="Optimized",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optimized versus initial

    The two impulse responses, and both to listen to. They look much alike — the optimization targets the *magnitude response*, not the waveform — so the difference is one to hear rather than to see: the initialization rings, the optimized build does not.
    """)
    return


@app.cell
def _(fs, ir_init, ir_optim, mo, np, pyFDN):
    plot = pyFDN.plot_impulse_response(
        ir_optim,
        ir_init,
        fs=fs,
        labels=["Optimized", "Random Initialization"],
    )

    audio_blocks = mo.vstack(
        [
            mo.Html("Random Initialization").style({"font-size": "2.0em"}),
            mo.audio(np.asarray(ir_init), rate=fs),
            mo.Html("Optimized").style({"font-size": "2.0em"}),
            mo.audio(np.asarray(ir_optim), rate=fs),
        ],
        gap=1,
    )
    mo.vstack([plot, audio_blocks], gap=3)
    return


if __name__ == "__main__":
    app.run()
