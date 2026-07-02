# gallery_category: FDN Design & Analysis

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from docs.references import paper_link

    return mo, paper_link


@app.cell
def _(mo, paper_link):
    mo.md(f""" 
    # Colorless FDN

    FDN optimized for reduced metallic ringing (perceptually colorless reverberation). 
    Original method published in *{paper_link("Differentiable_FDN_For_Colorless_Reverberation")}.*

    Parameters are loaded from `.mat` files (e.g. from [diff-fdn-colorless](https://github.com/gdalsanto/diff-fdn-colorless)). The impulse response is computed with `pyFDN.dss_to_impz`. The modal decomposition (residue histogram) is omitted here: pyFDN provides it via `pyFDN.dss_to_pr_direct` / `pyFDN.dss_to_pr_flamo`, but for these FDNs it means solving for `sum(delays)` ≈ 9000 modes, which is too heavy for this quick example.

    - Original script in Matlab: Gloria Dal Santo, Wed, 18. Oct 2023
    - Python translation: Sebastian J. Schlecht, 2026-02-18
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    from scipy.io import loadmat
    from scipy.linalg import expm

    import pyFDN

    return Path, expm, loadmat, np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameters
    """)
    return


@app.cell
def _(Path, pyFDN):
    fs = 48000
    rt = 3.0
    ir_len = int(rt * fs)
    g = pyFDN.db_to_lin(pyFDN.rt_to_slope(rt, fs))

    # Resolve param_dir: nbsphinx runs with cwd = notebook dir (docs/examples/), so go up to repo root
    _param_candidates = [
        Path.cwd().parent.parent
        / "examples"
        / "resources"
        / "colorless_FDN",  # from docs/examples/
        Path.cwd().parent / "examples" / "resources" / "colorless_FDN",  # from docs/
        Path.cwd() / "resources" / "colorless_FDN",  # from examples/
        Path.cwd() / "examples" / "resources" / "colorless_FDN",  # from project root
    ]
    param_dir = next((p for p in _param_candidates if p.is_dir()), _param_candidates[0])
    return fs, g, ir_len, param_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choose parameter file

    Pick the FDN size $N$ and delay set; the matching `param_init_*` file provides the random initialization.
    """)
    return


@app.cell
def _(mo, param_dir):
    import re

    _pairs = sorted(
        {
            (int(match.group(1)), int(match.group(2)))
            for p in param_dir.glob("param_N*_d*.mat")
            if (match := re.fullmatch(r"param_N(\d+)_d(\d+)\.mat", p.name))
        }
    ) or [(16, 1)]
    _options = {f"N = {n}, delay set {d}": (n, d) for n, d in _pairs}
    _default = "N = 16, delay set 1"
    param_choice = mo.ui.dropdown(
        options=_options,
        value=_default if _default in _options else next(iter(_options)),
        label="Parameter file",
    )
    mo.vstack([param_choice])
    return (param_choice,)


@app.cell
def _(param_choice):
    N, delay_set = param_choice.value
    print(f"Selected: N={N}, delay set {delay_set}")
    return N, delay_set


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load parameters from mat file

    `load_colorless_params(path, g)` loads m, A, B, C from the mat file, builds Ag = expm(skew(A)) @ diag(g^m) using `pyFDN.skew`, and returns (m_int, Ag, B, C, D) for use with dss2impz.
    """)
    return


@app.cell
def _(Path, expm, loadmat, np, pyFDN):
    def load_colorless_params(path):
        """Load colorless FDN parameters from a .mat file. Returns (m_int, A, B, C, D)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Parameter file not found: {path}")
        data = loadmat(path)
        m = np.asarray(data["m"], dtype=np.float64).ravel()
        A = np.asarray(data["A"], dtype=np.float64)
        B = np.asarray(data["B"], dtype=np.float64).ravel().reshape(-1, 1)
        C = np.asarray(data["C"], dtype=np.float64)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        D = np.zeros((1, 1))
        A_skew = pyFDN.skew(A)
        A = expm(A_skew)
        m_int = np.round(m).astype(np.int64)
        return m_int, A, B, C, D

    return (load_colorless_params,)


@app.cell
def _(N, delay_set, g, ir_len, load_colorless_params, np, param_dir, pyFDN):
    path_optim = param_dir / f"param_N{N}_d{delay_set}.mat"
    m, A, B, C, D = load_colorless_params(path_optim)
    _Gamma = np.diag(g**m)
    Ag = A @ _Gamma
    ir_optim = pyFDN.dss_to_impz(ir_len, m, Ag, B, C, D).squeeze()
    return A, B, C, D, ir_optim, m


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare to initialization parameters
    """)
    return


@app.cell
def _(N, delay_set, g, ir_len, load_colorless_params, np, param_dir, pyFDN):
    path_init = param_dir / f"param_init_N{N}_d{delay_set}.mat"
    m_i, A_i, B_i, C_i, D_i = load_colorless_params(path_init)
    _Gamma = np.diag(g**m_i)
    Ag_i = A_i @ _Gamma
    ir_init = pyFDN.dss_to_impz(ir_len, m_i, Ag_i, B_i, C_i, D_i).squeeze()
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
    ## Plot impulse responses
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
