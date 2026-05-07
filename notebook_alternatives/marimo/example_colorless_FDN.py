import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Colorless FDN

    FDN optimized for reduced metallic ringing (perceptually colorless reverberation). Original method published in *"Differentiable Feedback Delay Network for Colorless Reverberation," G Dal Santo, K Prawda, SJ Schlecht, V Välimäki, 26th International Conference on Digital Audio Effects (DAFx23), 244-251.*

    Parameters are loaded from `.mat` files (e.g. from [diff-fdn-colorless](https://github.com/gdalsanto/diff-fdn-colorless)). The impulse response is computed with `pyFDN.dss2impz`. Modal decomposition (residue histogram) is omitted; pyFDN does not yet provide `dss2pr` (TODO).

    - Original script in Matlab: Gloria Dal Santo, Wed, 18. Oct 2023
    - Python translation: Sebastian J. Schlecht, 2026-02-18
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.io import loadmat
    from scipy.linalg import expm
    from IPython.display import Audio, display, HTML

    import pyFDN

    return Audio, HTML, Path, display, expm, loadmat, np, plt, pyFDN


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
        Path.cwd().parent.parent / "examples" / "resources" / "colorless_FDN",  # from docs/examples/
        Path.cwd().parent / "examples" / "resources" / "colorless_FDN",         # from docs/
        Path.cwd() / "resources" / "colorless_FDN",                             # from examples/
        Path.cwd() / "examples" / "resources" / "colorless_FDN",                # from project root
    ]
    param_dir = next((p for p in _param_candidates if p.is_dir()), _param_candidates[0])
    delay_set = 1
    N = 16
    return N, delay_set, fs, g, ir_len, param_dir


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
    path_optim = param_dir / f'param_N{N}_d{delay_set}.mat'
    m, A, B, C, D = load_colorless_params(path_optim)
    _Gamma = np.diag(g ** m)
    Ag = A @ _Gamma
    ir_optim = pyFDN.dss_to_impz(ir_len, m, Ag, B, C, D).squeeze()
    return (ir_optim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare to initialization parameters
    """)
    return


@app.cell
def _(N, delay_set, g, ir_len, load_colorless_params, np, param_dir, pyFDN):
    path_init = param_dir / f'param_init_N{N}_d{delay_set}.mat'
    m_i, A_i, B_i, C_i, D_i = load_colorless_params(path_init)
    _Gamma = np.diag(g ** m_i)
    Ag_i = A_i @ _Gamma
    ir_init = pyFDN.dss_to_impz(ir_len, m_i, Ag_i, B_i, C_i, D_i).squeeze()
    return (ir_init,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot impulse responses
    """)
    return


@app.cell
def _(Audio, HTML, display, fs, ir_init, ir_len, ir_optim, np, plt, pyFDN):
    t = np.arange(ir_len) / fs
    plt.figure(figsize=(10, 3))
    plt.plot(t, pyFDN.mulaw_encode(ir_optim), alpha=0.8, lw=0.6, label="Optimized")
    plt.plot(t, pyFDN.mulaw_encode(ir_init), alpha=0.8, lw=0.6, label="Random Initialization")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [mu-law]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    display(HTML("<h4>Random Initialization</h4>"))
    display(Audio(ir_init, rate=fs))

    display(HTML("<h4>Optimized</h4>"))
    display(Audio(ir_optim, rate=fs))
    return


if __name__ == "__main__":
    app.run()
