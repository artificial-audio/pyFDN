import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def cell_imports():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from pathlib import Path
    from scipy.io import loadmat
    from scipy.linalg import expm
    from scipy import signal as scipy_signal
    import soundfile as sf
    from io import BytesIO
    import torch
    import pyFDN

    return (
        BytesIO,
        Path,
        expm,
        go,
        loadmat,
        make_subplots,
        mo,
        np,
        pyFDN,
        sf,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Colorless FDN

    FDN optimized for reduced metallic ringing (perceptually colorless reverberation). Original method published in *"Differentiable Feedback Delay Network for Colorless Reverberation," G Dal Santo, K Prawda, SJ Schlecht, V Välimäki, 26th International Conference on Digital Audio Effects (DAFx23), 244-251.*

    Parameters are loaded from `.mat` files (e.g. from [diff-fdn-colorless](https://github.com/gdalsanto/diff-fdn-colorless)). The impulse response is computed with `pyFDN.dss2impz`.

    - Original script in Matlab: Gloria Dal Santo, Wed, 18. Oct 2023
    - Python translation: Sebastian J. Schlecht, 2026-02-18
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Parameters
    """)
    return


@app.cell
def cell_params_ui(mo):
    N_ui = mo.ui.dropdown(
        options={"4": 4, "6": 6, "8": 8, "16": 16},
        value="4",
        label="FDN order N",
    )
    delay_set_ui = mo.ui.radio(
        options={"Set 1": 1, "Set 2": 2},
        value="Set 1",
        label="Delay set",
    )
    controls = mo.hstack([N_ui, delay_set_ui])

    controls
    return N_ui, delay_set_ui


@app.cell
def cell_params(N_ui, Path, delay_set_ui, pyFDN):
    fs = 48000
    rt = 3.0
    ir_len = int(rt * fs)
    g = pyFDN.db_to_lin(pyFDN.rt_to_slope(rt, fs))
    _param_candidates = [
        Path(__file__).parent / "resources" / "colorless_FDN",
        Path.cwd() / "resources" / "colorless_FDN",
        Path.cwd() / "examples" / "resources" / "colorless_FDN",
    ]
    param_dir = next((p for p in _param_candidates if p.is_dir()), _param_candidates[0])
    N = N_ui.value
    delay_set = delay_set_ui.value
    return N, delay_set, fs, g, ir_len, param_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Load parameters from mat file

    `load_colorless_params(path)` loads m, A, B, C from the mat file, builds `Ag = expm(skew(A)) @ diag(g^m)` using `pyFDN.skew`, and returns `(m_int, Ag, B, C, D)` for use with `pyFDN.dss_to_impz`.
    """)
    return


@app.cell
def cell_load_helper(Path, expm, loadmat, np, pyFDN):
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
def cell_ir_optim(
    N,
    delay_set,
    g,
    ir_len,
    load_colorless_params,
    np,
    param_dir,
    pyFDN,
):
    path_optim = param_dir / f"param_N{N}_d{delay_set}.mat"
    m, A, B, C, D = load_colorless_params(path_optim)
    Gamma = np.diag(g**m)
    Ag = A @ Gamma
    ir_optim = pyFDN.dss_to_impz(ir_len, m, Ag, B, C, D).squeeze()
    return Ag, B, C, D, ir_optim, m


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Compare to Initialization Parameters
    """)
    return


@app.cell
def cell_ir_init(
    N,
    delay_set,
    g,
    ir_len,
    load_colorless_params,
    np,
    param_dir,
    pyFDN,
):
    path_init = param_dir / f"param_init_N{N}_d{delay_set}.mat"
    m_i, A_i, B_i, C_i, D_i = load_colorless_params(path_init)
    Gamma_i = np.diag(g**m_i)
    Ag_i = A_i @ Gamma_i
    ir_init = pyFDN.dss_to_impz(ir_len, m_i, Ag_i, B_i, C_i, D_i).squeeze()
    return Ag_i, B_i, C_i, D_i, ir_init, m_i


@app.cell
def cell_header_ir(mo):
    mo.md("""
    ## Impulse Responses
    """)
    return


@app.cell
def cell_plot_ir(fs, go, ir_init, ir_len, ir_optim, np, pyFDN):
    t = np.arange(ir_len) / fs
    fig_ir = go.Figure()
    fig_ir.add_trace(go.Scatter(x=t, y=pyFDN.mulaw_encode(ir_optim), name="Optimized", line=dict(width=0.6), opacity=0.8))
    fig_ir.add_trace(go.Scatter(x=t, y=pyFDN.mulaw_encode(ir_init), name="Random Init", line=dict(width=0.6), opacity=0.8))
    fig_ir.update_layout(
        xaxis_title="Time [s]",
        yaxis_title="Amplitude [mu-law]",
        height=300,
    )
    return


@app.cell
def cell_audio(BytesIO, fs, ir_init, ir_optim, mo, np, sf):
    def _ir_to_audio(ir):
        buf = BytesIO()
        sf.write(buf, ir / (np.max(np.abs(ir)) + 1e-9), fs, format="WAV", subtype="FLOAT")
        buf.seek(0)
        return mo.audio(buf)

    audio_panel = mo.hstack(
        [
            mo.vstack([mo.md("**Random Init**"), _ir_to_audio(ir_init)]),
            mo.vstack([mo.md("**Optimized**"), _ir_to_audio(ir_optim)]),
        ]
    )

    audio_panel
    return


@app.cell
def cell_header_edc(mo):
    mo.md("""
    ## Energy Decay Curves
    """)
    return


@app.cell
def cell_edc(fs, go, ir_init, ir_len, ir_optim, np, pyFDN):
    edc_optim = pyFDN.edc(ir_optim)
    edc_init = pyFDN.edc(ir_init)

    def _to_db(e):
        return 10 * np.log10(np.maximum(e / (e[0] + 1e-30), 1e-8))

    t_edc = np.arange(ir_len) / fs
    _stride = max(1, ir_len // 4000)
    fig_edc = go.Figure()
    fig_edc.add_trace(go.Scatter(x=t_edc[::_stride], y=_to_db(edc_optim)[::_stride], name="Optimized", line=dict(width=1)))
    fig_edc.add_trace(go.Scatter(x=t_edc[::_stride], y=_to_db(edc_init)[::_stride], name="Random Init", line=dict(width=1)))
    fig_edc.add_hline(y=-60, line=dict(color="black", width=0.6, dash="dash"), opacity=0.6, annotation_text="−60 dB (RT60)")
    fig_edc.update_layout(
        xaxis_title="Time [s]",
        yaxis_title="EDC [dB]",
        yaxis=dict(range=[-80, 5]),
        height=300,
    )
    return


@app.cell
def cell_header_modal(mo):
    mo.md("""
    ## Modal Decomposition

    Poles and residues via `pyFDN.flamo_to_pr` (FLAMO-based Newton/Ehrlich–Aberth refinement).
    The **residue histogram** shows how uniformly modal energy is distributed —
    a colorless design aims for a flat distribution across frequencies.
    """)
    return


@app.cell
def cell_modal(Ag, B, C, D, fs, m, pyFDN, torch):
    _model = pyFDN.dss_to_flamo(
        A=Ag, B=B, C=C, D=D, m=m, Fs=fs, nfft=2**16, shell=True, dtype=torch.float64
    )
    residues, poles, direct, is_pair, meta_data = pyFDN.flamo_to_pr(
        _model,
        quality_threshold=1e-10,
        refinement_tol=1e-10,
        maximum_iterations=80,
        reject_unstable_poles=True,
        deflation_type="fullDeflation",
    )
    return direct, is_pair, poles, residues


@app.cell
def cell_modal_init(Ag_i, B_i, C_i, D_i, fs, m_i, pyFDN, torch):
    _model_i = pyFDN.dss_to_flamo(
        A=Ag_i, B=B_i, C=C_i, D=D_i, m=m_i, Fs=fs, nfft=2**16, shell=True, dtype=torch.float64
    )
    residues_i, poles_i, direct_i, is_pair_i, meta_data_i = pyFDN.flamo_to_pr(
        _model_i,
        quality_threshold=1e-10,
        refinement_tol=1e-10,
        maximum_iterations=80,
        reject_unstable_poles=True,
        deflation_type="fullDeflation",
    )
    return direct_i, is_pair_i, poles_i, residues_i


@app.cell
def cell_residue_histogram(
    fs,
    go,
    make_subplots,
    np,
    poles,
    poles_i,
    pyFDN,
    residues,
    residues_i,
):
    res_mag = pyFDN.lin_to_db(np.abs(residues[:, 0, 0]))
    res_mag_i = pyFDN.lin_to_db(np.abs(residues_i[:, 0, 0]))
    pole_freq = np.abs(np.angle(poles)) / np.pi * (fs / 2)
    pole_freq_i = np.abs(np.angle(poles_i)) / np.pi * (fs / 2)

    fig_residue = make_subplots(rows=1, cols=2, subplot_titles=["Residue Magnitude Distribution", "Residue vs Modal Frequency"])
    fig_residue.add_trace(go.Histogram(x=res_mag, nbinsx=60, name="Optimized", opacity=0.7), row=1, col=1)
    fig_residue.add_trace(go.Histogram(x=res_mag_i, nbinsx=60, name="Random Init", opacity=0.7), row=1, col=1)
    fig_residue.add_trace(go.Scatter(x=pole_freq_i, y=res_mag_i, mode="markers", name="Random Init", marker=dict(size=3, opacity=0.4), showlegend=False), row=1, col=2)
    fig_residue.add_trace(go.Scatter(x=pole_freq, y=res_mag, mode="markers", name="Optimized", marker=dict(size=3, opacity=0.4), showlegend=False), row=1, col=2)
    fig_residue.update_xaxes(title_text="|Residue|", row=1, col=1)
    fig_residue.update_yaxes(title_text="Count", row=1, col=1)
    fig_residue.update_xaxes(title_text="Modal Frequency [Hz]", row=1, col=2)
    fig_residue.update_yaxes(title_text="|Residue|", row=1, col=2)
    fig_residue.update_layout(barmode="overlay", height=400)
    return


@app.cell
def cell_pole_plot(go, make_subplots, np, poles, poles_i):
    _theta = np.linspace(0, 2 * np.pi, 500)
    fig_poles = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Poles — Random Init ({len(poles_i)} modes)", f"Poles — Optimized ({len(poles)} modes)"],
    )

    for _col_idx, _p in enumerate([poles_i, poles], start=1):
        fig_poles.add_trace(
            go.Scatter(
                x=np.cos(_theta), y=np.sin(_theta),
                mode="lines", line=dict(color="black", width=0.8, dash="dash"),
                opacity=0.5, name="Unit circle", showlegend=(_col_idx == 1),
            ),
            row=1, col=_col_idx,
        )
        _marker = dict(size=4, color=np.abs(_p), colorscale="plasma", opacity=0.7, showscale=(_col_idx == 2))
        if _col_idx == 2:
            _marker["colorbar"] = dict(title="|pole| (decay rate)")
        fig_poles.add_trace(
            go.Scatter(
                x=_p.real, y=_p.imag, mode="markers",
                marker=_marker,
                name=["Random Init", "Optimized"][_col_idx - 1],
            ),
            row=1, col=_col_idx,
        )

    fig_poles.update_xaxes(title_text="Re")
    fig_poles.update_yaxes(title_text="Im", scaleanchor="x", scaleratio=1, row=1, col=1)
    fig_poles.update_yaxes(title_text="Im", scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig_poles.update_layout(height=500)
    return


@app.cell
def cell_modal_comparison(
    direct,
    direct_i,
    go,
    ir_init,
    ir_len,
    ir_optim,
    is_pair,
    is_pair_i,
    make_subplots,
    poles,
    poles_i,
    pyFDN,
    residues,
    residues_i,
):
    ir_pr = pyFDN.pr_to_impz(residues, poles, direct, is_pair, ir_len).squeeze()
    ir_pr_i = pyFDN.pr_to_impz(residues_i, poles_i, direct_i, is_pair_i, ir_len).squeeze()

    diff = ir_optim - ir_pr
    diff_i = ir_init - ir_pr_i

    fig_cmp = make_subplots(
        rows=3, cols=2, shared_xaxes=True,
        subplot_titles=["Random Init", "Optimized", "", "", "", ""],
    )

    for _col, (_ir_ref, _ir_modal, _diff_ir) in enumerate(zip(
        [ir_init, ir_optim],
        [ir_pr_i, ir_pr],
        [diff_i, diff],
    ), start=1):
        fig_cmp.add_trace(go.Scatter(y=_ir_ref, name="DSS (reference)", line=dict(width=0.5), opacity=0.8, showlegend=(_col == 1)), row=1, col=_col)
        fig_cmp.add_trace(go.Scatter(y=_ir_modal, name="PR (modal)", line=dict(width=0.5), opacity=0.8, showlegend=(_col == 1)), row=1, col=_col)
        fig_cmp.add_trace(go.Scatter(y=pyFDN.mulaw_encode(_ir_ref), name="DSS", line=dict(width=0.4), opacity=0.8, showlegend=False), row=2, col=_col)
        fig_cmp.add_trace(go.Scatter(y=pyFDN.mulaw_encode(_ir_modal), name="PR", line=dict(width=0.4), opacity=0.8, showlegend=False), row=2, col=_col)
        fig_cmp.add_trace(go.Scatter(y=pyFDN.mulaw_encode(_diff_ir), line=dict(width=0.4, color="green"), showlegend=False), row=3, col=_col)

    fig_cmp.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig_cmp.update_yaxes(title_text="Amplitude [mu-law]", row=2, col=1)
    fig_cmp.update_yaxes(title_text="Difference [mu-law]", row=3, col=1)
    fig_cmp.update_xaxes(title_text="Sample", row=3)
    fig_cmp.update_layout(title_text="Modal Decomposition vs DSS Reference", height=700)
    return


if __name__ == "__main__":
    app.run()
