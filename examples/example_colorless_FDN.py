import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def cell_imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.io import loadmat
    from scipy.linalg import expm
    import soundfile as sf
    from io import BytesIO
    import torch
    import pyFDN

    return BytesIO, Path, expm, loadmat, mo, np, plt, pyFDN, sf, torch


@app.cell
def cell_header(mo):
    header = mo.md(
        """
        # Colorless FDN

        FDN optimized for reduced metallic ringing (perceptually colorless reverberation).
        Original method published in *"Differentiable Feedback Delay Network for Colorless Reverberation,"
        G Dal Santo, K Prawda, SJ Schlecht, V Välimäki, DAFx23, 244-251.*

        Parameters are loaded from `.mat` files (e.g. from
        [diff-fdn-colorless](https://github.com/gdalsanto/diff-fdn-colorless)).

        - Original script in Matlab: Gloria Dal Santo, Wed, 18. Oct 2023
        - Python translation: Sebastian J. Schlecht, 2026-02-18
        - Marimo port with modal decomposition: 2026-04-17
        """
    )
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
    header_ir = mo.md("## Impulse Responses")
    return


@app.cell
def cell_plot_ir(fs, ir_init, ir_len, ir_optim, np, plt, pyFDN):
    t = np.arange(ir_len) / fs
    fig_ir, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, pyFDN.mulaw_encode(ir_optim), alpha=0.8, lw=0.6, label="Optimized")
    ax.plot(t, pyFDN.mulaw_encode(ir_init), alpha=0.8, lw=0.6, label="Random Init")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [mu-law]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_ir.tight_layout()
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
    return


@app.cell
def cell_header_edc(mo):
    header_edc = mo.md("## Energy Decay Curves")
    return


@app.cell
def cell_edc(fs, ir_init, ir_len, ir_optim, np, plt, pyFDN):
    edc_optim = pyFDN.edc(ir_optim)
    edc_init = pyFDN.edc(ir_init)

    def _to_db(e):
        return 10 * np.log10(np.maximum(e / (e[0] + 1e-30), 1e-8))

    t_edc = np.arange(ir_len) / fs
    fig_edc, ax_edc = plt.subplots(figsize=(10, 3))
    ax_edc.plot(t_edc, _to_db(edc_optim), lw=1, label="Optimized")
    ax_edc.plot(t_edc, _to_db(edc_init), lw=1, label="Random Init")
    ax_edc.axhline(-60, color="k", lw=0.6, ls="--", alpha=0.6, label="−60 dB (RT60)")
    ax_edc.set_ylim(-80, 5)
    ax_edc.set_xlabel("Time [s]")
    ax_edc.set_ylabel("EDC [dB]")
    ax_edc.legend()
    ax_edc.grid(True, alpha=0.3)
    fig_edc.tight_layout()
    return


@app.cell
def cell_header_modal(mo):
    header_modal = mo.md(
        """
        ## Modal Decomposition

        Poles and residues via `flamo_to_pr` (FLAMO-based Newton/Ehrlich–Aberth refinement).
        The **residue histogram** shows how uniformly modal energy is distributed —
        a colorless design aims for a flat distribution across frequencies.
        """
    )
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
    return poles, residues


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
    return poles_i, residues_i


@app.cell
def cell_residue_histogram(fs, np, plt, poles, poles_i, residues, residues_i):
    res_mag = np.abs(residues[:, 0, 0])
    res_mag_i = np.abs(residues_i[:, 0, 0])
    pole_freq = np.abs(np.angle(poles)) / np.pi * (fs / 2)
    pole_freq_i = np.abs(np.angle(poles_i)) / np.pi * (fs / 2)

    fig_residue, axes_res = plt.subplots(1, 2, figsize=(12, 4))

    axes_res[0].hist(res_mag, bins=60, alpha=0.7, label="Optimized", edgecolor="none")
    axes_res[0].hist(res_mag_i, bins=60, alpha=0.7, label="Random Init", edgecolor="none")
    axes_res[0].set_xlabel("|Residue|")
    axes_res[0].set_ylabel("Count")
    axes_res[0].set_title("Residue Magnitude Distribution")
    axes_res[0].legend()
    axes_res[0].grid(True, alpha=0.3)

    axes_res[1].scatter(pole_freq_i, res_mag_i, s=3, alpha=0.4, label="Random Init")
    axes_res[1].scatter(pole_freq, res_mag, s=3, alpha=0.4, label="Optimized")
    axes_res[1].set_xlabel("Modal Frequency [Hz]")
    axes_res[1].set_ylabel("|Residue|")
    axes_res[1].set_title("Residue vs Modal Frequency")
    axes_res[1].legend()
    axes_res[1].grid(True, alpha=0.3)

    fig_residue.tight_layout()
    return


@app.cell
def cell_pole_plot(np, plt, poles, poles_i):
    _theta = np.linspace(0, 2 * np.pi, 500)
    fig_poles, axes_poles = plt.subplots(1, 2, figsize=(12, 5))

    for ax_p, p, label_z in zip(
        axes_poles,
        [poles_i, poles],
        ["Random Init", "Optimized"],
    ):
        ax_p.plot(np.cos(_theta), np.sin(_theta), "k--", lw=0.8, alpha=0.5, label="Unit circle")
        sc = ax_p.scatter(p.real, p.imag, s=4, c=np.abs(p), cmap="plasma", alpha=0.7)
        fig_poles.colorbar(sc, ax=ax_p, label="|pole| (decay rate)")
        ax_p.set_aspect("equal")
        ax_p.set_xlabel("Re")
        ax_p.set_ylabel("Im")
        ax_p.set_title(f"Poles — {label_z} ({len(p)} modes)")
        ax_p.grid(True, alpha=0.2)

    fig_poles.tight_layout()
    fig_poles
    return


@app.cell
def cell_residue_scatter(
    fs,
    np,
    plt,
    poles,
    poles_i,
    pyFDN,
    residues,
    residues_i,
):
    _res_mag = np.abs(residues[:, 0, 0])
    _res_mag_i = np.abs(residues_i[:, 0, 0])
    _freq = np.abs(np.angle(poles)) / np.pi * (fs / 2)
    _freq_i = np.abs(np.angle(poles_i)) / np.pi * (fs / 2)

    fig_res_sc, axes_res_sc = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax_r, freq, res, label_r in zip(
        axes_res_sc,
        [_freq_i, _freq],
        [_res_mag_i, _res_mag],
        ["Random Init", "Optimized"],
    ):
        ax_r.scatter(freq, pyFDN.lin_to_db(res), s=4, alpha=0.7)
        ax_r.set_xlabel("Modal Frequency [Hz]")
        ax_r.set_ylabel("|Residue|")
        ax_r.set_title(f"Residue Magnitudes — {label_r} ({len(res)} modes)")
        ax_r.grid(True, alpha=0.3)

    fig_res_sc.tight_layout()
    fig_res_sc
    return


@app.cell
def cell_spectrogram(fs, ir_init, ir_optim, plt):
    fig_spec, axes_spec = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax_s, ir, label in zip(
        axes_spec,
        [ir_init, ir_optim],
        ["Random Init", "Optimized"],
    ):
        ax_s.specgram(ir, Fs=fs, NFFT=2048, noverlap=1792, cmap="inferno", vmin=-120, vmax=-20)
        ax_s.set_xlabel("Time [s]")
        ax_s.set_ylabel("Frequency [Hz]")
        ax_s.set_title(label)
        ax_s.set_ylim(0, fs / 2)
    fig_spec.tight_layout()
    fig_spec
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
