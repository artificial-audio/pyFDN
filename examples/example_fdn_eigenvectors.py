# gallery_category: Analysis & Verification
# gallery_description: Compute and visualize FDN mode shapes from the left and right eigenvectors of the loop polynomial.

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
    # FDN eigenvectors (mode shapes)

    Demonstrates how to compute the mode shapes of an FDN from the left and right eigenvectors of the loop polynomial $P(z) = D_m(z) - A$.

    Each residue factors into the input/output drive and an undriven part:

    $$\rho_i = \frac{(c\, r_i)\,(l_i^H b)}{l_i^H P'(\lambda_i)\, r_i},$$

    where $r_i$ and $l_i$ are the right/left null vectors of $P(\lambda_i)$. The eigenvectors live on the delay lines; expanding each entry along its delay line with $\lambda_i^k$ gives the mode shape over the full state.
    """)
    return


@app.cell(hide_code=True)
def _(mo, pyFDN):
    mo.md(f"""
    Reference: *{pyFDN.paper_link("Schlecht2024ModalExcitationFeedback")}.*

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
    ## Define FDN and modal decomposition
    """)
    return


@app.cell
def _(np, pyFDN):
    np.random.seed(1)
    delays = np.array([13, 19, 23])
    build = pyFDN.fdn_build_gallery(
        delays=delays,
        io_type="ones",
        direct_gain=0.0,
        rt=None,
        rng=1,
    )
    # Bake delay-proportional broadband decay into the lossless feedback matrix.
    delays = build.delays
    A = np.diag(0.98**delays) @ build.A
    b, c, d = build.B, build.C, build.D
    num_delays = delays.size
    return A, b, c, d, delays, num_delays


@app.cell
def _(A, b, c, d, delays, pyFDN):
    residues, poles, direct, is_pair, meta = pyFDN.dss_to_pr(delays, A, b, c, d)
    num_modes = poles.size
    rv = meta["eigenvectors"]["right"]  # (N, num_modes)
    lv = meta["eigenvectors"]["left"]
    undriven = meta["undrivenResidues"]
    print(f"Number of modes (conjugate pairs reduced): {num_modes}")
    return direct, is_pair, lv, num_modes, poles, residues, rv, undriven


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Residues from eigenvectors

    Reassemble the residues from the eigenvectors and the undriven part; the result matches the residues returned by the modal decomposition.
    """)
    return


@app.cell
def _(b, c, lv, np, residues, rv, undriven):
    res_compact = undriven * (c @ rv).ravel() * (lv.conj().T @ b).ravel()
    max_residue_error = np.max(np.abs(residues[:, 0, 0] - res_compact))
    print(f"Max |residue - compact residue| = {max_residue_error:.3e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impulse response check

    Compare the time-domain recursion with the modal synthesis.
    """)
    return


@app.cell
def _(A, b, c, d, delays, direct, is_pair, np, plt, poles, pyFDN, residues):
    ir_len = 1000
    ir_time = pyFDN.dss_to_impz(delays, A, b, c, d, ir_len)[:, 0, 0]
    ir_modal = pyFDN.pr_to_impz(residues, poles, direct, is_pair, ir_len)[:, 0, 0]

    fig_ir, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(ir_time, label="Impulse response (time domain)")
    _ax.plot(ir_modal + 1.0, label="IR pole residue (+1 offset)")
    _ax.set_title(
        "Time domain vs modal synthesis "
        f"(max err = {np.max(np.abs(ir_time - ir_modal)):.2e})"
    )
    _ax.set_xlabel("Time (samples)")
    _ax.set_ylabel("Impulse response value")
    _ax.legend()
    fig_ir.tight_layout()
    fig_ir
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mode shapes over the full state space

    Expand the eigenvectors along each delay line: state $k$ of delay line $j$ carries $r_{j,i}\,\lambda_i^k$. Horizontal lines mark the delay-line boundaries.
    """)
    return


@app.cell
def _(delays, np, num_delays, plt, poles, rv):
    rv_state_blocks = []
    for _j in range(num_delays):
        _powers = poles[None, :] ** np.arange(delays[_j])[:, None]
        rv_state_blocks.append(rv[_j, :][None, :] * _powers)
    rv_state = np.vstack(rv_state_blocks)  # (sum(delays), num_modes)

    fig_state, _ax = plt.subplots(figsize=(8, 5.2))
    _lim = np.max(np.abs(np.real(rv_state)))
    _image = _ax.imshow(
        np.real(rv_state),
        cmap="RdBu",
        vmin=-_lim,
        vmax=_lim,
        aspect="auto",
        interpolation="nearest",
    )
    for _boundary in np.cumsum(delays)[:-1]:
        _ax.axhline(_boundary - 0.5, color="black", linewidth=2)
    fig_state.colorbar(_image, ax=_ax, label="Re")
    _ax.set_title("Right eigenvectors expanded over the state space")
    _ax.set_xlabel("Eigenvalue index i")
    _ax.set_ylabel("State space index")
    fig_state.tight_layout()
    fig_state
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Right eigenvectors on the delay lines

    The raw right eigenvectors, one row per delay line.
    """)
    return


@app.cell
def _(np, plt, rv):
    fig_rv, _ax = plt.subplots(figsize=(8, 3))
    _lim = np.max(np.abs(np.real(rv)))
    _image = _ax.imshow(
        np.real(rv),
        cmap="RdBu",
        vmin=-_lim,
        vmax=_lim,
        aspect="auto",
        interpolation="nearest",
    )
    fig_rv.colorbar(_image, ax=_ax, label="Re", orientation="horizontal")
    _ax.set_title("Right eigenvectors (per delay line)")
    _ax.set_xlabel("Eigenvalue index i")
    _ax.set_ylabel("Delay index")
    fig_rv.tight_layout()
    fig_rv
    return


if __name__ == "__main__":
    app.run()
