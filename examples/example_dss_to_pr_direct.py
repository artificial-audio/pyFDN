# gallery_category: Translation Examples

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
    # Direct DSS→PR example

    Uses ``dss_to_pr_direct`` (numeric DSS-only path) with modes ``eig``, ``roots``, and ``polyeig``.
    Compares time-domain IR from ``dss_to_impz`` with modal reconstruction from each mode.
    """)
    return


@app.cell
def _():
    import numpy as np

    import pyFDN

    return np, pyFDN


@app.cell
def _(np, pyFDN):
    np.random.seed(11)

    delays = np.array([41, 53, 67, 79], dtype=int)
    build = pyFDN.fdn_build_gallery(
        delays=delays,
        io_type="identity",
        direct_gain=1.0,
        rt60=None,
        rng=11,
    )
    # Uniform feedback attenuation for a stable, decaying system to analyse.
    A = 0.65 * build.A
    b, c, d, delays = build.B, build.C, build.D, build.delays
    return A, b, c, d, delays


@app.cell
def _(A, b, c, d, delays, np, pyFDN):
    ir_len = 1024
    ir_time = pyFDN.dss_to_impz(ir_len, delays, A, b, c, d)[:, 0, 0]

    ir_modals = {}
    modes = ["eig", "roots", "polyeig"]
    for _mode in modes:
        residues, poles, direct, is_pair, _ = pyFDN.dss_to_pr_direct(
            delays, A, b, c, d, mode=_mode
        )
        ir_modals[_mode] = pyFDN.pr_to_impz(residues, poles, direct, is_pair, ir_len)[
            :, 0, 0
        ]
        err = np.max(np.abs(ir_time - ir_modals[_mode]))
        print(f"{_mode}: max |IR_time - IR_modal| = {err}")
    return ir_modals, ir_time, modes


@app.cell
def _(ir_modals, ir_time, modes, pyFDN):
    pyFDN.plot_impulse_response(
        ir_time,
        *(ir_modals[_mode] for _mode in modes),
        labels=["IR from dss_to_impz"] + [f"IR from {_mode}" for _mode in modes],
        title="DSS time response vs modal reconstruction",
    )
    return


if __name__ == "__main__":
    app.run()
