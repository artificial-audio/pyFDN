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
    # FDN Gallery

    Overview of feedback matrices and full FDN systems available in pyFDN.

    `fdn_matrix_gallery` provides pure feedback matrices — each is an N×N matrix
    used as the recirculation matrix in a delay network. Losslessness properties are
    characterised by orthogonality (`A @ A.T ≈ I`) and unilosslessness (diagonal
    similarity to an orthogonal matrix).

    `fdn_system_gallery` provides complete state-space systems `(A, B, C, D)` —
    structures where the input/output coupling is part of the design, such as
    series allpass, nested allpass, and the Schroeder reverberator. These are
    checked for the stronger uniallpass condition.

    Reference: *Schlecht, "FDNTB: The Feedback Delay Network Toolbox," DAFx-20, 2020.*

    Original MATLAB: Sebastian J. Schlecht, 28 December 2019.
    """)
    return


@app.cell
def _():
    import plotly.io as pio

    import pyFDN

    pio.renderers.default = "sphinx_gallery"
    return (pyFDN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Retrieve all matrix types

    Query `fdn_matrix_gallery()` with no arguments to get the list of available types.
    """)
    return


@app.cell
def _(pyFDN):
    N = 8
    matrix_types = pyFDN.fdn_matrix_gallery()
    print(f"Available types ({len(matrix_types)}):")
    for t in matrix_types:
        print(f"  {t}")
    return N, matrix_types


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build matrices and check losslessness

    A matrix is orthogonal if `A @ A.T ≈ I`.  A matrix is spectrally lossless if
    all eigenvalues lie on the unit circle.  The table below shows both checks.
    """)
    return


@app.cell
def _(N, matrix_types, mo, pyFDN):
    results = {}
    for mtype in matrix_types:
        try:
            A = pyFDN.fdn_matrix_gallery(N, mtype)
            results[mtype] = {
                "matrix": A,
                "orthogonal": pyFDN.is_orthogonal(A),
                "unilossless": pyFDN.is_unilossless(A),  # TODO: fails on a few matrices
            }
        except NotImplementedError:
            results[mtype] = {"matrix": None, "orthogonal": None, "unilossless": None}

    table = mo.ui.table(
        [
            {
                "Type": mtype,
                "Orthogonal": r["orthogonal"],
                "Unilossless": r["unilossless"],
            }
            for mtype, r in results.items()
        ]
    )
    return table, results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrix heatmaps

    Visualise each matrix as a colour-coded grid.
    """)
    return


@app.cell
def _(mo, pyFDN, results):
    implemented = [(t, r) for t, r in results.items() if r["matrix"] is not None]
    figs = [
        pyFDN.plot_matrix(
            r["matrix"],
            title=f"{t}<br><sup>orth={r['orthogonal']}, uni={r['unilossless']}</sup>",
        )
        for t, r in implemented
    ]
    mo.hstack(figs, wrap=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # FDN System Gallery

    Overview of full FDN system types available in `pyFDN.fdn_system_gallery`.
    Each type returns a complete state-space system `(A, B, C, D)` visualised
    as the block matrix `[A  b; c  d]`.
    """)
    return


@app.cell
def _(pyFDN):
    system_types = pyFDN.fdn_system_gallery()
    print(f"Available system types ({len(system_types)}):")
    for _t in system_types:
        print(f"  {_t}")
    return (system_types,)


@app.cell
def _(N, mo, pyFDN, system_types):
    sys_results = {}
    for stype in system_types:
        try:
            s = pyFDN.fdn_system_gallery(N, stype)
            is_ua, _ = pyFDN.is_uniallpass(s.A, s.B, s.C, s.D)
            sys_results[stype] = {"system": s, "uniallpass": is_ua}
        except NotImplementedError:
            sys_results[stype] = None

    sys_figs = [
        pyFDN.plot_system_matrix(
            *sys_results[stype]["system"],
            title=f"{stype}<br><sup>uniallpass={sys_results[stype]['uniallpass']}</sup>",
        )
        for stype in system_types
        if sys_results[stype] is not None
    ]
    mo.hstack(sys_figs, wrap=True)
    return


if __name__ == "__main__":
    app.run()
