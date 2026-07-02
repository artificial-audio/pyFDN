# gallery_category: Allpass FDN Examples

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
    # Allpass FDN completion

    For a given feedback matrix **A**, the goal is to construct **b**, **c**, and **d** such that the FDN is uniallpass.

    See *{paper_link("Allpass_Feedback_Delay_Networks")}.*

    — Original MATLAB: Sebastian J. Schlecht, 9 June 2020
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    import numpy as np
    import scipy

    import pyFDN

    np.random.seed(3)
    return np, pyFDN, scipy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: complete allpass FDN (full MIMO)

    Given **A** from a random orthogonal matrix, use **complete_fdn** with **k = N** to get **B, C, D** so that **V = [A,B;C,D]** is orthogonal. Check with **is_uniallpass**.
    """)
    return


@app.cell
def _(np, pyFDN):
    _N = 12
    _num_io = _N
    _U = pyFDN.random_orthogonal(_N)
    _G = np.diag(np.random.rand(_N))
    _A = _U @ _G
    _B, _C, _D, _X = pyFDN.complete_fdn(_A, k=_num_io)
    _is_a, _P = pyFDN.is_uniallpass(_A, _B, _C, _D, tol=1e-07)
    assert _is_a, "Completed system should be uniallpass"
    pyFDN.plot_system_matrix(_A, _B, _C, _D)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: complete diagonally similar to orthogonal (MIMO)

    **A** is diagonally similar to an orthogonal block; use **complete_fdn** with **k=1** and check with **is_uniallpass**.
    """)
    return


@app.cell
def _(np, pyFDN, scipy):
    _N = 3
    _num_io = 2
    X1 = scipy.linalg.block_diag(np.diag(np.random.rand(_N)), np.eye(_num_io))
    V = pyFDN.random_orthogonal(_N + _num_io)
    XVX = np.linalg.solve(X1, V @ X1)
    XAX = XVX[:_N, :_N]
    _B, _C, _D, _X = pyFDN.complete_fdn(XAX, k=_num_io)
    _is_a, _P = pyFDN.is_uniallpass(XAX, _B, _C, _D, tol=1e-07)
    assert _is_a, "Completed system should be uniallpass"
    pyFDN.plot_system_matrix(XAX, _B, _C, _D)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: complete series allpass (SISO)

    **A** from Schroeder series allpass; use **complete_fdn** with **k=1** and check with **is_uniallpass**.
    """)
    return


@app.cell
def _(np, pyFDN):
    _N = 4
    _g = np.random.uniform(0.5, 0.99, _N)
    _A, _, _, _ = pyFDN.series_allpass(_g)
    _B, _C, _D, _X = pyFDN.complete_fdn(_A, k=1)
    _is_a, _P = pyFDN.is_uniallpass(_A, _B, _C, _D, tol=1e-07)
    assert _is_a, "Completed system should be uniallpass"
    pyFDN.plot_system_matrix(_A, _B, _C, _D)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: nested allpass (SISO)

    **A** from nested allpass; use **complete_fdn** with **k=1** and check with **is_uniallpass**. (Completion can sometimes fail for poor singular-value structure.)
    """)
    return


@app.cell
def _(np, pyFDN):
    _N = 3
    _g = np.random.uniform(0.5, 0.99, _N)
    _A, _, _, _ = pyFDN.nested_allpass(_g)
    _B, _C, _D, _X = pyFDN.complete_fdn(_A, k=1)
    _is_a, _P = pyFDN.is_uniallpass(_A, _B, _C, _D, tol=1e-07)
    assert _is_a, "Completed system should be uniallpass"
    pyFDN.plot_system_matrix(_A, _B, _C, _D)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Homogeneous allpass with random admissible X

    Random delays and gain matrix **G**, then random admissible diagonal **X** (Python translation of `randAdmissibleHomogeneousAllpass`), and **homogeneous_allpass_fdn(G, X)**. Then complete feedback matrix with **complete_fdn** and check again with **is_uniallpass**.
    """)
    return


@app.cell
def _(np, pyFDN):
    # Homogeneous allpass FDN with random admissible diagonal X
    _N = 4
    delays = np.random.randint(1, 31, size=_N)  # delays in samples, 1..30
    _g = 0.99
    _G = np.diag(_g**delays)  # global gain per sample
    _X = pyFDN.rand_admissible_homogeneous_allpass(_G, (0.7, 0.999))  # gain matrix
    _X @ _G @ _G
    _A, b, c, d, _U = pyFDN.homogeneous_allpass_fdn(_G, _X)
    is_a0, P0 = pyFDN.is_uniallpass(_A, b, c, d, tol=1e-07)
    assert is_a0, "Completed system should be uniallpass"
    _B, _C, _D, _X = pyFDN.complete_fdn(_A, k=1)
    _is_a, _P = pyFDN.is_uniallpass(_A, _B, _C, _D, tol=1e-07)
    assert _is_a, "Completed system should be uniallpass"
    pyFDN.plot_system_matrix(_A, _B, _C, _D)
    return


if __name__ == "__main__":
    app.run()
