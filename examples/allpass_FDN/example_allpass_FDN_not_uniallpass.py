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
    # Allpass but not uniallpass FDN

    Example of an FDN that is **allpass only for specific delay lengths**, not for arbitrary delays. So it is not *uniallpass* (allpass for any choice of delays).

    We build a SISO system by taking a random orthogonal system matrix and applying a **non-diagonal similarity transform** (on the first two delays). The resulting FDN is allpass (and stable) for some delay vectors and not for others.

    See {paper_link("Allpass_Feedback_Delay_Networks")}.

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
    from scipy.linalg import orth

    import pyFDN

    return np, orth, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build system matrix (orthogonal + non-diagonal similarity)

    Start from a random orthogonal (N+1)×(N+1) system matrix **V**, then apply **X⁻¹ V X** with **X** non-diagonal so the result is no longer uniallpass.
    """)
    return


@app.cell
def _(np, orth):
    np.random.seed(1)
    N = 3

    V = orth(np.random.randn(N + 1, N + 1))

    X = np.eye(N + 1)
    X[0, 0] = -0.1
    X[1, 0] = 0.5

    # V = np.linalg.solve(X, V @ X)
    V = np.linalg.inv(X) @ V @ X

    A = V[:-1, :-1]
    b = V[:-1, -1:]
    c = V[-1:, :-1]
    d = V[-1:, -1:]

    print("A =", A)
    print("b =", b.T)
    print("c =", c)
    print("d =", d)
    return A, b, c, d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: allpass and stable for m = [1, 1, 1]

    For this delay vector the system is allpass and the denominator roots lie inside the unit circle (stable).
    """)
    return


@app.cell
def _(A, b, c, d, np, pyFDN):
    _m = np.array([1, 1, 1])
    _is_a, _den, _num = pyFDN.is_allpass(A, b, c, d, _m)

    _roots_den = np.roots(_den)
    _stable = np.all(np.abs(_roots_den) < 1)
    assert _stable, "Expected stable (|roots| < 1)"
    assert _is_a, "Expected allpass for m = [1,1,1]"
    print("Allpass: OK, stable: OK")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: not allpass and unstable for m = [2, 1, 1]

    For this delay vector the system is **not** allpass and has at least one pole outside the unit circle (unstable).
    """)
    return


@app.cell
def _(A, b, c, d, np, pyFDN):
    _m = np.array([2, 1, 1])
    _is_a, _den, _num = pyFDN.is_allpass(A, b, c, d, _m)

    _roots_den = np.roots(_den)
    unstable = np.any(np.abs(_roots_den) > 1)
    assert unstable, "Expected unstable for m = [2,1,1]"
    assert not _is_a, "Expected not allpass for m = [2,1,1]"
    print("Not allpass: OK, unstable: OK")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: allpass and stable for m = [2, 2, 1]

    For this delay vector the system is again allpass and stable.
    """)
    return


@app.cell
def _(A, b, c, d, np, pyFDN):
    _m = np.array([2, 2, 1])
    _is_a, _den, _num = pyFDN.is_allpass(A, b, c, d, _m)

    _roots_den = np.roots(_den)
    _stable = np.all(np.abs(_roots_den) < 1)
    assert _stable, "Expected stable (|roots| < 1)"
    assert _is_a, "Expected allpass for m = [2,2,1]"
    print("Allpass: OK, stable: OK")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: not uniallpass

    The system is allpass only for specific delay vectors (e.g. [1,1,1] and [2,2,1]), not for arbitrary delays. So it is **not uniallpass**. `is_uniallpass` checks the Lyapunov matrix and system matrix structure; it should return False here.
    """)
    return


@app.cell
def _(A, b, c, d, pyFDN):
    is_uni, P = pyFDN.is_uniallpass(A, b, c, d)
    assert not is_uni, "Expected not uniallpass (allpass only for specific delays)"
    print("Not uniallpass: OK")
    return


if __name__ == "__main__":
    app.run()
