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
    # Nearest Sign-Agnostic Orthogonal Matrix

    Given a non-negative matrix **B** (e.g., measured energy flow between delay
    lines), find the orthogonal matrix **U** that minimises ``‖B − |U|‖_F`` where
    ``|·|`` is element-wise absolute value.

    The challenge is assigning the right ±1 sign to each element.  A naive
    approach (just solving the ordinary Procrustes problem with `nearest_orthogonal`)
    ignores the freedom in signs.  The sign-agnostic algorithm does:

    1. Normalise **B** to doubly stochastic via Sinkhorn-Knopp.
    2. Initialise with a random sign pattern.
    3. Alternate: (a) solve nearest-orthogonal via SVD, (b) update signs.
    4. Repeat from 2 with new random initialisations; keep best.

    Reference: *{paper_link("Schlecht2018SignagnosticMatrixDesign")}.*

    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.io as pio

    import pyFDN

    pio.renderers.default = "sphinx_gallery"
    return np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    Generate a random orthogonal matrix **A** and strip its signs to get **B = |A|**.
    The goal is to recover a matrix close to **A** from **B** alone.
    """)
    return


@app.cell
def _(np, pyFDN):
    np.random.seed(42)
    N = 8

    # Ground-truth orthogonal matrix
    A_original = pyFDN.random_orthogonal(N)
    # Target: absolute values only (signs removed)
    B = np.abs(A_original)
    return A_original, B, N


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solve

    Compare the naive **nearest_orthogonal** (ignores sign freedom) against the
    **sign-agnostic** solution.
    """)
    return


@app.cell
def _(B, N, np, pyFDN):
    classic = pyFDN.nearest_orthogonal(B)
    sign_agnostic = pyFDN.nearest_sign_agnostic_orthogonal(B, max_trials=10_000)

    error_classic = np.linalg.norm(np.abs(classic) - B, "fro") / N**2
    error_sign_agnostic = np.linalg.norm(np.abs(sign_agnostic) - B, "fro") / N**2

    print(f"Classic Procrustes error:      {error_classic:.6f}")
    print(f"Sign-agnostic Procrustes error: {error_sign_agnostic:.6f}")
    print(f"Improvement factor: {error_classic / error_sign_agnostic:.1f}×")

    assert pyFDN.is_orthogonal(classic, tol=1e-8), "classic solution not orthogonal"
    assert pyFDN.is_orthogonal(sign_agnostic, tol=1e-8), "sign-agnostic not orthogonal"
    assert error_sign_agnostic <= error_classic + 1e-10, (
        "sign-agnostic should not be worse"
    )
    return classic, sign_agnostic


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualise

    Plot the three matrices side by side: original **A**, input **|A|**, and the two solutions.
    """)
    return


@app.cell
def _(A_original, B, classic, np, pyFDN, sign_agnostic):
    def align_signs(M, ref):
        M = M * np.sign(np.sum(M * ref, axis=1, keepdims=True))  # rows
        M = M * np.sign(np.sum(M * ref, axis=0, keepdims=True))  # cols
        return M

    classic_aligned = align_signs(classic, A_original)
    sign_agnostic_aligned = align_signs(sign_agnostic, A_original)

    def orth_label(M):
        return "orth ✓" if pyFDN.is_orthogonal(M, tol=1e-8) else "orth ✗"

    matrices = [
        A_original,
        B,
        sign_agnostic_aligned,
        np.abs(sign_agnostic_aligned),
        classic_aligned,
        np.abs(classic_aligned),
    ]
    titles = [
        f"Original A ({orth_label(A_original)})",
        "|A| (input)",
        f"Sign-agnostic aligned<br>({orth_label(sign_agnostic_aligned)})",
        "|Sign-agnostic aligned|",
        f"Nearest orthogonal aligned<br>({orth_label(classic_aligned)})",
        "|Nearest orthogonal aligned|",
    ]

    fig = pyFDN.plot_matrix_grid(
        matrices,
        titles=titles,
        ncols=2,
        zmin=-1,
        zmax=1,
        show_ticks=True,
        title="Sign-agnostic vs. classic nearest orthogonal",
        height=1000,
    )
    fig.show()
    return


if __name__ == "__main__":
    app.run()
