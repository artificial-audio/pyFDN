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

    Reference: *Schlecht and Habets, "Sign-Agnostic Matrix Design for Spatial
    Artificial Reverberation with Feedback Delay Networks," AES Conf. on Spatial
    Reproduction, 2018.*

    Original MATLAB: Sebastian J. Schlecht, 29. January 2020.
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    import pyFDN

    pio.renderers.default = "sphinx_gallery"
    return go, make_subplots, np, pyFDN


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
def _(A_original, B, N, classic, go, make_subplots, np, pyFDN, sign_agnostic):
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

    fig = make_subplots(rows=3, cols=2, subplot_titles=titles)
    tick_vals = list(range(N))
    for idx, (mat, _title) in enumerate(zip(matrices, titles, strict=True)):
        row, col = divmod(idx, 2)
        row, col = row + 1, col + 1
        fig.add_trace(
            go.Heatmap(
                z=mat,
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                showscale=(idx == 5),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(
            tickvals=tick_vals, ticktext=[str(i) for i in tick_vals], row=row, col=col
        )
        fig.update_yaxes(
            tickvals=tick_vals,
            ticktext=[str(i) for i in tick_vals],
            autorange="reversed",
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Sign-agnostic vs. classic nearest orthogonal",
        height=1000,
        template="plotly_white",
    )
    fig.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
