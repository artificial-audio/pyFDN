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
    # Interpolate between two orthogonal matrices

    Interpolate between two orthogonal matrices so that each interpolant is orthogonal (geodesic on the orthogonal group). Then use three of these matrices as FDN feedback matrices and plot their impulse responses via `pyFDN.dss2impz`.

    Reference: *{paper_link("Schlecht2015PracticalConsiderationsTimevarying")}* .

    - Original version: Sebastian J. Schlecht, Friday, 10. April 2020
    - Translation: Sebastian J. Schlecht, Thursday, 19. February 2026
    """)
    return


@app.cell
def _():
    import math

    import numpy as np
    import plotly.io as pio

    pio.renderers.default = "sphinx_gallery"  # interactive in Jupyter + docs HTML
    from scipy.linalg import hadamard

    import pyFDN

    return hadamard, math, np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameters and interpolation

    Start from a Hadamard matrix (sign-normalized) and the identity; interpolate along 20 steps and verify each interpolant is orthogonal.
    """)
    return


@app.cell
def _(hadamard, math, np, pyFDN):
    N = 4
    # Hadamard, normalized and sign-normalized on diagonal (match MATLAB fdnMatrixGallery)
    A = hadamard(N) / math.sqrt(N)
    A = A @ np.diag(np.sign(np.diag(A)))
    B = np.eye(N)

    num_t = 20
    T = np.linspace(0, 1, num_t)
    C = np.zeros((N, N, num_t))
    for it, t in enumerate(T):
        C[:, :, it] = pyFDN.interpolate_orthogonal(A, B, t)

    is_orth = [pyFDN.is_orthogonal(C[:, :, it]) for it in range(num_t)]
    assert all(is_orth), "All interpolants should be orthogonal"
    print("All interpolants are orthogonal.")
    return C, N, T, num_t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Animation of the interpolated feedback matrix

    Animate **C(t)** as a heatmap; use the slider or play button to move along the geodesic from A (t=0) to B (t=1).
    """)
    return


@app.cell
def _(C, T, pyFDN):
    # animate one plot_matrix heatmap per time step (slider + play built in).
    _fig = pyFDN.animate(
        lambda M: pyFDN.plot_matrix(M, zmin=-1, zmax=1),
        [C[:, :, k] for k in range(C.shape[2])],
        labels=T,
        label_prefix="t = ",
        label_format=".2f",
    )
    _fig.update_layout(height=420, width=420)
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three impulse responses via dss2impz

    Use three feedback matrices (t=0, t=0.63, t=1) in a minimal FDN and compute the impulse response for each with `pyFDN.dss2impz`. Plot the three IRs (mu-law encoded for visibility).
    """)
    return


@app.cell
def _(C, N, T, np, num_t, pyFDN):
    ir_len = 2000
    delays = np.array([101, 163, 197, 241], dtype=np.int64)
    g = 0.999
    b = np.ones((N, 1))
    c = np.ones((1, N))
    d = np.array([[0.0]])
    idx = [0, num_t // 3 * 2, num_t - 1]
    # Pick three interpolants: start (A), middle, end (B)
    labels = [f"t={T[i]:.2f}" for i in idx]
    irs = []
    for i in idx:
        Af = C[:, :, i] @ np.diag(g**delays)
        ir = pyFDN.dss_to_impz(
            ir_len, delays, Af, b, c, d
        )  # feedback matrix with gain per delay
        ir = np.asarray(ir).squeeze().ravel()
        irs.append(ir)
    pyFDN.plot_impulse_response(
        *irs,
        labels=labels,
        title="FDN impulse response for three interpolated feedback matrices",
    )
    return


if __name__ == "__main__":
    app.run()
