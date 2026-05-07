import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interpolate between two orthogonal matrices

    Interpolate between two orthogonal matrices so that each interpolant is orthogonal (geodesic on the orthogonal group). Then use three of these matrices as FDN feedback matrices and plot their impulse responses via `pyFDN.dss2impz`.

    Reference: *Schlecht, S., Habets, E. (2015). Practical considerations of time-varying feedback delay networks.* Proc. Audio Eng. Soc. Conv.

    - Original version: Sebastian J. Schlecht, Friday, 10. April 2020
    - Translation: Sebastian J. Schlecht, Thursday, 19. February 2026
    """)
    return


@app.cell
def _():
    import math
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "sphinx_gallery"  # interactive in Jupyter + docs HTML
    from scipy.linalg import hadamard
    import pyFDN

    return go, hadamard, math, np, pyFDN


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
def _(C, T):
    # Plotly Express: one call, slider + play (animation_frame = axis to slice)
    import plotly.express as px
    zmin = -1
    zmax = 1
    _fig = px.imshow(C, animation_frame=2, origin='upper', range_color=(zmin, zmax), color_continuous_scale='RdBu', color_continuous_midpoint=0, labels=dict(animation_frame='t'))
    _fig.update_layout(height=420)
    _fig.layout.sliders[0].currentvalue = dict(prefix='t = ', visible=True)
    for k, step in enumerate(_fig.layout.sliders[0].steps):
        step['label'] = f'{T[k]:.2f}'
    # Optional: label slider with T values
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
def _(C, N, T, go, np, num_t, pyFDN):
    ir_len = 2000
    delays = np.array([101, 163, 197, 241], dtype=np.int64)
    g = 0.999
    b = np.ones((N, 1))
    c = np.ones((1, N))
    d = np.array([[0.0]])
    idx = [0, num_t // 3 * 2, num_t - 1]
    # Pick three interpolants: start (A), middle, end (B)
    labels = [f't={T[i]:.2f}' for i in idx]
    irs = []
    for i in idx:
        Af = C[:, :, i] @ np.diag(g ** delays)
        ir = pyFDN.dss_to_impz(ir_len, delays, Af, b, c, d)  # feedback matrix with gain per delay
        ir = np.asarray(ir).squeeze().ravel()
        irs.append(ir)
    _fig = go.Figure()
    for i, (ir, label) in enumerate(zip(irs, labels)):
        _fig.add_trace(go.Scatter(y=pyFDN.mulaw_encode(ir) + 2 * i, mode='lines', name=label, line=dict(width=1.5)))
    _fig.update_layout(title='FDN impulse response for three interpolated feedback matrices', xaxis_title='Time [samples]', yaxis_title='Amplitude [μ-law]', legend_title='Interpolation', hovermode='x unified', template='plotly_white', height=400)
    _fig.show()
    return


if __name__ == "__main__":
    app.run()
