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
    # FLAMO→PR (Notebook)

    **In-depth math documentation** of the refinement fix, with an **SOS filter in the loop** and plotting.

    ## Goals

    1. Build a delay state-space model with a **one-pole absorption (SOS) in the loop** via `dss_to_flamo(..., sos_filter=...)`.
    2. Extract poles/residues via `pyFDN.flamo_to_pr`.
    3. Explain the key math fix: why we evaluate Newton/Ehrlich–Aberth in **w-plane** (`w = z^{-1}`) while FLAMO probing naturally gives derivatives in **z-plane**.
    4. Verify numerically that the derivative identities are consistent.
    5. Verify time-domain impulse response reconstruction from modal data and **plot poles + IR**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Notation
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Characteristic matrix of the recursive loop with feedforward $F(z)$ and feedback $G(z)$
    $$
    P(z) = I - F(z)G(z)
    $$
    In the standard FDN delay form this matches
    $$
    P(z) = I - A\,\mathrm{diag}(z^{-m_1},\dots,z^{-m_N}).
    $$

    - Transfer decomposition:
    $$
    H(z) = C(z)\,P(z)^{-1}F(z)B(z) + D(z).
    $$

    - Poles are roots of
    $$
    \det P(z)=0.
    $$

    We use FLAMO's native recursion probes (`probe_recursion`, `probe_recursion_with_derivative`, `log_det_derivative`, `log_det_derivative_w`) directly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Core decomposition and residue formula

    Given
    $$
    H(z) = C(z)\,P(z)^{-1}F(z)B(z)+D(z),
    $$
    and a simple pole $\lambda_i$ where $\det P(\lambda_i)=0$, define right/left null vectors
    $$
    P(\lambda_i)r_i=0,\qquad \ell_i^H P(\lambda_i)=0.
    $$
    Then the residue matrix is
    $$
    \rho_i = \frac{\left(C(\lambda_i)r_i\right)\left(\ell_i^H F(\lambda_i) B(\lambda_i)\right)}{\ell_i^H\,\frac{dP}{dz}(\lambda_i)\,r_i}.
    $$

    This is the adjugate-free null-vector formula used in `flamo_to_pr`.

    Why it is useful:
    - It works for general matrix-polynomial/rational characteristic forms.
    - It avoids explicitly forming adjugates.
    - It aligns naturally with FLAMO probes for $P(z)$, $dP/dz$, $F(z)$, $B(z)$, $C(z)$, $D(z)$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) Newton term in z-plane

    For simultaneous pole refinement (Ehrlich–Aberth style), we need a scalar Newton-like term. For matrix characteristic equations, the natural choice is
    $$
    N_z(z) = \frac{d}{dz}\log\det P(z).
    $$
    Using Jacobi's formula:
    $$
    \frac{d}{dz}\log\det P(z) = \mathrm{tr}\left(P(z)^{-1}\frac{dP}{dz}(z)\right).
    $$

    This is exactly what FLAMO exposes natively via:
    - `Recursion.log_det_derivative(z)` (z-plane), and
    - `Recursion.log_det_derivative_w(w)` (w-plane).

    When these are available we can avoid numerical `solve+trace` fallback for the Newton term.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) Why switch to w-plane ($w=z^{-1}$)?

    For delay systems, the characteristic expression is naturally a polynomial-like object in $w=z^{-1}$.

    Define
    $$
    q(w) = \det P(1/w).
    $$
    Then poles in z map to roots in w by $w_i = 1/z_i$.

    Chain rule gives the key identity:
    $$
    \frac{d}{dw}\log q(w)
    = \frac{d}{dz}\log\det P(z)\cdot\frac{dz}{dw}
    = N_z(z)\cdot\left(-\frac{1}{w^2}\right)
    = -z^2 N_z(z),\quad z=1/w.
    $$
    So
    $$
    N_w(w):=\frac{q'(w)}{q(w)} = -z^2\,N_z(z).
    $$

    This is exactly the conversion implemented in the refinement step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### EAI form used in code
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    At pole index $i$:

    1. Compute $N_z(z_i)$.
    2. Convert to w-plane Newton term:
    $$
    N_w(w_i) = -z_i^2 N_z(z_i),\quad w_i=1/z_i.
    $$
    3. Deflation in w-plane:
    $$
    D_w(w_i)=\sum_{j\neq i}\frac{1}{w_i-w_j}.
    $$
    4. EAI update in w:
    $$
    w_i^{\text{new}} = w_i - \frac{1}{N_w(w_i)-D_w(w_i)}.
    $$
    5. Map back:
    $$
    z_i^{\text{new}} = 1/w_i^{\text{new}}.
    $$

    Using deflation directly in w-plane is important because roots are distributed more naturally for delay-polynomial structure there.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4) Numerical stability note

    The refinement variable is now always $w$. So we evaluate Newton terms directly as
    $$
    N_w(w)=\frac{d}{dw}\log\det P(1/w)
    $$
    whenever FLAMO exposes `log_det_derivative_w(w)`.

    If only z-domain derivatives are available, we use the exact chain-rule fallback:
    $$
    N_w(w)= -\frac{1}{w^2} N_z(1/w),
    \quad N_z(z)=\frac{d}{dz}\log\det P(z).
    $$

    This keeps the optimization state in one variable (w) and minimizes numerical conversions during iteration.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    import pyFDN

    np.random.seed(7)
    print("pyFDN version:", getattr(pyFDN, "__version__", "unknown"))
    return np, plt, pyFDN, torch


@app.cell
def _(np, pyFDN, torch):
    # Build a small stable FDN in DSS form with an SOS (first-order absorption) in the loop
    Fs = 48000.0
    delays = np.array([531, 673, 798, 977], dtype=int)
    build = pyFDN.fdn_build_gallery(
        fs=Fs,
        delays=delays,
        io_type="identity",
        direct_gain=0.0,
        rt=None,
        rng=7,
    )

    # First-order absorption in the loop: canonical (1, 6, N) SOS bank.
    rt_dc, rt_ny = 0.5, 0.1  # reverb time at DC and Nyquist (seconds)
    sos = pyFDN.first_order_absorption(rt_dc, rt_ny, build.delays, Fs)

    # DSS -> FLAMO with SOS in the loop (delay -> filter -> A).
    nfft = 2**16
    model = pyFDN.dss_to_flamo(
        A=build.A,
        B=build.B,
        C=build.C,
        D=build.D,
        m=build.delays,
        Fs=Fs,
        nfft=nfft,
        shell=True,
        sos_filter=sos,
        dtype=torch.float64,
    )
    return Fs, build.delays, model, sos


@app.cell
def _(model, pyFDN):
    # Run modal decomposition (delays and recursion are read from the model)
    residues, poles, direct, is_pair, meta = pyFDN.flamo_to_pr(
        model,
        quality_threshold=1e-10,
        refinement_tol=1e-10,
        maximum_iterations=80,
        reject_unstable_poles=True,
        deflation_type="fullDeflation",
        verbose=True,
    )
    return direct, is_pair, poles, residues


@app.cell
def _(
    Fs,
    delays,
    direct,
    is_pair,
    mo,
    model,
    np,
    plt,
    poles,
    pyFDN,
    residues,
    sos,
):
    # Reference IR from FLAMO (only way to get the true IR when the loop has an SOS)
    ir_flamo = pyFDN.flamo_time_response(model, fs=int(Fs)).squeeze()
    if ir_flamo.ndim > 1:
        ir_flamo = ir_flamo[:, 0, 0]
    ir_flamo = np.asarray(ir_flamo, dtype=np.float64)

    # Modal reconstruction and comparison
    ir_len = min(20000, len(ir_flamo))
    ir_modal = pyFDN.pr_to_impz(residues, poles, direct, is_pair, ir_len)[:, 0, 0]
    ir_ref = ir_flamo[:ir_len]

    err = np.max(np.abs(ir_ref - ir_modal))
    print("max |IR_flamo - IR_modal|:", err)

    # Plot poles in the complex plane with SOS gain-per-sample curve
    angles_sos, mag_sos = pyFDN.sos_gain_per_sample_curves(sos, delays, nfft=512)
    plt.figure(figsize=(10, 5))
    plt.plot(
        pyFDN.rad_to_hertz(angles_sos, Fs),
        pyFDN.lin_to_db(mag_sos),
        alpha=0.7,
        label="SOS gain per sample",
    )
    plt.scatter(
        pyFDN.rad_to_hertz(np.angle(poles), Fs),
        pyFDN.lin_to_db(np.abs(poles)),
        marker=".",
        color="red",
        label="Poles",
    )
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    # plt.ylim(-1,1)
    plt.title("SOS gain per sample and poles in the complex plane")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    ax1 = plt.gca()

    # Plot FLAMO vs modal IR
    _fig_ir = pyFDN.plot_impulse_response(
        ir_ref,
        ir_modal,
        labels=["IR from FLAMO", "IR from poles/residues"],
        title="FLAMO time response vs modal reconstruction",
    )

    mo.vstack([ax1, _fig_ir])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5) Practical interpretation of the fix

    ### What changed conceptually

    - **Before:** parts of the refinement were still expressed in z-domain and then converted for w updates.
    - **Now (fully w-domain):**
      - Root variables are represented as $w$ throughout refinement.
      - Newton term is computed as $d/dw\,\log\det P(1/w)$ (native FLAMO when available, chain-rule fallback otherwise).
      - Deflation and EAI update are done directly in $w$.
      - Conversion to $z=1/w$ happens only once at the end, before residue computation.

    ### Why this is better

    1. **Single-variable formulation:** easier to reason about and debug.
    2. **Closer to delay-polynomial structure:** delays are naturally polynomial in $w=z^{-1}$.
    3. **Numerical robustness:** avoids repeated z↔w conversion during iteration.
    4. **Lower layering:** direct calls into FLAMO recursion APIs, fewer adaptation layers.

    If you want, the next extension is to add a side-by-side diagnostic cell comparing convergence traces of the new fully-w formulation against a legacy z-formulation.
    """)
    return


if __name__ == "__main__":
    app.run()
