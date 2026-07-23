import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import block_diag, expm

    import pyFDN
    from pyFDN.dsp.time_varying_matrix import TimeVaryingMatrix

    return TimeVaryingMatrix, block_diag, expm, mo, np, plt, pyFDN


@app.cell
def _(mo):
    mo.md(r"""
    # Three coupled rooms: does the GFDN track the physical Markov process?

    Two topologies, same three physical rooms:

    - **Chain**: room 1 ↔ room 2 ↔ room 3, no direct 1–3 aperture ($S_{13}=0$).
    - **Complete**: all three rooms mutually coupled ($S_{13}>0$ too).

    Pipeline: physical dimensions $\to$ Cremer–Müller generator $\mathbf Q$ $\to$ pairwise angles
    $\theta_{ij}$ (rate-matched via each room's own clock $\Delta t_i\propto V_i$) $\to$ skew-symmetric
    $\mathbf K\to\mathbf R_{\rm room}=\exp(\mathbf K)\to$ lifted into the full GFDN feedback matrix
    $\mathbf A=\bigl[\mathbf R_{\rm room} \otimes I_{N_{tot} / N}]\operatorname{blkdiag}(\mathbf Q_1,\mathbf Q_2,\mathbf Q_3)$.
    We then compare the GFDN's **exact** per-line energy ledger against $\mathbf e(t)=e^{t\mathbf Q}\mathbf e(0)$,
    the true physical energy trajectory — for both topologies, side by side.
    """)
    return


@app.cell
def _(mo):
    V1 = mo.ui.slider(30,
                      400,
                      value=160,
                      step=10,
                      label="room 1 volume $V_1$ (m³)")
    V2 = mo.ui.slider(30,
                      400,
                      value=80,
                      step=10,
                      label="room 2 volume $V_2$ (m³)")
    V3 = mo.ui.slider(30,
                      400,
                      value=120,
                      step=10,
                      label="room 3 volume $V_3$ (m³)")
    S12 = mo.ui.slider(0.1,
                       6.0,
                       value=2.0,
                       step=0.1,
                       label="aperture $S_{12}$ (m²)")
    S23 = mo.ui.slider(0.1,
                       6.0,
                       value=1.5,
                       step=0.1,
                       label="aperture $S_{23}$ (m²)")
    S13 = mo.ui.slider(0.1,
                       6.0,
                       value=0.5,
                       step=0.1,
                       label="aperture $S_{13}$ (m²) — complete case only")

    a1 = mo.ui.slider(0, 1, value = 0.1, step = 0.05, label = "room 1 absorption $a_1$")
    a2 = mo.ui.slider(0, 1, value = 0.3, step = 0.05, label = "room 2 absorption $a_2$")
    a3 = mo.ui.slider(0, 1, value = 0.2, step = 0.05, label = "room 3 absorption $a_3$")

    Nper = mo.ui.slider(4,
                        48,
                        value=24,
                        step=2,
                        label="delay lines per room $N_i$ (equal)")

    avg_delays = mo.ui.slider(100, 10000, value=2500, step=100, label="Avg delay line length in samples")


    dur = mo.ui.slider(0.5,
                       10.0,
                       value=5.0,
                       step=0.25,
                       label="impulse‑response length (s)")
    mo.md(
        f"### Controls\n{V1}\n\n{V2}\n\n{V3}\n\n{S12}\n\n{S23}\n\n{S13}\n\n{a1}\n\n{a2}\n\n{a3}\n\n{Nper}\n\n{avg_delays}\n\n{dur}"
    )
    return Nper, S12, S13, S23, V1, V2, V3, a1, a2, a3, avg_delays, dur


@app.cell
def _(S12, S13, S23, V1, V2, V3, a1, a2, a3, mo, np):
    # --- Cremer-Muller generators for the two topologies -----------------------
    c = 343.0
    V = np.array([V1.value, V2.value, V3.value])
    absorp = np.array([a1.value, a2.value, a3.value])
    num_rooms = 3

    def make_beta(S, V):
        beta = c * S / (4.0 * V[:, None])
        np.fill_diagonal(beta, 0.0)
        return beta

    def make_Q(beta):
        Q = beta.T.copy()
        np.fill_diagonal(Q, -beta.sum(axis=1))
        return Q

    def make_gamma(absorp, A, V):
        gamma = (c * absorp * A) / (4 * V)
        return np.diag(gamma)

    S_chain = np.array([[0, S12.value, 0], [S12.value, 0, S23.value],
                        [0, S23.value, 0]])
    S_complete = np.array([[0, S12.value,
                            S13.value], [S12.value, 0, S23.value],
                           [S13.value, S23.value, 0]])

    beta_chain, beta_complete = make_beta(S_chain, V), make_beta(S_complete, V)
    Q_chain, Q_complete = make_Q(beta_chain), make_Q(beta_complete)

    # assuming a perfect cube
    _A = 6 * (np.cbrt(V)**2)
    # get the diagonal absorption matrix
    gamma = make_gamma(absorp, _A, V)

    mo.md(rf"""
    ## 0 · Physical generators

    $\mathbf Q_{{\rm chain}}$ (no 1–3 aperture):
    $$\mathbf Q_{{\rm chain}} = \begin{{pmatrix}}{Q_chain[0,0]:.4f}&{Q_chain[0,1]:.4f}&{Q_chain[0,2]:.4f}\\{Q_chain[1,0]:.4f}&{Q_chain[1,1]:.4f}&{Q_chain[1,2]:.4f}\\{Q_chain[2,0]:.4f}&{Q_chain[2,1]:.4f}&{Q_chain[2,2]:.4f}\end{{pmatrix}}\ \mathrm{{s}}^{{-1}}$$

    $\mathbf Q_{{\rm complete}}$ (all three coupled):
    $$\mathbf Q_{{\rm complete}} = \begin{{pmatrix}}{Q_complete[0,0]:.4f}&{Q_complete[0,1]:.4f}&{Q_complete[0,2]:.4f}\\{Q_complete[1,0]:.4f}&{Q_complete[1,1]:.4f}&{Q_complete[1,2]:.4f}\\{Q_complete[2,0]:.4f}&{Q_complete[2,1]:.4f}&{Q_complete[2,2]:.4f}\end{{pmatrix}}\ \mathrm{{s}}^{{-1}}$$

    Both have zero column sums (pure exchange, no absorption in this notebook) and are reversible w.r.t.
    $\pi_i\propto V_i$ — target long‑run split $V_1{{:}}V_2{{:}}V_3 = {V1.value}{{:}}{V2.value}{{:}}{V3.value}$.
    """)
    return Q_chain, Q_complete, V, beta_chain, beta_complete, gamma, num_rooms


@app.cell
def _(Nper, V, avg_delays, mo, np, num_rooms, pyFDN):
    # --- Geometry: equal N per room, M_i proportional to V_i -------------------
    fs = 48000
    Nroom = Nper.value
    N1 = N2 = N3 = Nroom
    Ntot = N1 + N2 + N3

    gscale = V / V[0]
    delay_min = 0.5 * avg_delays.value
    delay_max = 2.0 * avg_delays.value
    delays = []

    for _i in range(num_rooms):
        delays.append(pyFDN.sample_delay_lengths(
            Nroom, (int(delay_min * gscale[_i]), int(delay_max * gscale[_i])), coprime=True, rng=331 + 10*_i))

    delays = np.concatenate(delays).astype(int)
    M1, M2, M3 = int(delays[:N1].sum()), int(delays[N1:N1+N2].sum()), int(delays[N1+N2:].sum())
    delays_per_fdn = np.asarray([delays[:N1], delays[N1:N1+N2], delays[N1+N2:]], dtype=np.int32)
    M = np.asarray([M1, M2, M3])
    dt_i = M / (Nroom * fs)


    mo.md(rf"""
    ## 1 · Geometry — $M_i \propto V_i$, equal $N_i$ (so $\mathbf D=\mathbf I$, no weighting needed)

    | | room 1 | room 2 | room 3 |
    |---|---|---|---|
    | $V_i$ | {V[0]} | {V[1]} | {V[2]}| 
    | $N_i$ | {N1} | {N2} | {N3} |
    | actual $\sum d_j$ | {int(delays[:N1].sum())} | {int(delays[N1:N1+N2].sum())} | {int(delays[N1+N2:].sum())} |
    | $\Delta t_i=M_i/(N_if_s)$ | {dt_i[0]*1000:.3f} ms | {dt_i[1]*1000:.3f} ms | {dt_i[2]*1000:.3f} ms |
    """)
    return N1, N2, N3, Nroom, Ntot, delays, delays_per_fdn, dt_i, fs


@app.cell
def _(beta_chain, beta_complete, dt_i, expm, mo, np, num_rooms):
    # --- Pairwise angles, K, R_room for both topologies -------------------------
    def make_theta(beta):
        n = len(dt_i)
        theta = np.zeros((n, n))
        for _i in range(n):
            for _j in range(_i + 1, n):
                val = 0.5 * (beta[_i, _j] * dt_i[_i] + beta[_j, _i] * dt_i[_j])
                val = min(val, 0.999)
                theta[_i, _j] = theta[_j, _i] = np.arcsin(
                    np.sqrt(val)) if val > 0 else 0.0
        return theta

    def make_K(theta):
        tri = np.triu(theta, 1)
        return tri - tri.T

    theta_chain, theta_complete = make_theta(beta_chain), make_theta(
        beta_complete)
    K_chain, K_complete = make_K(theta_chain), make_K(theta_complete)
    R_room_chain, R_room_complete = expm(K_chain), expm(K_complete)

    # sanity check: beta_ij*dt_i should equal beta_ji*dt_j (design consistency)
    _check = []
    for _i in range(num_rooms):
        for _j in range(_i + 1, num_rooms):
            _check.append(
                (beta_chain[_i, _j] * dt_i[_i], beta_chain[_j, _i] * dt_i[_j]))

    mo.md(rf"""
    ## 2 · Pairwise angles $\theta_{{ij}}$ (rate‑matched, $\sin^2\theta_{{ij}}=\beta_{{ij}}\Delta t_i$)

    | edge | $\theta_{{ij}}$ (chain) | $\theta_{{ij}}$ (complete) |
    |---|---|---|
    | 1-2 | {K_chain[0,1]:.4f} | {K_complete[0,1]:.4f} |
    | 2-1 | {K_chain[1, 0]: .4f} | {K_complete[1, 0]:.4f} |
    | 1-3 | {K_chain[0,2]:.4f} (no aperture) | {K_complete[0,2]:.4f} |
    | 3-1 | {K_chain[2, 0]: .4f} | {K_complete[2, 0]:.4f} |
    | 2-3 | {K_chain[1,2]:.4f} | {K_complete[1,2]:.4f} |
    | 3-2 | {K_chain[2,1]:.4f} | {K_complete[2,1]:.4f} |

    Consistency check ($\beta_{{ij}}\Delta t_i$ vs. $\beta_{{ji}}\Delta t_j$, should match by the $\Delta t_i\propto V_i$
    design): edge 1–2: ${_check[0][0]:.5f}$ vs ${_check[0][1]:.5f}$; edge 1–3: ${_check[1][0]:.5f}$ vs
    ${_check[1][1]:.5f}$; edge 2–3: ${_check[2][0]:.5f}$ vs ${_check[2][1]:.5f}$.
    """)
    return R_room_chain, R_room_complete


@app.cell
def _(
    N1,
    N2,
    N3,
    Nroom,
    Ntot,
    R_room_chain,
    R_room_complete,
    TimeVaryingMatrix,
    block_diag,
    delays_per_fdn,
    fs,
    gamma,
    mo,
    np,
    num_rooms,
    pyFDN,
):
    np.random.seed(1)

    # time varying matrix for faster mixing
    modulation_frequency = 1.0  # hz
    modulation_amplitude = 3.0
    spread = 0.3
    tv_matrix = TimeVaryingMatrix(
            Ntot, modulation_frequency, modulation_amplitude, fs, spread
        )

    Qblocks = block_diag(pyFDN.random_orthogonal(N1),
                         pyFDN.random_orthogonal(N2),
                         pyFDN.random_orthogonal(N3))

    def lift(R_room):
        # Lift into the full GFDN feedback matrix
        U = np.zeros((Ntot, num_rooms))
        U[:N1, 0] = 1.0 / np.sqrt(N1)
        U[N1:N1 + N2, 1] = 1.0 / np.sqrt(N2)
        U[N1 + N2:, 2] = 1.0 / np.sqrt(N3)
        return np.eye(Ntot) + U @ (R_room - np.eye(num_rooms)) @ U.T

    def alt_lift(R_room):
        # Simpler lifting with kroneckers
        mat = np.zeros((num_rooms * Nroom, num_rooms * Nroom))
        for _i in range(num_rooms):
            for _j in range(num_rooms):
                mat[_i*Nroom: (_i+1)*Nroom, _j*Nroom:(_j+1)*Nroom] = R_room[_i, _j] * np.eye(Nroom)
        return mat

    def get_decay(_gamma, _M):
        return np.diag(np.exp(-_gamma * _M / (2 * fs)))

    def get_decay_matrix():
        Gamma = []
        for _i in range(num_rooms):
            Gamma.append(get_decay(gamma[_i, _i], delays_per_fdn[_i]))    
        return block_diag(*Gamma)

    alt_R_chain = alt_lift(R_room_chain)
    alt_R_complete = alt_lift(R_room_complete)
    assert pyFDN.is_orthogonal(alt_R_chain)
    assert pyFDN.is_orthogonal(alt_R_complete)

    A_chain = alt_R_chain @ Qblocks
    A_complete = alt_R_complete @ Qblocks

    Gamma = get_decay_matrix()
    A_chain_lossy = A_chain @ Gamma
    A_complete_lossy = A_complete @ Gamma


    _ok_chain = pyFDN.is_unilossless(A_chain)
    _ok_complete = pyFDN.is_unilossless(A_complete)

    mo.md(rf"""
    ## 3 · The two GFDN feedback matrices

    Same internal room mixers $\mathbf Q_1,\mathbf Q_2,\mathbf Q_3$ for both — only the room‑coupling
    $\mathbf R_{{\rm room}}$ differs. `is_unilossless`: chain = **{_ok_chain}**, complete = **{_ok_complete}**
    (both should be `True` — pure exchange, no absorption).
    """)
    return A_chain, A_chain_lossy, A_complete, A_complete_lossy, tv_matrix


@app.cell
def _(
    A_chain,
    A_chain_lossy,
    A_complete,
    A_complete_lossy,
    N1,
    N2,
    N3,
    Ntot,
    delays,
    dur,
    fs,
    np,
    num_rooms,
    pyFDN,
    tv_matrix,
):
    # --- Impulse responses + exact per-room energy ledger, all rooms excited equally -----
    n_samp = int(dur.value * fs)
    B = np.zeros((Ntot, num_rooms))
    B[:N1, 0] = 1.0 / np.sqrt(N1)
    B[N1:N1+N2, 1] = 1.0 / np.sqrt(N2)
    B[N1+N2:, 2] = 1.0 / np.sqrt(N3)
    # output taken from all rooms
    C_lines = np.eye(Ntot)

    def run(A):
        Y = np.zeros((num_rooms, n_samp, Ntot))
        _x = np.zeros((n_samp, num_rooms))
        for _src in range(num_rooms):
            _x[:] = 0.0
            _x[0, _src] = 1.0
            Y[_src] = pyFDN.process_fdn(_x, delays, A, B, C_lines, np.zeros((Ntot, num_rooms)), extra_matrix=tv_matrix)
        return Y 

    Y_chain = run(A_chain)
    Y_chain_lossy = run(A_chain_lossy)
    Y_complete = run(A_complete)
    Y_complete_lossy = run(A_complete_lossy)

    def ledger(Y):
        cs = np.concatenate([np.zeros(
            (num_rooms, 1, Ntot)), np.cumsum(Y**2, axis=1)],
                            axis=1)
        n_ex = n_samp - int(delays.max()) - 1
        idx = np.arange(n_ex)
        E = np.zeros((num_rooms, n_ex, num_rooms))
        bounds = [0, N1, N1 + N2, N1 + N2 + N3]
        for _j, _m in enumerate(delays):
            _room = 0 if _j < bounds[1] else (1 if _j < bounds[2] else 2)
            E[:, :, _room] += cs[:, idx + 1 + _m, _j] - cs[:, idx + 1, _j]
        return E, n_ex

    E_ex_chain, n_ex = ledger(Y_chain)
    E_ex_chain_lossy, _ = ledger(Y_chain_lossy)
    E_ex_complete, _ = ledger(Y_complete)
    E_ex_complete_lossy, _ = ledger(Y_complete_lossy)
    tsec = np.arange(n_samp) / fs
    return (
        E_ex_chain,
        E_ex_chain_lossy,
        E_ex_complete,
        E_ex_complete_lossy,
        Y_chain_lossy,
        Y_complete_lossy,
        n_ex,
        tsec,
    )


@app.cell
def _(Q_chain, Q_complete, gamma, n_ex, np, tsec):
    # --- True physical trajectory e(t) = expm(Q t) e(0), source = room 1 -------
    def trajectory(Q, D=None):
        if D is None:
            D = np.zeros_like(Q)
        w, Vv = np.linalg.eig(Q - D)
        w, Vv = w.real, Vv.real  # Q is reversible w.r.t. pi ~ V, so spectrum is real
        Vinv = np.linalg.inv(Vv)
        e0 = np.array([1.0, 0.0, 0.0])
        coeffs = Vinv @ e0
        modes = np.exp(np.outer(tsec[:n_ex], w))
        return np.einsum("ik,tk->ti", Vv, modes * coeffs[None, :])

    Eref_chain = trajectory(Q_chain)
    Eref_chain_lossy = trajectory(Q_chain, gamma)

    Eref_complete = trajectory(Q_complete)
    Eref_complete_lossy = trajectory(Q_complete, gamma)
    return Eref_chain, Eref_chain_lossy, Eref_complete, Eref_complete_lossy


@app.cell
def _(
    E_ex_chain,
    E_ex_complete,
    Eref_chain,
    Eref_complete,
    V,
    mo,
    n_ex,
    num_rooms,
    plt,
    tsec,
):
    # --- The comparison plot -----------------------------------------------------
    _fig, _axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    _labels = ["room 1", "room 2", "room 3"]
    _colors = ["C0", "C1", "C2"]
    _targets = V / V.sum()

    for _ax, _Eex, _Eref, _title in zip(
            _axs, [E_ex_chain, E_ex_complete], [Eref_chain, Eref_complete],
        ["chain (1–2, 2–3)", "complete (all coupled)"]):
        for _r in range(num_rooms):
            _ax.plot(tsec[:n_ex],
                     _Eex[0, :, _r],
                     color=_colors[_r],
                     lw=1.1,
                     label=f"GFDN {_labels[_r]}")
            _ax.plot(tsec[:n_ex],
                     _Eref[:, _r],
                     "--",
                     color=_colors[_r],
                     lw=1.0,
                     alpha=0.7)
            _ax.axhline(_targets[_r], color=_colors[_r], lw=0.5, ls=":")
        _ax.set_title(_title, fontsize=10)
        _ax.set_xlabel("time (s)")
        _ax.set_ylim(-0.02, 1.0)
    _axs[0].set_ylabel("fraction of total energy")
    _axs[1].legend(fontsize=7.5, loc="upper right")
    _fig.suptitle(
        "Solid = exact lossless GFDN ledger, dashed = physical $e^{tQ}$, dotted = target $V_i/V_{\\rm tot}$",
        fontsize=9.5)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(
    E_ex_chain_lossy,
    E_ex_complete_lossy,
    Eref_chain_lossy,
    Eref_complete_lossy,
    V,
    mo,
    n_ex,
    num_rooms,
    plt,
    tsec,
):
    _fig, _axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    _labels = ["room 1", "room 2", "room 3"]
    _colors = ["C0", "C1", "C2"]
    _targets = V / V.sum()

    for _ax, _Eex, _Eref, _title in zip(
            _axs, [E_ex_chain_lossy, E_ex_complete_lossy], [Eref_chain_lossy, Eref_complete_lossy],
        ["chain (1–2, 2–3)", "complete (all coupled)"]):
        for _r in range(num_rooms):
            _ax.semilogy(tsec[:n_ex],
                     _Eex[0, :, _r],
                     color=_colors[_r],
                     lw=1.1,
                     label=f"GFDN {_labels[_r]}")
            _ax.semilogy(tsec[:n_ex],
                     _Eref[:, _r],
                     "--",
                     color=_colors[_r],
                     lw=1.0,
                     alpha=0.7)
            _ax.axhline(_targets[_r], color=_colors[_r], lw=0.5, ls=":")
        _ax.set_title(_title, fontsize=10)
        _ax.set_xlabel("time (s)")
        _ax.set_ylim(1e-6, 1.0)
    _axs[0].set_ylabel("fraction of total energy (log scale)")
    _axs[1].legend(fontsize=7.5, loc="upper right")
    _fig.suptitle(
        "Solid = exact lossy GFDN ledger, dashed = physical $e^{t(Q-D)}$, dotted = target $V_i/V_{\\rm tot}$",
        fontsize=9.5)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(
    E_ex_chain,
    E_ex_complete,
    Eref_chain,
    Eref_complete,
    N1,
    N2,
    V,
    delays,
    mo,
    n_ex,
    np,
):
    _tail = slice(int(0.85 * n_ex), n_ex)

    _err_chain = np.max(np.abs(E_ex_chain[0] - Eref_chain))
    _err_complete = np.max(np.abs(E_ex_complete[0] - Eref_complete))

    _gfdn_tail_chain = E_ex_chain[0, _tail].mean(0)
    _gfdn_split_chain = _gfdn_tail_chain / _gfdn_tail_chain.sum()
    _ref_tail_chain = Eref_chain[_tail].mean(0)
    _ref_split_chain = _ref_tail_chain / _ref_tail_chain.sum()

    _gfdn_tail_complete = E_ex_complete[0, _tail].mean(0)
    _gfdn_split_complete = _gfdn_tail_complete/ _gfdn_tail_complete.sum()
    _ref_tail_complete = Eref_complete[_tail].mean(0)
    _ref_split_complete = _ref_tail_complete / _ref_tail_complete.sum()

    _target_split = V / V.sum()

    _M1, _M2, _M3 = int(delays[:N1].sum()), int(delays[N1:N1+N2].sum()), int(delays[N1+N2:].sum())
    _equipartition_split = np.array([_M1, _M2, _M3]) / (_M1 + _M2 + _M3)

    mo.md(
        rf"""
    ## 4 · How closely does the GFDN track the physics?

    Max absolute deviation between the exact GFDN ledger and the true $e^{{tQ}}$ trajectory over the
    whole impulse response: chain = **{_err_chain:.4f}**, complete = **{_err_complete:.4f}**.

    **Numeric long-run split for chain topology** (mean over the last 15% of the simulation):

    | | room 1 | room 2 | room 3 |
    |---|---|---|---|
    | GFDN (exact ledger) | {_gfdn_split_chain[0]:.4f} | {_gfdn_split_chain[1]:.4f} | {_gfdn_split_chain[2]:.4f} |
    | reference $e^{{tQ}}$ | {_ref_split_chain[0]:.4f} | {_ref_split_chain[1]:.4f} | {_ref_split_chain[2]:.4f} |
    | target $V_i/V_{{\rm tot}}$ | {_target_split[0]:.4f} | {_target_split[1]:.4f} | {_target_split[2]:.4f} |
    | $M_i/M_{{\rm tot}}$ (theoretical GFDN asymptote) | {_equipartition_split[0]:.4f} | {_equipartition_split[1]:.4f} | {_equipartition_split[2]:.4f} |

    **Numeric long-run split for complete topology** (mean over the last 15% of the simulation):

    | | room 1 | room 2 | room 3 |
    |---|---|---|---|
    | GFDN (exact ledger) | {_gfdn_split_complete[0]:.4f} | {_gfdn_split_complete[1]:.4f} | {_gfdn_split_complete[2]:.4f} |
    | reference $e^{{tQ}}$ | {_ref_split_complete[0]:.4f} | {_ref_split_complete[1]:.4f} | {_ref_split_complete[2]:.4f} |
    | target $V_i/V_{{\rm tot}}$ | {_target_split[0]:.4f} | {_target_split[1]:.4f} | {_target_split[2]:.4f} |
    | $M_i/M_{{\rm tot}}$ (theoretical GFDN asymptote) | {_equipartition_split[0]:.4f} | {_equipartition_split[1]:.4f} | {_equipartition_split[2]:.4f} |

    **The GFDN's true long-run target is the $M_i/M_{{\rm tot}}$ row, not $V_i/V_{{\rm tot}}$ directly**
    — the mixing angle $\theta$ only sets *how fast* the GFDN approaches it, never *where* it ends up.
    $V_i/V_{{\rm tot}}$ and $M_i/M_{{\rm tot}}$ should be close to each other by design (that's the whole
    point of $M_i\propto V_i$), but they're two different quantities: if the GFDN row disagrees with
    $M_i/M_{{\rm tot}}$, suspect insufficient equilibration time (try increasing `dur` well beyond a
    few relaxation times and see if it keeps drifting toward $M_i/M_{{\rm tot}}$) or something in the
    live delay-line/matrix realization; if it already matches $M_i/M_{{\rm tot}}$ but that itself is
    off from $V_i/V_{{\rm tot}}$, the fix is tightening the delay-length design so $M_1/M_2$ lands
    closer to $V_1/V_2$.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(rf"""
    ### Plot the EDCs for the chain topology
    """)
    return


@app.cell
def _(
    N1,
    N2,
    Y_chain_lossy,
    Y_complete_lossy,
    mo,
    np,
    num_rooms,
    plt,
    pyFDN,
    tsec,
):
    h_mic_chain_lossy = np.stack([Y_chain_lossy[:, :, :N1].sum(-1), 
                                  Y_chain_lossy[:, :, N1:N1+N2].sum(-1), 
                                  Y_chain_lossy[:, :, N1+N2:].sum(-1)], 
                                 axis=-1)
    h_mic_complete_lossy = np.stack([Y_complete_lossy[:, :, :N1].sum(-1), 
                                  Y_complete_lossy[:, :, N1:N1+N2].sum(-1), 
                                  Y_complete_lossy[:, :, N1+N2:].sum(-1)], 
                                  axis=-1)
    def db(_Y, is_energy_signal:bool=True):
        tmp = np.log10(np.abs(_Y) + 1e-10)
        return 10*tmp if is_energy_signal else 20*tmp


    _fig, _axs = plt.subplots(3, 3, figsize=(8, 6.6), sharex=True, sharey=True)
    _tmax =  tsec[-1]
    _nmax = int(_tmax * len(tsec) / tsec[-1])
    for _src in range(num_rooms):
        for _rec in range(num_rooms):
            _ax = _axs[_src, _rec]
            _ax.plot(tsec[:_nmax] * 1000, db(pyFDN.auxiliary.acoustics.edc(h_mic_chain_lossy[_src, :_nmax, _rec])), lw=1.0, color="C0")
            _ax.plot(tsec[:_nmax] * 1000, db(pyFDN.auxiliary.acoustics.edc(h_mic_complete_lossy[_src, :_nmax, _rec])), lw=1.0, color="C2")
            _ax.set_title(f"source room {_src+1} → receiver room {_rec+1}", fontsize=9)
    for _ax in _axs[-1]:
        _ax.set_xlabel("time (ms)")
    for _ax in _axs[:, 0]:
        _ax.set_ylabel("$h(t)$")
    _axs[0, 0].set_ylim(-100, 20)
    _axs[0,0].legend(['1-2, 2-3', '1-2, 1-3, 2-3'])
    _fig.suptitle("EDCs (first %.0f ms)" % (_tmax * 1000), fontsize=10)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


if __name__ == "__main__":
    app.run()
