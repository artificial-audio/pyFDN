import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import block_diag
    from typing import Optional
    from numpy.typing import NDArray

    import pyFDN
    from pyFDN.td import TimeVaryingMatrix
    from pyFDN.auxiliary.physics_based_coupling import (
        create_lossless_coupling_matrix, create_diagonal_absorption_matrix,
        create_state_transition_matrix, trajectory, trajectory_with_delays,
        get_coupling_angles, get_coupling_matrix, get_decay_matrix,
        get_feedback_matrix, run_gfdn, gfdn_energy_ledger)

    return (
        NDArray,
        TimeVaryingMatrix,
        block_diag,
        create_diagonal_absorption_matrix,
        create_lossless_coupling_matrix,
        create_state_transition_matrix,
        get_coupling_angles,
        get_coupling_matrix,
        get_decay_matrix,
        get_feedback_matrix,
        gfdn_energy_ledger,
        mo,
        np,
        plt,
        pyFDN,
        run_gfdn,
        trajectory,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Two coupled rooms: does the GFDN track the physical Markov process?

    Pipeline: physical dimensions $\to$ Cremer–Müller generator $\mathbf Q$ $\to$ pairwise angles
    $\mathbf A=\begin{bmatrix}\cos(\theta) I_{N1} & \sin(\theta) I_{N1} \\ -\sin(\theta)  I_{N2} & \cos(\theta)  I_{N2} \end{bmatrix}\operatorname{blkdiag}(\mathbf Q_1,\mathbf Q_2)$.
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

    a1 = mo.ui.slider(0,
                      1,
                      value=0.1,
                      step=0.05,
                      label="room 1 absorption $a_1$")
    a2 = mo.ui.slider(0,
                      1,
                      value=0.3,
                      step=0.05,
                      label="room 2 absorption $a_2$")

    S12 = mo.ui.slider(0.1,
                       10.0,
                       value=2.0,
                       step=0.1,
                       label="aperture $S_{12}$ (m²)")
    Nper = mo.ui.slider(4,
                        48,
                        value=16,
                        step=2,
                        label="delay lines per room $N_i$ (equal)")

    dur = mo.ui.slider(0.5,
                       10.0,
                       value=2.0,
                       step=0.25,
                       label="impulse‑response length (s)")

    avg_delays = mo.ui.slider(100,
                              10000,
                              value=1000,
                              step=100,
                              label="Avg delay line length in samples")

    mo.md(
        f"### Controls\n{V1}\n\n{V2}\n\n{a1}\n\n{a2}\n\n{S12}\n\n{Nper}\n\n{dur}\n\n{avg_delays}"
    )
    return Nper, S12, V1, V2, a1, a2, avg_delays, dur


@app.cell
def _(
    S12,
    V1,
    V2,
    a1,
    a2,
    create_diagonal_absorption_matrix,
    create_lossless_coupling_matrix,
    create_state_transition_matrix,
    mo,
    np,
):
    # --- Cremer-Muller generators for the two topologies -----------------------
    c = 343.0
    V = np.array([V1.value, V2.value])
    absorp = np.array([a1.value, a2.value])
    num_rooms = 2

    S = np.array([[0, S12.value], [S12.value, 0]])
    beta = create_lossless_coupling_matrix(S, V)
    Q = create_state_transition_matrix(beta)

    # assuming a perfect cube
    _A = 6 * (np.cbrt(V)**2) * absorp
    gamma = create_diagonal_absorption_matrix(_A, V)

    mo.md(rf"""
    ## 0 · Physical generators

    $\mathbf Q$ :
    $$\mathbf Q = \begin{{pmatrix}}{Q[0,0]:.4f}&{Q[0,1]:.4f}\\{Q[1,0]:.4f}&{Q[1,1]:.4f}\end{{pmatrix}}\ \mathrm{{s}}^{{-1}}$$

    Both have zero column sums (pure exchange, no absorption in this notebook) and are reversible w.r.t.
    $\pi_i\propto V_i$ — target long‑run split $V_1{{:}}V_2{{:}}= {V1.value}{{:}}{V2.value}$.

    $\mathbf \Gamma$ :
    $$\mathbf \Gamma = \begin{{pmatrix}}{gamma[0,0]:.4f}&{gamma[0,1]:.4f}\\{gamma[1,0]:.4f}&{gamma[1,1]:.4f}\end{{pmatrix}}\ \mathrm{{s}}^{{-1}}$$
    """)
    return Q, V, beta, gamma, num_rooms


@app.cell
def _(Nper, V, avg_delays, mo, np, pyFDN):
    # --- Geometry: equal N per room, M_i proportional to V_i -------------------
    fs = 48000
    Nroom = Nper.value
    N1 = N2 = Nroom
    Ntot = N1 + N2
    M_avg = avg_delays.value

    gscale = V[0] / V[1]
    d2 = pyFDN.sample_delay_lengths(N2, (M_avg * 0.5, M_avg * 1.5),
                                    coprime=True,
                                    rng=122)
    d1 = pyFDN.sample_delay_lengths(
        N1, (int(M_avg * 0.5 * gscale), int(M_avg * 1.5 * gscale)),
        coprime=True,
        rng=331)
    delays = np.concatenate([d1, d2]).astype(int)
    delays_per_fdn = np.asarray([delays[:N1], delays[N1:]], dtype=np.int32)
    M1, M2 = int(d1.sum()), int(d2.sum())
    M = [M1, M2]
    ratio_mismatch = (V[0] / V[1]) / (M1 / M2)

    k1 = Nroom / np.sum(delays[:N1])
    k2 = Nroom / np.sum(delays[N1:])
    dt_i = 1.0 / (np.array([k1, k2], dtype=np.float32) * fs)

    mo.md(rf"""
    ## 1 · Geometry — $M_i \propto V_i$, equal $N_i$ (so $\mathbf D=\mathbf I$, no weighting needed)

    | | room 1 | room 2 |
    |---|---|---|
    | $N_i$ | {N1} | {N2} |
    | $M_i$ | {M[0]} | {M[1]} |
    | Ratio mismatch| {ratio_mismatch:.4f} |
    | $\Delta t_i=M_i/(N_if_s)$ | {dt_i[0]*1000:.3f} ms | {dt_i[1]*1000:.3f} ms | 
    """)
    return N1, N2, Ntot, delays, delays_per_fdn, dt_i, fs, ratio_mismatch


@app.cell
def _(beta, dt_i, get_coupling_angles, get_coupling_matrix, mo, np, num_rooms):
    theta = get_coupling_angles(beta, dt_i)
    K = get_coupling_matrix(theta)
    R_room = np.array([[np.cos(theta[0, 1]),
                        np.sin(theta[0, 1])],
                       [-np.sin(theta[1, 0]),
                        np.cos(theta[1, 0])]])

    # sanity check: beta_ij*dt_i should equal beta_ji*dt_j (design consistency)
    _check = []
    for _i in range(num_rooms):
        for _j in range(_i + 1, num_rooms):
            _check.append((beta[_i, _j] * dt_i[_i], beta[_j, _i] * dt_i[_j]))

    mo.md(rf"""
    ## 2 · Pairwise angles $\theta_{{ij}}$ (rate‑matched, $\sin^2\theta_{{ij}}=\beta_{{ij}}\Delta t_i$)

    | edge | $\theta_{{ij}}$|
    |---|---|
    | 1-2 | {K[0,1]:.4f} | 
    | 2-1 | {K[1, 0]: .4f} |


    Consistency check ($\beta_{{ij}}\Delta t_i$ vs. $\beta_{{ji}}\Delta t_j$, should match by the $\Delta t_i\propto V_i$
    design): edge 1–2: ${_check[0][0]:.5f}$ vs ${_check[0][1]:.5f}$;
    """)
    return (R_room,)


@app.cell
def _(
    N1,
    N2,
    Ntot,
    R_room,
    TimeVaryingMatrix,
    block_diag,
    delays_per_fdn,
    fs,
    gamma,
    get_decay_matrix,
    get_feedback_matrix,
    mo,
    np,
    pyFDN,
    ratio_mismatch,
):
    # --- Lift into the full GFDN feedback matrix --------------------------------
    np.random.seed(12343)
    # time varying matrix for faster mixing
    modulation_frequency = 1.0  # hz
    modulation_amplitude = 3.0
    spread = 0.3
    tv_matrix = TimeVaryingMatrix(Ntot, modulation_frequency,
                                  modulation_amplitude, fs, spread)

    Qblocks = block_diag(pyFDN.random_orthogonal(N1),
                         pyFDN.random_orthogonal(N2))

    def compensating_matrix():
        rho = np.sqrt(ratio_mismatch)
        return block_diag(rho * np.eye(N1), np.eye(N2))

    Gamma = get_decay_matrix(gamma, delays_per_fdn, fs)
    D = compensating_matrix()

    A = get_feedback_matrix(R_room, Qblocks, N1)
    A_lossy = A @ Gamma
    _ok = pyFDN.is_unilossless(A)

    mo.md(rf"""
    ## 3 · The two GFDN feedback matrices

    Same internal room mixers $\mathbf Q_1,\mathbf Q_2$ for both — only the room‑coupling
    $\mathbf R_{{\rm room}}$ differs. `is_unilossless`: = **{_ok}**
    (both should be `True` — pure exchange, no absorption).
    """)
    return A, A_lossy, tv_matrix


@app.cell
def _(
    A,
    A_lossy,
    N1,
    N2,
    Ntot,
    delays,
    dur,
    fs,
    gfdn_energy_ledger,
    np,
    num_rooms,
    run_gfdn,
    tv_matrix,
):
    # --- Impulse responses + exact per-room energy ledger, all rooms excited equally -----
    n_samp = int(dur.value * fs)
    B = np.zeros((Ntot, num_rooms))
    B[:N1, 0] = 1.0 / np.sqrt(N1)
    B[N1:, 1] = 1.0 / np.sqrt(N2)

    # output taken from all rooms
    C_lines = np.eye(Ntot)
    Y = run_gfdn(A,
                 B,
                 C_lines,
                 delays,
                 n_samp,
                 _src=[0, 1],
                 tv_matrix=tv_matrix)
    Y_lossy = run_gfdn(A_lossy,
                       B,
                       C_lines,
                       delays,
                       n_samp,
                       _src=[0, 1],
                       tv_matrix=tv_matrix)

    E_ex, n_ex = gfdn_energy_ledger(Y, num_rooms, N1, delays)
    E_ex_lossy, _ = gfdn_energy_ledger(Y_lossy, num_rooms, N1, delays)
    return E_ex, E_ex_lossy, Y, Y_lossy, n_ex, n_samp


@app.cell
def _(Q, fs, gamma, n_samp, np, num_rooms, trajectory):
    # --- True physical trajectory e(t) = expm(Q t) e(0), source = room 1 -------
    tsec = np.arange(n_samp) / fs
    _, Eref = trajectory(Q, np.array([1.0, 0]), np.eye(num_rooms), fs, n_samp)
    _, Eref_lossy = trajectory(Q,
                               np.array([1.0, 0]),
                               np.eye(num_rooms),
                               fs,
                               n_samp,
                               gamma=gamma)
    return Eref, Eref_lossy, tsec


@app.cell
def _(
    E_ex,
    E_ex_lossy,
    Eref,
    Eref_lossy,
    NDArray,
    V,
    mo,
    n_ex,
    np,
    num_rooms,
    plt,
    tsec,
):
    # --- The comparison plot -----------------------------------------------------
    _fig, _axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    _labels = ["room 1", "room 2"]
    _colors = ["C0", "C1"]
    _targets = V / V.sum()

    def db(_Y: NDArray, is_energy_signal: bool = True):
        _tmp = np.log10(np.abs(_Y) + 1e-10)
        return 10 * _tmp if is_energy_signal else 20 * _tmp

    for _ax, _Eex, _Eref, _title in zip(_axs, [E_ex, db(E_ex_lossy)],
                                        [Eref, db(Eref_lossy)],
                                        ["Lossless", "Lossy"]):
        for _r in range(num_rooms):
            _ax.plot(tsec[:n_ex],
                     _Eex[0, :, _r],
                     color=_colors[_r],
                     lw=1.1,
                     label=f"GFDN {_labels[_r]}")
            _ax.plot(tsec,
                     _Eref[:, _r],
                     "--",
                     color=_colors[_r],
                     lw=1.0,
                     alpha=0.7)
            _ax.axhline(_targets[_r], color=_colors[_r], lw=0.5, ls=":")
        _ax.set_title(_title, fontsize=10)
        _ax.set_xlabel("time (s)")
    _axs[0].set_ylim(-0.02, 1.0)
    _axs[1].set_ylim(-40, 10.0)
    _axs[0].set_ylabel("fraction of total energy")
    _axs[1].legend(fontsize=7.5, loc="upper right")
    _axs[1].set_ylabel('fraction of total energy (dB)')
    _fig.suptitle(
        "Solid = exact GFDN ledger, dashed = physical $e^{tQ}$, dotted = target $V_i/V_{\\rm tot}$",
        fontsize=9.5)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return (db,)


@app.cell
def _(E_ex, E_ex_lossy, Eref, Eref_lossy, N1, V, delays, mo, n_ex, np):
    _tail = slice(int(0.85 * n_ex), n_ex)

    _err = np.max(np.abs(E_ex[0] - Eref[:n_ex]))
    _err_lossy = np.max(np.abs(E_ex_lossy[0] - Eref_lossy[:n_ex]))

    _gfdn_tail = E_ex[0, _tail].mean(0)
    _gfdn_split = _gfdn_tail / _gfdn_tail.sum()
    _ref_tail = Eref[_tail].mean(0)
    _ref_split = _ref_tail / _ref_tail.sum()
    _target_split = V / V.sum()

    _M1, _M2 = int(delays[:N1].sum()), int(delays[N1:].sum())
    _equipartition_split = np.array([_M1, _M2]) / (_M1 + _M2)

    mo.md(rf"""
    ## 4 · How closely does the GFDN track the physics?

    Max absolute deviation between the exact GFDN ledger and the true $e^{{tQ}}$ trajectory over the
    whole impulse response: lossless = **{_err:.4f}**, lossy = **{_err_lossy:.4f}**.

    **Numeric long-run split** (mean over the last 15% of the simulation):

    | | room 1 | room 2 |
    |---|---|---|
    | GFDN (exact ledger) | {_gfdn_split[0]:.4f} | {_gfdn_split[1]:.4f} |
    | reference $e^{{tQ}}$ | {_ref_split[0]:.4f} | {_ref_split[1]:.4f} |
    | target $V_i/V_{{\rm tot}}$ | {_target_split[0]:.4f} | {_target_split[1]:.4f} |
    | $M_i/M_{{\rm tot}}$ (theoretical GFDN asymptote) | {_equipartition_split[0]:.4f} | {_equipartition_split[1]:.4f} |

    **The GFDN's true long-run target is the $M_i/M_{{\rm tot}}$ row, not $V_i/V_{{\rm tot}}$ directly**
    — the mixing angle $\theta$ only sets *how fast* the GFDN approaches it, never *where* it ends up.
    $V_i/V_{{\rm tot}}$ and $M_i/M_{{\rm tot}}$ should be close to each other by design (that's the whole
    point of $M_i\propto V_i$), but they're two different quantities: if the GFDN row disagrees with
    $M_i/M_{{\rm tot}}$, suspect insufficient equilibration time (try increasing `dur` well beyond a
    few relaxation times and see if it keeps drifting toward $M_i/M_{{\rm tot}}$) or something in the
    live delay-line/matrix realization; if it already matches $M_i/M_{{\rm tot}}$ but that itself is
    off from $V_i/V_{{\rm tot}}$, the fix is tightening the delay-length design so $M_1/M_2$ lands
    closer to $V_1/V_2$.
    """)
    return


@app.cell
def _(mo):
    mo.md(rf"""
    ### Plot the impulse responses and the EDCs
    """)
    return


@app.cell
def _(N1, Y, Y_lossy, mo, np, num_rooms, plt, tsec):
    # physical receivers: one microphone per room = coherent sum of its lines
    h_mic = np.stack([Y[:, :, :N1].sum(-1), Y[:, :, N1:].sum(-1)], axis=-1)
    h_mic_lossy = np.stack(
        [Y_lossy[:, :, :N1].sum(-1), Y_lossy[:, :, N1:].sum(-1)], axis=-1)

    # --- The four impulse responses (microphone = sum of the room's lines) -----
    _fig, _axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    _tmax = min(0.5, tsec[-1])
    _nmax = int(_tmax * len(tsec) / tsec[-1])
    for _src in range(num_rooms):
        for _rec in range(num_rooms):
            _ax = _axs[_src, _rec]
            _ax.plot(tsec[:_nmax] * 1000,
                     h_mic[_src, :_nmax, _rec],
                     lw=0.4,
                     color="C0")
            _ax.plot(tsec[:_nmax] * 1000,
                     h_mic_lossy[_src, :_nmax, _rec],
                     lw=0.4,
                     color="C2")
            _ax.set_title(f"source room {_src+1} → receiver room {_rec+1}",
                          fontsize=9)
    for _ax in _axs[-1]:
        _ax.set_xlabel("time (ms)")
    for _ax in _axs[:, 0]:
        _ax.set_ylabel("$h(t)$")
    _ymax = np.abs(h_mic[:, :_nmax]).max()
    _axs[0, 0].set_ylim(-_ymax, _ymax)
    _axs[0, 0].legend(['Lossless', 'Lossy'])
    _fig.suptitle("Four lossless + lossy impulse responses (first %.0f ms)" %
                  (_tmax * 1000),
                  fontsize=10)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return h_mic, h_mic_lossy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lossless IRs
    """)
    return


@app.cell
def _(fs, h_mic, mo, num_rooms):
    mo.vstack([[
        mo.vstack([
            mo.md(f"src={_src}, rec={_rec}"),
            mo.audio(src=h_mic[_src, :, _rec].T, rate=fs)
        ]) for _rec in range(num_rooms) for _src in range(num_rooms)
    ]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lossy IRs
    """)
    return


@app.cell
def _(fs, h_mic_lossy, mo, num_rooms):
    mo.vstack([[
        mo.vstack([
            mo.md(f"src={_src}, rec={_rec}"),
            mo.audio(src=h_mic_lossy[_src, :, _rec].T, rate=fs)
        ]) for _src in range(num_rooms) for _rec in range(num_rooms)
    ]])
    return


@app.cell
def _(db, h_mic, h_mic_lossy, mo, num_rooms, plt, pyFDN, tsec):
    _fig, _axs = plt.subplots(2, 2, figsize=(8, 4.6), sharex=True, sharey=True)
    _tmax = tsec[-1]
    _nmax = int(_tmax * len(tsec) / tsec[-1])
    for _src in range(num_rooms):
        for _rec in range(num_rooms):
            _ax = _axs[_src, _rec]
            _ax.plot(tsec[:_nmax] * 1000,
                     db(
                         pyFDN.auxiliary.acoustics.edc(h_mic[_src, :_nmax,
                                                             _rec])),
                     lw=1.0,
                     color="C0")
            _ax.plot(tsec[:_nmax] * 1000,
                     db(
                         pyFDN.auxiliary.acoustics.edc(
                             h_mic_lossy[_src, :_nmax, _rec])),
                     lw=1.0,
                     color="C2")
            _ax.set_title(f"source room {_src+1} → receiver room {_rec+1}",
                          fontsize=9)
    for _ax in _axs[-1]:
        _ax.set_xlabel("time (ms)")
    for _ax in _axs[:, 0]:
        _ax.set_ylabel("$h(t)$")
    _axs[0, 0].set_ylim(-100, 20)
    _axs[0, 0].legend(['Lossless', 'Lossy'])
    _fig.suptitle("Four lossless + lossy EDCs (first %.0f ms)" %
                  (_tmax * 1000),
                  fontsize=10)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


if __name__ == "__main__":
    app.run()
