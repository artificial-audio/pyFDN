import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from numpy.typing import NDArray, ArrayLike
    from scipy.linalg import block_diag, expm


    import pyFDN
    from pyFDN.build import FDNBuild
    from pyFDN.auxiliary.physics_based_coupling import (
        create_lossless_coupling_matrix, create_diagonal_absorption_matrix,
        create_state_transition_matrix, trajectory, trajectory_with_delays,
        match_delay_totals_to_volumes, get_coupling_angles, get_coupling_matrix, get_decay_matrix,
        get_feedback_matrix, get_room_time_varying_matrix, gfdn_energy_ledger)


    return (
        ArrayLike,
        FDNBuild,
        NDArray,
        block_diag,
        create_diagonal_absorption_matrix,
        create_lossless_coupling_matrix,
        create_state_transition_matrix,
        expm,
        match_delay_totals_to_volumes,
        get_coupling_angles,
        get_coupling_matrix,
        get_decay_matrix,
        get_feedback_matrix,
        get_room_time_varying_matrix,
        gfdn_energy_ledger,
        mo,
        np,
        plt,
        pyFDN,
        trajectory,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Three coupled rooms: does the GFDN track the physical Markov process?

    Two topologies, same three physical rooms:

    - **Chain**: room 1 ↔ room 2 ↔ room 3, no direct 1–3 aperture ($S_{13}=0$).
    - **Complete**: all three rooms mutually coupled ($S_{13}>0$ too).

    This notebook explores impulse-driven onset. The paper's noise-start SA verification and reproducible exports are in `export_paper_figures.py`.

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
    a3 = mo.ui.slider(0,
                      1,
                      value=0.2,
                      step=0.05,
                      label="room 3 absorption $a_3$")

    Nper = mo.ui.slider(4,
                        48,
                        value=16,
                        step=2,
                        label="delay lines per room $N_i$ (equal)")

    avg_delays = mo.ui.slider(100,
                              10000,
                              value=2000,
                              step=100,
                              label="Avg delay line length in samples")

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
def _(
    S12,
    S13,
    S23,
    V1,
    V2,
    V3,
    a1,
    a2,
    a3,
    create_diagonal_absorption_matrix,
    create_lossless_coupling_matrix,
    create_state_transition_matrix,
    mo,
    np,
):
    # --- Cremer-Muller generators for the two topologies -----------------------
    c = 343.0
    V = np.array([V1.value, V2.value, V3.value])
    absorp = np.array([a1.value, a2.value, a3.value])
    num_rooms = 3

    S_chain = np.array([[0, S12.value, 0], [S12.value, 0, S23.value],
                        [0, S23.value, 0]])
    S_complete = np.array([[0, S12.value,
                            S13.value], [S12.value, 0, S23.value],
                           [S13.value, S23.value, 0]])

    beta_chain, beta_complete = create_lossless_coupling_matrix(
        S_chain, V), create_lossless_coupling_matrix(S_complete, V)
    Q_chain, Q_complete = create_state_transition_matrix(
        beta_chain), create_state_transition_matrix(beta_complete)

    # assuming a perfect cube
    _A = 6 * (np.cbrt(V)**2) * absorp
    # get the diagonal absorption matrix
    gamma = create_diagonal_absorption_matrix(_A, V)

    mo.md(rf"""
    ## 0 · Physical generators

    $\mathbf Q_{{\rm chain}}$ (no 1–3 aperture):
    $$\mathbf Q_{{\rm chain}} = \begin{{pmatrix}}{Q_chain[0,0]:.4f}&{Q_chain[0,1]:.4f}&{Q_chain[0,2]:.4f}\\{Q_chain[1,0]:.4f}&{Q_chain[1,1]:.4f}&{Q_chain[1,2]:.4f}\\{Q_chain[2,0]:.4f}&{Q_chain[2,1]:.4f}&{Q_chain[2,2]:.4f}\end{{pmatrix}}\ \mathrm{{s}}^{{-1}}$$

    $\mathbf Q_{{\rm complete}}$ (all three coupled):
    $$\mathbf Q_{{\rm complete}} = \begin{{pmatrix}}{Q_complete[0,0]:.4f}&{Q_complete[0,1]:.4f}&{Q_complete[0,2]:.4f}\\{Q_complete[1,0]:.4f}&{Q_complete[1,1]:.4f}&{Q_complete[1,2]:.4f}\\{Q_complete[2,0]:.4f}&{Q_complete[2,1]:.4f}&{Q_complete[2,2]:.4f}\end{{pmatrix}}\ \mathrm{{s}}^{{-1}}$$

    Both have zero column sums (pure exchange, no absorption in this notebook) and are reversible w.r.t.
    $\pi_i\propto V_i$ — target long‑run split $V_1{{:}}V_2{{:}}V_3 = {V1.value}{{:}}{V2.value}{{:}}{V3.value}$.
    """)
    return (
        Q_chain,
        Q_complete,
        S_chain,
        S_complete,
        V,
        beta_chain,
        beta_complete,
        c,
        gamma,
        num_rooms,
    )


@app.cell
def _(ArrayLike, NDArray, Nper, V, avg_delays, match_delay_totals_to_volumes, mo, np, num_rooms, pyFDN):
    # --- Geometry: equal N per room, M_i proportional to V_i -------------------
    fs = 48000
    Nroom = Nper.value
    N1 = N2 = N3 = Nroom
    Ntot = N1 + N2 + N3

    def sample_delay_line_lengths(avg_tau: float, num_del_per_group:int) -> NDArray:
        """Sample co-prime delay line lengths according to the volume ratio"""
        gscale = V / V[0]
        delay_min = 0.5 * avg_tau
        delay_max = 1.5 * avg_tau
        delays = []

        for _i in range(num_rooms):
            delays.append(
                pyFDN.sample_delay_lengths(
                    num_del_per_group,
                    (int(delay_min * gscale[_i]), int(delay_max * gscale[_i])),
                    coprime=True,
                    rng=331 + 10 * _i))

        delays = np.concatenate(delays).astype(int)
        return delays


    co_prime_delays = sample_delay_line_lengths(avg_delays.value, Nroom)
    delays = match_delay_totals_to_volumes(co_prime_delays, Nroom, V)
    M1, M2, M3 = int(delays[:N1].sum()), int(delays[N1:N1 + N2].sum()), int(
        delays[N1 + N2:].sum())
    delays_per_fdn = np.asarray(
        [delays[:N1], delays[N1:N1 + N2], delays[N1 + N2:]], dtype=np.int32)
    _delay_totals = np.asarray([M1, M2, M3])
    dt_i = _delay_totals / (Nroom * fs)

    mo.md(rf"""
    ## 1 · Geometry — $M_i \propto V_i$, equal $N_i$ (so $\mathbf D=\mathbf I$, no weighting needed)

    | | room 1 | room 2 | room 3 |
    |---|---|---|---|
    | $V_i$ | {V[0]} | {V[1]} | {V[2]}| 
    | $N_i$ | {N1} | {N2} | {N3} |
    | actual $\sum d_j$ | {int(delays[:N1].sum())} | {int(delays[N1:N1+N2].sum())} | {int(delays[N1+N2:].sum())} |
    | $\Delta t_i=M_i/(N_if_s)$ | {dt_i[0]*1000:.3f} ms | {dt_i[1]*1000:.3f} ms | {dt_i[2]*1000:.3f} ms |
    | $V_i / \sum_i V_i$ | {V[0] / np.sum(V):.3f}  | {V[1] / np.sum(V):.3f} | {V[2] / np.sum(V):.3f} |
    | $M_i / \sum_i M_i$ | {M1 / np.sum(_delay_totals):.3f}  | {M2 / np.sum(_delay_totals):.3f} | {M3 / np.sum(_delay_totals):.3f} |
    """)
    return N1, N2, N3, Nroom, Ntot, delays, delays_per_fdn, dt_i, fs


@app.cell
def _(mo):
    coupling_angle_rule = mo.ui.dropdown(
        options=["arcsin(sqrt(p))", "sqrt(p)"], value="arcsin(sqrt(p))",
        label="Coupling angle rule, p = beta_ij * dt_i")
    mo.vstack([coupling_angle_rule])
    return (coupling_angle_rule,)


@app.cell
def _(
    beta_chain,
    beta_complete,
    coupling_angle_rule,
    dt_i,
    expm,
    get_coupling_angles,
    get_coupling_matrix,
    mo,
    np,
    num_rooms,
):
    if coupling_angle_rule.value == "sqrt(p)":
        # Symmetrise the two directional probabilities, as in the helper.
        # With exact volume/delay ratios these directional values coincide.
        _p_chain = beta_chain * dt_i[:, None]
        _p_complete = beta_complete * dt_i[:, None]
        theta_chain = np.sqrt(0.5 * (_p_chain + _p_chain.T))
        theta_complete = np.sqrt(0.5 * (_p_complete + _p_complete.T))
        _rule = r"\theta_{ij}^2=\beta_{ij}\Delta t_i"
    else:
        theta_chain, theta_complete = get_coupling_angles(
            beta_chain, dt_i), get_coupling_angles(beta_complete, dt_i)
        _rule = r"\sin^2\theta_{ij}=\beta_{ij}\Delta t_i"
    K_chain, K_complete = get_coupling_matrix(
        theta_chain), get_coupling_matrix(theta_complete)
    R_room_chain, R_room_complete = expm(K_chain), expm(K_complete)

    # sanity check: beta_ij*dt_i should equal beta_ji*dt_j (design consistency)
    _check = []
    for _i in range(num_rooms):
        for _j in range(_i + 1, num_rooms):
            _check.append(
                (beta_complete[_i, _j] * dt_i[_i], beta_complete[_j, _i] * dt_i[_j]))

    mo.md(rf"""
    ## 2 · Pairwise angles (${_rule}$)

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Coupling transfer comparison
    """)
    return


@app.cell
def _(
    Nroom,
    R_room_chain,
    R_room_complete,
    S_chain,
    S_complete,
    V,
    c,
    delays_per_fdn,
    fs,
    mo,
    np,
    plt,
):
    _tau = np.sum(delays_per_fdn, axis=1)
    _v = V / _tau
    coupling_rate_scale = np.diag(Nroom * _v * fs)
    _reference_scale = max(float(np.max(c * _S / 4)) for _S in (S_chain, S_complete))
    _reference_scale = max(_reference_scale, np.finfo(float).eps)
    coupling_transfer_matrices = {}
    for _name, _phi, _apertures in (
        ("Chained", R_room_chain, S_chain),
        ("Complete", R_room_complete, S_complete),
    ):
        _gfdn = coupling_rate_scale @ np.square(_phi)
        _physical = c * _apertures / 4
        np.fill_diagonal(_gfdn, 0)
        np.fill_diagonal(_physical, 0)
        _normalized_error = np.abs(_gfdn - _physical) / _reference_scale
        coupling_transfer_matrices[_name] = (_gfdn, _physical, _normalized_error)

    _transfer_max = max(float(_matrix.max()) for _matrices in coupling_transfer_matrices.values()
                        for _matrix in _matrices[:2])
    _plots = []
    for _name, _matrices in coupling_transfer_matrices.items():
        _fig, _axes = plt.subplots(1, 3, figsize=(11.5, 3.8), layout="constrained")
        for _column, _matrix in enumerate(_matrices):
            _ax = _axes[_column]
            _maximum = 1.0 if _column == 2 else _transfer_max
            _image = _ax.imshow(_matrix, vmin=0, vmax=max(_maximum, 1e-12), cmap="coolwarm")
            if _column == 2:
                _error_image = _image
            else:
                _transfer_image = _image
            _ax.set_xticks(range(len(V)), [f"R{_i+1}" for _i in range(len(V))])
            _ax.set_yticks(range(len(V)), [f"R{_i+1}" for _i in range(len(V))])
            _ax.set_xlabel("Sending room $j$")
            _ax.set_ylabel("Receiving room $i$")
            _ax.set_title((r"$H\Phi^{\odot 2}$", r"$cS/4$",
                           "Normalized absolute error")[_column])
        _fig.colorbar(_transfer_image, ax=_axes[:2], label=r"Transfer ($\mathrm{m^3/s}$)")
        _fig.colorbar(
            _error_image, ax=_axes[2], ticks=np.linspace(0, 1, 6), format="%.1f",
            label=r"$|H\Phi^{\odot 2}-cS/4|\,/\,T_{\max}$")
        _aperture_13 = S_chain[0, 2] if _name == "Chained" else S_complete[0, 2]
        _fig.suptitle(
            rf"{_name} topology: $S_{{13}}={_aperture_13:g}\,\mathrm{{m^2}}$ — zero diagonals")
        _plots.append(mo.mpl.interactive(_fig))
    mo.vstack(_plots)
    return


@app.cell
def _(
    N1,
    N2,
    N3,
    Nroom,
    R_room_chain,
    R_room_complete,
    block_diag,
    delays_per_fdn,
    fs,
    gamma,
    get_decay_matrix,
    get_feedback_matrix,
    get_room_time_varying_matrix,
    mo,
    np,
    pyFDN,
):
    np.random.seed(12314)

    # time varying matrix for faster mixing
    modulation_frequency = 1.0  # hz
    modulation_amplitude = 3.0
    spread = 0.3
    # Use identical random rotation parameters for both fixed coupling topologies.
    _rotation_rng = np.random.get_state()
    tv_chain = get_room_time_varying_matrix(
        R_room_chain, Nroom, modulation_frequency, modulation_amplitude, fs, spread)
    _after_rotation_rng = np.random.get_state()
    np.random.set_state(_rotation_rng)
    tv_complete = get_room_time_varying_matrix(
        R_room_complete, Nroom, modulation_frequency, modulation_amplitude, fs, spread)
    np.random.set_state(_after_rotation_rng)

    M = block_diag(pyFDN.random_orthogonal(N1),
                   pyFDN.random_orthogonal(N2), pyFDN.random_orthogonal(N3))
    G = get_decay_matrix(gamma, delays_per_fdn, fs)
    F_chain = np.kron(R_room_chain, np.eye(Nroom))
    F_complete = np.kron(R_room_complete, np.eye(Nroom))
    A_chain = get_feedback_matrix(R_room_chain, M, Nroom)  # F_chain @ M
    A_complete = get_feedback_matrix(R_room_complete, M, Nroom)
    A_chain_lossy = A_chain @ G
    A_complete_lossy = A_complete @ G

    _ok_chain = pyFDN.is_unilossless(A_chain)
    _ok_complete = pyFDN.is_unilossless(A_complete)

    mo.md(rf"""
    ## 3 · Fixed coupling and time-varying internal room mixers

    Same internal room mixers $\mathbf Q_1,\mathbf Q_2,\mathbf Q_3$ for both — only the room‑coupling
    $\mathbf R_{{\rm room}}$ differs. `is_unilossless`: chain = **{_ok_chain}**, complete = **{_ok_complete}**
    (both should be `True` — pure exchange, no absorption).

    $M=\operatorname{{blkdiag}}(M_1,M_2,M_3)$ contains the orthogonal room mixers.
    $G_{{jj}}=\exp(-\gamma_i m_j/(2f_s))$ contains per-line amplitude absorption.
    For either topology, $F=\Phi\otimes I_K$, `A = F @ M`, and `A_lossy = F @ M @ G`.
    The hook $H(n)=F R(n)F^\top$ makes the effective feedback $F R(n)M G$:
    only $M_i(n)=R_i(n)M_i$ varies, while the room coupling stays fixed.
    """)
    return (
        A_chain,
        A_chain_lossy,
        A_complete,
        A_complete_lossy,
        tv_chain,
        tv_complete,
    )


@app.cell
def _(
    A_chain,
    A_chain_lossy,
    A_complete,
    A_complete_lossy,
    FDNBuild,
    N1,
    N2,
    N3,
    Nroom,
    Ntot,
    delays,
    dur,
    fs,
    gfdn_energy_ledger,
    np,
    num_rooms,
    pyFDN,
    tv_chain,
    tv_complete,
):
    # --- Impulse responses + exact per-room energy ledger, all rooms excited equally -----
    n_samp = int(dur.value * fs)
    B = np.zeros((Ntot, num_rooms))
    B[:N1, 0] = 1.0 / np.sqrt(N1)
    B[N1:N1 + N2, 1] = 1.0 / np.sqrt(N2)
    B[N1 + N2:, 2] = 1.0 / np.sqrt(N3)
    # output taken from all rooms
    C_lines = np.eye(Ntot)
    chain_build = FDNBuild(
        A=A_chain, B=B, C=C_lines, D=np.zeros((Ntot, num_rooms)),
        delays=delays, fs=fs, post_matrix=tv_chain)
    chain_lossy_build = FDNBuild(
        A=A_chain_lossy, B=B, C=C_lines, D=np.zeros((Ntot, num_rooms)),
        delays=delays, fs=fs, post_matrix=tv_chain)
    complete_build = FDNBuild(
        A=A_complete, B=B, C=C_lines, D=np.zeros((Ntot, num_rooms)),
        delays=delays, fs=fs, post_matrix=tv_complete)
    complete_lossy_build = FDNBuild(
        A=A_complete_lossy, B=B, C=C_lines, D=np.zeros((Ntot, num_rooms)),
        delays=delays, fs=fs, post_matrix=tv_complete)

    def _render_sources(build):
        responses = np.zeros((num_rooms, n_samp, Ntot))
        for source in range(num_rooms):
            signal = np.zeros((n_samp, num_rooms))
            signal[0, source] = 1
            build.post_matrix.reset()
            responses[source] = pyFDN.process_fdn(signal, build)
        return responses

    Y_chain = _render_sources(chain_build)
    Y_chain_lossy = _render_sources(chain_lossy_build)
    Y_complete = _render_sources(complete_build)
    Y_complete_lossy = _render_sources(complete_lossy_build)

    E_ex_chain, n_ex = gfdn_energy_ledger(Y_chain, num_rooms, Nroom, delays)
    E_ex_chain_lossy, _ = gfdn_energy_ledger(Y_chain_lossy, num_rooms, Nroom,
                                             delays)
    E_ex_complete, _ = gfdn_energy_ledger(Y_complete, num_rooms, Nroom, delays)
    E_ex_complete_lossy, _ = gfdn_energy_ledger(Y_complete_lossy, num_rooms,
                                                Nroom, delays)
    return (
        E_ex_chain,
        E_ex_chain_lossy,
        E_ex_complete,
        E_ex_complete_lossy,
        Y_chain_lossy,
        Y_complete_lossy,
        n_ex,
        n_samp,
    )


@app.cell
def _(Q_chain, Q_complete, fs, gamma, n_samp, np, num_rooms, trajectory):
    src_weight = np.zeros(num_rooms)
    src_room = 0
    src_weight[src_room] = 1.0
    rec_weight = np.eye(num_rooms)

    _, Eref_chain = trajectory(Q_chain, src_weight, rec_weight, fs, n_samp)
    _, Eref_chain_lossy = trajectory(Q_chain, src_weight, rec_weight, fs,
                                     n_samp, gamma)

    _, Eref_complete = trajectory(Q_complete, src_weight, rec_weight, fs,
                                  n_samp)
    _, Eref_complete_lossy = trajectory(Q_complete, src_weight, rec_weight, fs,
                                        n_samp, gamma)
    return (
        Eref_chain,
        Eref_chain_lossy,
        Eref_complete,
        Eref_complete_lossy,
        src_room,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot and compare trajectories for the chained case
    """)
    return


@app.cell
def _(
    E_ex_chain,
    E_ex_chain_lossy,
    Eref_chain,
    Eref_chain_lossy,
    V,
    fs,
    mo,
    n_ex,
    n_samp,
    np,
    num_rooms,
    plt,
    src_room,
):
    # --- The comparison plot -----------------------------------------------------
    _fig, _axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    _labels = ["room 1", "room 2", "room 3"]
    _colors = ["C0", "C1", "C2"]
    _targets = V / V.sum()
    tsec = np.arange(n_samp) / fs

    def db(_Y, is_energy_signal: bool = True):
        _tmp = np.log10(np.abs(_Y) + 1e-10)
        return 10 * _tmp if is_energy_signal else 20 * _tmp

    for _ax, _Eex, _Eref, _title in zip(
            _axs, [E_ex_chain, db(E_ex_chain_lossy)], [Eref_chain, db(Eref_chain_lossy)],
        ["lossless chained (1–2, 2–3)", "lossy chained"]):
        for _r in range(num_rooms):
            _ax.plot(tsec[:n_ex],
                     _Eex[src_room, :, _r],
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
    _axs[1].set_ylim(-80, 5)
    _axs[0].set_ylabel("fraction of total energy")
    _axs[1].set_ylabel('fraction of total energy (dB)')
    _axs[1].legend(fontsize=9, loc="upper right")
    # _fig.suptitle(
    #     "Solid = exact lossless GFDN ledger, dashed = physical $e^{tQ}$, dotted = target $V_i/V_{\\rm tot}$",
    #     fontsize=9.5)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return db, tsec


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot and compare trajectories for the complete case
    """)
    return


@app.cell
def _(
    E_ex_complete,
    E_ex_complete_lossy,
    Eref_complete,
    Eref_complete_lossy,
    V,
    db,
    mo,
    n_ex,
    num_rooms,
    plt,
    src_room,
    tsec,
):
    _fig, _axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    _labels = ["room 1", "room 2", "room 3"]
    _colors = ["C0", "C1", "C2"]
    _targets = V / V.sum()

    for _ax, _Eex, _Eref, _title in zip(
            _axs, [E_ex_complete, db(E_ex_complete_lossy)],
        [Eref_complete, db(Eref_complete_lossy)],
        ["lossless complete (all coupled)", "lossy complete"]):
        for _r in range(num_rooms):
            _ax.plot(tsec[:n_ex],
                         _Eex[src_room, :, _r],
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
    _axs[1].set_ylim(-80, 5)
    _axs[0].set_ylabel("fraction of total energy")
    _axs[1].set_ylabel('fraction of total energy (dB)')
    _axs[1].legend(fontsize=9, loc="upper right")
    # _fig.suptitle(
    #     "Solid = exact lossy GFDN ledger, dashed = physical $e^{t(Q-D)}$, dotted = target $V_i/V_{\\rm tot}$",
    #     fontsize=9.5)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Get time constants for the lossless trajectories and T60s for the lossy trajectories
    """)
    return


@app.cell
def _(E_ex_chain, E_ex_complete, Eref_chain, Eref_complete, Q_chain, Q_complete, fs, mo, np):
    rows = []
    for name, ledger, reference, generator in (
        ("Chain", E_ex_chain[0], Eref_chain, Q_chain),
        ("Complete", E_ex_complete[0], Eref_complete, Q_complete),
    ):
        stop = min(len(ledger), len(reference))
        rates = np.linalg.eigvals(generator).real
        relaxation = sorted(-1 / rates[rates < -1e-10])
        errors = np.abs(ledger[:stop] - reference[:stop])
        early_stop = min(int(0.2 * fs), stop)
        for room in range(reference.shape[1]):
            rows.append({"Topology": name, "Room": room + 1,
                         "SA modal relaxation times (s)": ", ".join(f"{t:.4f}" for t in relaxation),
                         "Full MAE (pp)": 100 * errors[:, room].mean(),
                         "First 200 ms MAE (pp)": 100 * errors[:early_stop, room].mean()})
    mo.vstack([mo.md("Lossless relaxation rates are shared modes of the SA generator; "
                     "fitting an unshifted exponential to a nonzero equilibrium does not estimate them."),
               mo.ui.table(rows, selection=None)])
    return


@app.cell
def _(
    E_ex_chain_lossy,
    E_ex_complete_lossy,
    Eref_chain_lossy,
    Eref_complete_lossy,
    fs,
    np,
    num_rooms,
    src_room,
):
    def calculate_t60_from_energy(E_ledger, fs: float) -> float:
        """
        Calculates T60 reverberation time from an energy trajectory array
        using a peak-relative [-5 dB, -35 dB] slope fit on stored energy.

        Parameters:
        -----------
        energy_ledger : np.ndarray
            Array containing the sequential energy values (e.g., squared pressure samples, h^2).
        fs : float
            Sampling frequency of the trajectory in Hz.

        Returns:
        --------
        float : Estimated T60 decay time in seconds.
        """
        # 1. Convert the raw linear energy trajectory to a relative Decibel (dB) scale
        # Normalize against the peak energy so the max point acts as 0 dB
        peak_energy= np.max(E_ledger)
        peak_idx = np.argmax(E_ledger)
        energy_db = 10 * np.log10((E_ledger[peak_idx:] + np.finfo(float).tiny) / peak_energy)

        # Generate structural time track matching the array size (Time = Index / fs)
        time_axis = np.arange(len(energy_db)) / fs

        # 2. Extract boundaries for the ISO standard T30 window (-5 dB down to -35 dB)
        # Using np.argwhere to find where the signal crosses the boundary milestones
        idx_start_arr = np.argwhere(energy_db <= -5.0)
        idx_end_arr = np.argwhere(energy_db <= -35.0)

        if len(idx_start_arr) == 0 or len(idx_end_arr) == 0:
            raise ValueError(
                "The energy trajectory dynamic range is too narrow. "
                "Ensure the signal decays by at least 35 dB to isolate the evaluation window."
            )

        start_idx = idx_start_arr[0][0]
        end_idx = idx_end_arr[0][0]

        # 3. Slice out the clean decay segment
        decay_time_window = time_axis[start_idx:end_idx]
        decay_db_window = energy_db[start_idx:end_idx]

        # 4. Use linear regression (y = mx + c) to find the slope (m) of the isolated window
        slope, intercept = np.polyfit(decay_time_window, decay_db_window, 1)

        # 5. Extrapolate out to a full 60 dB drop
        # T60 = Total targeted drop (-60 dB) divided by the calculated decay slope (dB/second)
        t60 = float(-60.0 / slope)

        return t60

    for  _Eex, _Eref,in zip([E_ex_chain_lossy, E_ex_complete_lossy], [Eref_chain_lossy, Eref_complete_lossy]):
        _stop = min(_Eex.shape[1], len(_Eref))
        for _r in range(num_rooms):
            t60_ref = calculate_t60_from_energy(_Eref[:_stop, _r], fs)
            print(f'Cur reference T60 for room {_r+1} is {t60_ref:.3f}s')
            t60_gfdn = calculate_t60_from_energy(_Eex[src_room, :_stop, _r], fs)
            print(f'Cur GFDN T60 for room {_r+1} is {t60_gfdn:.3f}s')
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

    _err_chain = E_ex_chain[0] - Eref_chain[:n_ex]
    _err_complete = E_ex_complete[0] - Eref_complete[:n_ex]

    _gfdn_tail_chain = E_ex_chain[0, _tail].mean(0)
    _gfdn_split_chain = _gfdn_tail_chain / _gfdn_tail_chain.sum()
    _ref_tail_chain = Eref_chain[_tail].mean(0)
    _ref_split_chain = _ref_tail_chain / _ref_tail_chain.sum()

    _gfdn_tail_complete = E_ex_complete[0, _tail].mean(0)
    _gfdn_split_complete = _gfdn_tail_complete / _gfdn_tail_complete.sum()
    _ref_tail_complete = Eref_complete[_tail].mean(0)
    _ref_split_complete = _ref_tail_complete / _ref_tail_complete.sum()

    _target_split = V / V.sum()

    _M1, _M2, _M3 = int(delays[:N1].sum()), int(delays[N1:N1 + N2].sum()), int(
        delays[N1 + N2:].sum())
    _equipartition_split = np.array([_M1, _M2, _M3]) / (_M1 + _M2 + _M3)
    _density_ratio_chain = _gfdn_split_chain / _equipartition_split
    _density_ratio_complete = _gfdn_split_complete / _equipartition_split



    mo.md(rf"""
    ## 4 · How closely does the GFDN track the physics?

    Max absolute deviation between the exact GFDN ledger and the true $e^{{tQ}}$ trajectory over the
    whole impulse response: chain = **{np.max(np.abs(_err_chain)):.4f}**, complete = **{np.max(np.abs(_err_complete)):.4f}**.

    **Numeric long-run split for chain topology** (mean over the last 15% of the simulation):

    | | room 1 | room 2 | room 3 |
    |---|---|---|---|
    | GFDN (exact ledger) | {_gfdn_split_chain[0]:.4f} | {_gfdn_split_chain[1]:.4f} | {_gfdn_split_chain[2]:.4f} |
    | reference $e^{{tQ}}$ | {_ref_split_chain[0]:.4f} | {_ref_split_chain[1]:.4f} | {_ref_split_chain[2]:.4f} |
    | target $V_i/V_{{\rm tot}}$ | {_target_split[0]:.4f} | {_target_split[1]:.4f} | {_target_split[2]:.4f} |
    | $M_i/M_{{\rm tot}}$ (equipartition target) | {_equipartition_split[0]:.4f} | {_equipartition_split[1]:.4f} | {_equipartition_split[2]:.4f} |
    | cell energy / global cell energy (target: 1) | {_density_ratio_chain[0]:.4f} | {_density_ratio_chain[1]:.4f} | {_density_ratio_chain[2]:.4f} |

    **Numeric long-run split for complete topology** (mean over the last 15% of the simulation):

    | | room 1 | room 2 | room 3 |
    |---|---|---|---|
    | GFDN (exact ledger) | {_gfdn_split_complete[0]:.4f} | {_gfdn_split_complete[1]:.4f} | {_gfdn_split_complete[2]:.4f} |
    | reference $e^{{tQ}}$ | {_ref_split_complete[0]:.4f} | {_ref_split_complete[1]:.4f} | {_ref_split_complete[2]:.4f} |
    | target $V_i/V_{{\rm tot}}$ | {_target_split[0]:.4f} | {_target_split[1]:.4f} | {_target_split[2]:.4f} |
    | $M_i/M_{{\rm tot}}$ (equipartition target) | {_equipartition_split[0]:.4f} | {_equipartition_split[1]:.4f} | {_equipartition_split[2]:.4f} |
    | cell energy / global cell energy (target: 1) | {_density_ratio_complete[0]:.4f} | {_density_ratio_complete[1]:.4f} | {_density_ratio_complete[2]:.4f} |

    **Separate incomplete equipartition from delay-allocation error.**
    Uniform mean energy per delay cell implies the $M_i/M_{{\rm tot}}$ split.
    It matches SA only when these delay proportions equal $V_i/V_{{\rm tot}}$.
    Cell-energy ratios near 1 therefore can coexist with a room-energy mismatch against SA.
    A fixed lossless matrix need not attain equipartition: residual coherence and
    source-dependent modal excitation can sustain a bias. Modulation can reduce this
    bias, but a finite-window measurement is not a proof of asymptotic convergence.
    The endpoint plot and ledger both use the last 15% of the valid simulation;
    endpoint-derived cell totals still have a finite-window shift error because
    different cells correspond to different time shifts of their delay-line output.
    """)
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
    db,
    mo,
    np,
    num_rooms,
    plt,
    pyFDN,
    tsec,
):
    h_mic_chain_lossy = np.stack([
        Y_chain_lossy[:, :, :N1].sum(-1), Y_chain_lossy[:, :,
                                                        N1:N1 + N2].sum(-1),
        Y_chain_lossy[:, :, N1 + N2:].sum(-1)
    ],
                                 axis=-1)
    h_mic_complete_lossy = np.stack([
        Y_complete_lossy[:, :, :N1].sum(-1),
        Y_complete_lossy[:, :, N1:N1 + N2].sum(-1),
        Y_complete_lossy[:, :, N1 + N2:].sum(-1)
    ],
                                    axis=-1)

    _fig, _axs = plt.subplots(3, 3, figsize=(8, 6.6), sharex=True, sharey=True)
    _tmax = tsec[-1]
    _nmax = int(_tmax * len(tsec) / tsec[-1])
    for _src in range(num_rooms):
        for _rec in range(num_rooms):
            _ax = _axs[_src, _rec]
            _ax.plot(tsec[:_nmax] * 1000,
                     db(
                         pyFDN.auxiliary.acoustics.edc(
                             h_mic_chain_lossy[_src, :_nmax, _rec])),
                     lw=1.0,
                     color="C0")
            _ax.plot(tsec[:_nmax] * 1000,
                     db(
                         pyFDN.auxiliary.acoustics.edc(
                             h_mic_complete_lossy[_src, :_nmax, _rec])),
                     lw=1.0,
                     color="C2")
            _ax.set_title(f"source room {_src+1} → receiver room {_rec+1}",
                          fontsize=9)
    for _ax in _axs[-1]:
        _ax.set_xlabel("time (ms)")
    for _ax in _axs[:, 0]:
        _ax.set_ylabel("$h(t)$")
    _axs[0, 0].set_ylim(-100, 20)
    _axs[0, 0].legend(['1-2, 2-3', '1-2, 1-3, 2-3'])
    _fig.suptitle("EDCs (first %.0f ms)" % (_tmax * 1000), fontsize=10)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


if __name__ == "__main__":
    app.run()
