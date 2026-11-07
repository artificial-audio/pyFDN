"""Helper functions needed for physics-based coupling"""
import numpy as np
from scipy.linalg import block_diag, expm
from typing import Optional, Tuple, List, Union
from numpy.typing import NDArray, ArrayLike

import pyFDN
from pyFDN.td import Gain, Series, TimeOperator, TimeVaryingMatrix


def create_lossless_coupling_matrix(aperture_area: NDArray,
                                    volume: ArrayLike,
                                    c: float = 343):
    """Create the coupling coefficient from Cremer-Muller theory"""
    num_rooms = len(volume)
    beta = np.zeros((num_rooms, num_rooms))
    for i in range(num_rooms):
        for j in range(num_rooms):
            if i == j:
                continue
            else:
                beta[i, j] = c * aperture_area[i, j] / (4 * volume[i])
    return beta


def create_diagonal_absorption_matrix(absorp_area: List,
                                      volume: List,
                                      c: float = 343):
    """Create the diagonal absorption matrix from Cremer-Muller theory"""
    gamma = (c * absorp_area) / (4 * volume)
    return np.diag(gamma)


def create_state_transition_matrix(beta: NDArray):
    """Create the state transition matrix of the CT system from Cremer-Muller theory"""
    Q = beta.T.copy()
    np.fill_diagonal(Q, -beta.sum(axis=1))
    return Q


def trajectory(Q: NDArray,
               src_weight: Union[ArrayLike, NDArray],
               mic_weight: Union[ArrayLike, NDArray],
               fs: float,
               n_samp: int,
               gamma: Optional[NDArray] = None):
    r"""
    True physical trajectory e(t) = c_R^\top expm(Q t) b_S = c_R^\top V exp(\Lambda t) V^{-1} b_S
    Args:
        Q (NDarray): state transition matrix of size Nrooms x Nrooms
        _src_weight (ArrayLike): source weightings for each room of size num_src x Nrooms
        _mic_weight (NDArray, ArrayLike): receiver weightings for each room of size num_rec x Nrooms
        fs (float): sampling frequency
        n_samp (int): number of time samples
        _gamma (NDArray, optional): diagonal absorption matrix
    Returns:
        NDArray: the common decays and the energy decay trajectory of size num_src x num_time_samps x num_rec
    """

    if gamma is None:
        gamma = np.zeros_like(Q)
    w, Vv = np.linalg.eig(Q - gamma)
    # Q is reversible w.r.t. pi ~ V, so spectrum is real
    w, Vv = w.real, Vv.real
    Vinv = np.linalg.inv(Vv)
    coeffs = Vinv @ src_weight
    tsec = np.arange(n_samp) / fs
    modes = np.exp(np.outer(tsec, w))
    T = np.einsum("ik,tk->ti", Vv, modes * coeffs[None, :])
    return w, np.einsum("ti, mi -> tm", T, mic_weight)


def trajectory_with_delays(
    Q: NDArray,
    src_weight: ArrayLike,
    rec_weight: ArrayLike,
    src_delay: NDArray,
    rec_delay: NDArray,
    src_room: List[int],
    rec_room: List[int],
    fs: float,
    n_samp: int,
    gamma: Optional[NDArray] = None,
) -> NDArray:
    """
    Continuous-time Markov model with per-room propagation delays between
    each source/receiver and each room, to account for multiple aperture
    chains (different rooms may be reached through different sequences of
    apertures, so a single shared delay per source/receiver is no longer
    correct -- the delay is room-specific).

    Args:
        src_weight : (n_src, n_room)
        rec_weight : (n_rec, n_room)
        src_room   : (n_src,) room index of each source
        rec_room   : (n_rec,) room index of each receiver
        src_delay  : (n_src, n_room) source->room aperture-chain delay (samples)
        rec_delay  : (n_rec, n_room) room->receiver aperture-chain delay (samples)
    Returns:
        w   : (n_room,) eigenvalues (common decay rates)
        out : (n_samp, n_src, n_rec) energy decay trajectory
    """
    src_delay = np.asarray(src_delay)
    rec_delay = np.asarray(rec_delay)
    if gamma is None:
        gamma = np.zeros_like(Q)
    A = Q - gamma
    w, V = np.linalg.eig(A)
    w = w.real
    V = V.real
    Vinv = np.linalg.inv(V)
    t = np.arange(n_samp) / fs
    modes = np.exp(np.outer(t, w))
    n_src = src_weight.shape[0]
    n_rec = rec_weight.shape[0]
    n_room = src_weight.shape[1]

    assert src_delay.shape == (
        n_src,
        n_room), f"src_delay must be (n_src, n_room), got {src_delay.shape}"
    assert rec_delay.shape == (
        n_rec,
        n_room), f"rec_delay must be (n_rec, n_room), got {rec_delay.shape}"

    def delay_room_energy(room_energy: NDArray, delays: NDArray,
                          skip_room: int) -> NDArray:
        """Shift each room's trace by its own delay; leave `skip_room` untouched."""
        out = room_energy.copy()
        for room in range(n_room):
            if room == skip_room:
                continue
            L = int(delays[room])
            if L <= 0:
                continue
            out[:, room] = 0.0
            if L < n_samp:
                out[L:, room] = room_energy[:-L, room]
        return out

    out = np.zeros((n_samp, n_src, n_rec))
    for s in range(n_src):
        coeff = Vinv @ src_weight[s]
        room_energy = (V @ (modes * coeff[None, :]).T).T
        room_energy = delay_room_energy(room_energy, src_delay[s], src_room[s])

        for r in range(n_rec):
            room_energy_obs = delay_room_energy(room_energy, rec_delay[r],
                                                rec_room[r])
            out[:, s, r] = room_energy_obs @ rec_weight[r]

    return w, out


def match_delay_totals_to_volumes(base_delays: ArrayLike, lines_per_group:int, volumes:ArrayLike, target_total=None):
    """Match integer volume ratios exactly, retaining each group's delay spread.

    Start from independently sampled coprime delays, then apportion integer
    lengths to exact group totals. No coprimality constraint is imposed on
    the result. Its total number of cells stays close to the original total.
    Supply target_total to choose the nearest feasible total to a requested
    global mean instead of retaining the original sampled total.
    """
    volumes = np.asarray(volumes, dtype=float)
    integer_volumes = np.rint(volumes).astype(np.int64)
    if np.any(volumes <= 0) or not np.allclose(volumes, integer_volumes):
        raise ValueError("Exact-ratio comparison requires positive integer volumes.")
    weights = integer_volumes // np.gcd.reduce(integer_volumes)
    base = np.asarray(base_delays, dtype=np.int64).reshape(
        len(volumes), lines_per_group)
    state_count = base.sum() if target_total is None else target_total
    multiplier = max(
        int(np.rint(state_count / weights.sum())),
        int(np.max(np.ceil(lines_per_group / weights))),
    )
    targets = multiplier * weights
    matched = []
    for group, target in zip(base, targets):
        # Reserve one cell per line, then use largest-remainder rounding.
        ideal = 1 + (target - lines_per_group) * group / group.sum()
        lengths = np.floor(ideal).astype(np.int64)
        remainder = int(target - lengths.sum())
        order = np.argsort(-(ideal - lengths), kind="stable")
        lengths[order[:remainder]] += 1
        matched.append(lengths)
    return np.concatenate(matched)



def get_coupling_angles(beta: NDArray, dt_i: List) -> NDArray:
    """Rate-match two-room coupling; use the pairwise rule approximately for N>2.

    With volume-proportional storage, beta_ij * dt_i = C_ij / H.
    Directional values are averaged when integer storage only approximates
    that ratio. The full skew exponential is not an exact multi-room match.
    """
    beta = np.asarray(beta, dtype=float)
    dt_i = np.asarray(dt_i, dtype=float)
    n = len(dt_i)
    if beta.shape != (n, n) or not np.all(np.isfinite(beta)) or np.any(beta < 0):
        raise ValueError("beta must be a finite nonnegative square matrix matching dt_i")
    if dt_i.ndim != 1 or not np.all(np.isfinite(dt_i)) or np.any(dt_i <= 0):
        raise ValueError("dt_i must contain positive finite mean traversal times")
    probabilities = beta * dt_i[:, None]
    probabilities = 0.5 * (probabilities + probabilities.T)
    if np.any(probabilities > 1):
        raise ValueError("Coupling fraction exceeds one; reduce aperture area or delay storage")
    theta = np.arcsin(np.sqrt(probabilities))
    np.fill_diagonal(theta, 0.0)
    return theta


def get_coupling_matrix(theta: NDArray):
    """Construct a skew symmetric matrix from the coupling coefficients"""
    tri = np.triu(theta, 1)
    return tri - tri.T


def lift(R_room: NDArray, Nroom: int):
    """
    Lift R_room from num_rooms x num_rooms to num_rooms x Nroom, num_rooms x Nroom.
    Nroom is the number of delay lines per room (same for each room)
    """
    # Simpler lifting with kroneckers
    num_rooms = R_room.shape[0]
    mat = np.zeros((num_rooms * Nroom, num_rooms * Nroom))
    for _i in range(num_rooms):
        for _j in range(num_rooms):
            mat[_i * Nroom:(_i + 1) * Nroom,
                _j * Nroom:(_j + 1) * Nroom] = R_room[_i, _j] * np.eye(Nroom)
    return mat


def get_decay(_gamma: ArrayLike, _M: ArrayLike, fs: float):
    """Get the diagonal decay matrix" of size num_rooms x num_rooms"""
    return np.diag(np.exp(-_gamma * _M / (2 * fs)))


def get_decay_matrix(gamma: NDArray, delays_per_fdn: NDArray, fs: float):
    """
    Get the block diagonal decay matrix
    Args:
        gamma (NDArray): num_rooms x num_rooms absorption matrix derived from physics
        delays_per_fdn (List): num_rooms x Nroom delay line lengths
        fs (float): sampling frequency
    """
    num_rooms = gamma.shape[0]
    Gamma = []
    for _i in range(num_rooms):
        Gamma.append(get_decay(gamma[_i, _i], delays_per_fdn[_i], fs))
    return block_diag(*Gamma)


def get_feedback_matrix(R_room: NDArray,
                        Qblocks: NDArray,
                        Nroom: int,
                        D: Optional[NDArray] = None):
    """
    Get the GFDN orthonormal feedback matrix.
    Args:
        R_room (NDArray): num_rooms x num_rooms coupling matrix
        Qblocks (NDArray: block diagonal per-room feedback matrix of shape num_rooms x Nroom, num_rooms x Nroom
        D (NDArray, optional): compensating matrix if the delay line ratio does not match the volume ratio exactly
    """
    R_room_lift = lift(R_room, Nroom)
    if D is None:
        return R_room_lift @ Qblocks
    else:
        return D @ R_room_lift @ np.linalg.inv(D) @ Qblocks


def get_room_time_varying_matrix(
    R_room: ArrayLike,
    Nroom: int,
    cycles_per_second: float = 1.0,
    amplitude: float = 3.0,
    fs: float = 48000.0,
    spread: float = 0.3,
) -> Series:
    r"""Construct a feedback hook that modulates only the internal room mixers.

    Keep the usual static feedback matrix ``A = F @ M @ G``, where
    ``F = kron(R_room, eye(Nroom))``, ``M`` contains the block-diagonal room
    mixers and ``G`` is optional diagonal absorption. This hook applies
    ``F @ R(n) @ F.T`` AFTER ``A``, so the effective feedback is exactly
    ``F @ R(n) @ M @ G``. The coupling matrix ``R_room`` stays fixed.

    ``R(n)`` uses the adjacent-pair rotations of :class:`TimeVaryingMatrix`.
    Since each room has an even number of contiguous channels, all rotations
    remain inside their room. Each pair has its own random modulation phase,
    frequency and amplitude, as in ``TimeVaryingMatrix``. Seed NumPy before
    construction for reproducible parameters.

    Parameters
    ----------
    R_room : array-like, shape (num_rooms, num_rooms)
        Real orthogonal room coupling matrix, before lifting to line space.
    Nroom : int
        Positive even number of delay lines per room, equal for all rooms.
    cycles_per_second, amplitude, fs, spread : float
        Modulation settings, with the same meaning as ``TimeVaryingMatrix``.

    Returns
    -------
    Series
        Stateful runtime operator for ``process_dss(..., post_matrix=hook)``
        or ``FDNBuild.post_matrix`` with ``process_fdn``. Call ``hook.reset()``
        before independent renders or stereo-channel renders that should
        start with identical modulation. This is a runtime hook, not an SOS
        array for build serialisation or static frequency-domain analysis.

    Examples
    --------
    Use the same coupling matrix that was passed to ``get_feedback_matrix``::

        hook = get_room_time_varying_matrix(Phi, K, fs=fs)
        y = pyFDN.process_dss(x, delays, A, B, C, D, post_matrix=hook)

    A frozen ``FDNBuild`` can be updated using ``dataclasses.replace``::

        from dataclasses import replace
        build = replace(build, post_matrix=hook)
        hook.reset()
        y = pyFDN.process_fdn(x, build)

    Do not use this hook with a similarity-scaled coupling matrix or with
    ``A`` containing only the internal mixers: it expects the standard
    combined feedback ``F @ M @ G``.
    """
    if np.iscomplexobj(R_room):
        raise ValueError("R_room must be a real orthogonal matrix")
    coupling = np.asarray(R_room, dtype=float)
    if coupling.ndim != 2 or coupling.shape[0] == 0 or coupling.shape[
            0] != coupling.shape[1]:
        raise ValueError("R_room must be a non-empty square matrix")
    if not np.all(np.isfinite(coupling)) or not np.allclose(
            coupling.T @ coupling, np.eye(len(coupling)), rtol=0, atol=1e-10):
        raise ValueError("R_room must be a real orthogonal matrix")
    if (isinstance(Nroom, (bool, np.bool_))
            or not isinstance(Nroom,
                              (int, np.integer)) or Nroom <= 0 or Nroom % 2):
        raise ValueError("Nroom must be a positive even integer")
    if not np.all(np.isfinite([cycles_per_second, amplitude, fs, spread
                               ])) or fs <= 0:
        raise ValueError(
            "Modulation parameters must be finite and fs must be positive")

    coupling_lift = np.kron(coupling, np.eye(Nroom))
    rotation = TimeVaryingMatrix(
        len(coupling) * Nroom, cycles_per_second, amplitude, fs, spread)
    # Series acts left to right: F.T first, then room rotations, then F.
    return Series([Gain(coupling_lift.T), rotation, Gain(coupling_lift)])


def run_gfdn(A,
             B: NDArray,
             C: NDArray,
             delays: ArrayLike,
             n_samp: int,
             src: Optional[Union[int, List]] = None,
             tv_matrix: Optional[TimeOperator] = None):
    """
    Generic wrapper around process_dss.
    Args:
        A (NDArray): Ntot x Ntot feedback matrix (with losses)
        B (NDArray): (Ntot, num_inputs) input matrix
        C (NDArray) : (num_outputs, Ntot) output matrix
        delays (ArrayLike): delay line lengths in samples
        src (int, list, optional): input indices to excite; None renders every input
        n_samp (int): number of time samples
        tv_matrix (TimeOperator, optional): feedback hook, including the operator
            returned by get_room_time_varying_matrix. Reset it before independent
            renders when identical initial modulation is required.
    Returns:
        NDArray : GFDN output of size (number of selected inputs, n_samp, num_outputs)
    """

    num_inputs = B.shape[1]
    num_outputs = C.shape[0]

    sources = np.arange(num_inputs) if src is None else np.atleast_1d(src)
    if any(not isinstance(k, (int, np.integer)) or k < 0 or k >= num_inputs for k in sources):
        raise ValueError("src must select valid input indices")
    Y = np.zeros((len(sources), n_samp, num_outputs))
    for n_src, k in enumerate(sources):
        x = np.zeros((n_samp, num_inputs))
        x[0, k] = 1.0

        if tv_matrix is not None:
            tv_matrix.reset()
        Y[n_src] = pyFDN.process_dss(
            x,
            delays,
            A,
            B,
            C,
            np.zeros((num_outputs, num_inputs)),
            post_matrix=tv_matrix,
        )
    return Y


def gfdn_energy_ledger(Y: NDArray, num_rooms: int, Nroom: int,
                       delays: ArrayLike):
    """
    Calculate the GFDN energy ledger
    Args:
        Y (NDArray): of size num_inputs x n_samp x num_outputs
        num_rooms (int): number of rooms in GFDN
        Nroom (int): number of delay lines per room
        delays (ArrayLike): delay line lengths in samples
    Returns:
        NDArray: energy ledger of shape num_rooms x time samples x num_rooms
    """
    Ntot = num_rooms * Nroom
    # cumulative sum of the energy
    cs = np.concatenate(
        [np.zeros((Y.shape[0], 1, Ntot)),
         np.cumsum(Y**2, axis=1)], axis=1)
    n_samp = Y.shape[1]
    n_ex = n_samp - int(delays.max()) - 1
    if n_ex <= 0:
        raise ValueError("Simulation must exceed the longest delay")
    idx = np.arange(n_ex)
    E = np.zeros((Y.shape[0], n_ex, num_rooms))
    bounds = [i * Nroom for i in range(num_rooms)]
    for _j, _m in enumerate(delays):
        # current room
        _room = next((i - 1 for i, b in enumerate(bounds) if _j < b),
                     len(bounds) - 1)
        E[:, :, _room] += cs[:, idx + 1 + _m, _j] - cs[:, idx + 1, _j]
    return E, n_ex


def room_energy_ledger_from_rirs(
    rirs: np.ndarray,
    receiver_positions: np.ndarray,
    ROOM_START: List,
    ROOM_DIMS: List,
):
    """
    Estimate the spatially integrated room energy from a dense set of RIRs.
    Args:
        rirs : (Nrec, Nt)
            Pressure RIRs.
        receiver_positions : (Nrec,3)
            Receiver coordinates.
        ROOM_START (List): room start coordinates for each room
        ROOM_DIMS (List): room dimensions for each room
    Returns:
        room_energy : (Nroom,Nt)
            Estimated room energies.
        room_masks : List[np.ndarray]
            Boolean mask of receivers belonging to each room.
    """
    room_bounds = []
    n_rooms = len(ROOM_START)
    for i in range(n_rooms):
        room_bounds.append((np.array(ROOM_START[i]),
                            np.array(ROOM_START[i]) + np.array(ROOM_DIMS[i])))
    n_samples = rirs.shape[1]
    room_energy = np.zeros((n_rooms, n_samples))
    room_masks = []

    pressure2 = rirs**2
    for room_idx, (xyz_min, xyz_max) in enumerate(room_bounds):
        mask = np.all(
            (receiver_positions >= xyz_min)
            & (receiver_positions <= xyz_max),
            axis=1,
        )

        room_masks.append(mask)
        room_energy[room_idx] = pressure2[mask].mean(axis=0)

    return room_energy, room_masks
