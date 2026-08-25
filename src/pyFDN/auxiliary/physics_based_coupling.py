"""Helper functions needed for physics-based coupling"""
import numpy as np
from scipy.linalg import block_diag, expm
from typing import Optional, Tuple, List, Union
from numpy.typing import NDArray, ArrayLike

import pyFDN
from pyFDN.dsp.time_varying_matrix import TimeVaryingMatrix


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
    Q = beta.copy()
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


def get_coupling_angles(beta: NDArray, dt_i: List) -> NDArray:
    """
    Get the coupling angles from the coupling coefficients
    and mean free path lengths for all FDNs
    """
    n = len(dt_i)
    theta = np.zeros((n, n))
    for _i in range(n):
        for _j in range(_i + 1, n):
            val = 0.5 * (beta[_i, _j] * dt_i[_i] + beta[_j, _i] * dt_i[_j])
            val = 0.999 if val > 0.999 else val
            theta[_i,
                  _j] = theta[_j,
                              _i] = np.arcsin(np.sqrt(val)) if val > 0 else 0.0
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


def run_gfdn(A,
             _B: NDArray,
             _C: NDArray,
             delays: ArrayLike,
             n_samp: int,
             _src: Union[int, List] = 0,
             tv_matrix: Optional[TimeVaryingMatrix] = None):
    """
    Generic wrapper around process_fdn.
    Args:
        A (NDArray): Ntot x Ntot feedback matrix (with losses)
        _B (NDArray): (Ntot, num_inputs) input matrix
        _C (NDArray) : (num_outputs, Ntot) output matrix
        delays (ArrayLike): delay line lengths in samples
        _src (int, list): room where the source is
        n_samp (int): number of time samples
        tv_matrix (Optional): if using a time varying matrix for increased mixing
    Returns:
        NDArray : GFDN output of size (num_inputs, n_samp, num_outputs)
    """

    num_inputs = _B.shape[1]
    num_outputs = _C.shape[0]

    Y = np.zeros((num_inputs, n_samp, num_outputs))
    for n_src in range(num_inputs):
        x = np.zeros((n_samp, num_inputs))
        k = _src[n_src] if isinstance(_src, list) else _src
        x[0, k] = 1.0

        Y[n_src] = pyFDN.process_fdn(
            x,
            delays,
            A,
            _B,
            _C,
            np.zeros((num_outputs, num_inputs)),
            extra_matrix=tv_matrix,
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
        [np.zeros((num_rooms, 1, Ntot)),
         np.cumsum(Y**2, axis=1)], axis=1)
    n_samp = Y.shape[1]
    n_ex = n_samp - int(delays.max()) - 1
    idx = np.arange(n_ex)
    E = np.zeros((num_rooms, n_ex, num_rooms))
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
