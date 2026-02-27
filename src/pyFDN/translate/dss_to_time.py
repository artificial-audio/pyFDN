"""
Convert delay state-space (A, B, C, D, m) into a time-domain graph prototype.

This mirrors `dss_to_flamo`, but instantiates FLAMO-like time-domain modules from
`pyFDN.auxiliary.flamo_time`.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from pyFDN.auxiliary.flamo_time import (
    Shell,
    delay_module,
    gain_module,
    sos_filter_module,
    system,
)


def dss_to_time(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    m: np.ndarray,
    Fs: float,
    *,
    block_size: int = 512,
    shell: bool = True,
    sos_filter: np.ndarray | None = None,
    post_delay_module: Any = None,
    feedback_delay_blocks: int = 1,
) -> Shell | Any:
    """
    Build a time-domain graph from delay state-space (A, B, C, D, m).

    Signal flow:
      input -> B -> [Recursion: delay -> (optional filter/module), feedback A] -> C
      in parallel with direct path D, summed at output.
    """

    A_arr = np.asarray(A, dtype=np.float32)
    B_arr = np.asarray(B, dtype=np.float32)
    C_arr = np.asarray(C, dtype=np.float32)
    D_arr = np.asarray(D, dtype=np.float32)
    m_arr = np.asarray(m, dtype=np.float32).reshape(-1)

    n_lines = int(A_arr.shape[0])
    if A_arr.shape != (n_lines, n_lines):
        raise ValueError(f"A must be square [N, N], got {A_arr.shape}")
    if m_arr.shape[0] != n_lines:
        raise ValueError(
            "m must have one delay per line; "
            f"expected {n_lines}, got {m_arr.shape[0]}"
        )
    if B_arr.shape[0] != n_lines:
        raise ValueError(f"B first dimension must be N={n_lines}, got {B_arr.shape}")
    if C_arr.shape[1] != n_lines:
        raise ValueError(f"C second dimension must be N={n_lines}, got {C_arr.shape}")
    if D_arr.shape != (C_arr.shape[0], B_arr.shape[1]):
        raise ValueError(
            "D must be [N_out, N_in] matching C and B; "
            f"got D={D_arr.shape}, C={C_arr.shape}, B={B_arr.shape}"
        )

    lengths_seconds = m_arr / float(Fs)
    delays = delay_module(lengths_seconds, Fs=float(Fs))
    gain_A = gain_module(A_arr)
    gain_B = gain_module(B_arr)
    gain_C = gain_module(C_arr)
    gain_D = gain_module(D_arr)

    if sos_filter is not None:
        filter_module = sos_filter_module(np.asarray(sos_filter, dtype=np.float32))
        delay_chain = system.Series(
            OrderedDict({"delay": delays, "filter": filter_module})
        )
    else:
        delay_chain = delays

    if post_delay_module is not None:
        delay_chain = system.Series(
            OrderedDict({"delay": delay_chain, "post_delay_module": post_delay_module})
        )

    feedback_loop = system.Recursion(
        fF=delay_chain,
        fB=gain_A,
        feedback_delay_blocks=feedback_delay_blocks,
    )
    fdn_branch = system.Series(
        OrderedDict(
            {
                "input_gain": gain_B,
                "feedback_loop": feedback_loop,
                "output_gain": gain_C,
            }
        )
    )
    core = system.Parallel(brA=fdn_branch, brB=gain_D, sum_output=True)

    if shell:
        return system.Shell(core=core, block_size=block_size)
    return core

