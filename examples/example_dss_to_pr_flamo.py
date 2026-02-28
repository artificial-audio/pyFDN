"""
Dedicated FLAMO/autograd DSS2PR example.

This example intentionally uses only the FLAMO probing architecture:
``dss_to_pr_flamo`` (autograd backend).
"""

from __future__ import annotations

import numpy as np

import pyFDN
from pyFDN.auxiliary.flamo import gain_module


def main() -> None:
    np.random.seed(7)

    # Small stable FDN
    n = 4
    delays = np.array([53, 67, 79, 97], dtype=int)
    a_num = 0.7 * pyFDN.random_orthogonal(n)
    b = np.eye(n, 1)
    c = np.eye(1, n)
    d = np.zeros((1, 1))

    # FLAMO graph for the feedback matrix A(z)
    feedback_graph = gain_module(a_num, nfft=2**12, device="cpu")

    # FLAMO-only modal decomposition path
    residues, poles, direct, is_pair, _ = pyFDN.dss_to_pr_flamo(
        delays,
        feedback_graph,  # graph input
        b,
        c,
        d,
        feedback_delay_units=0,
        maximum_iterations=70,
        verbose=False,
    )

    ir_len = 1024
    ir_time = pyFDN.dss_to_impz(ir_len, delays, a_num, b, c, d)[:, 0, 0]
    ir_modal = pyFDN.pr_to_impz(residues, poles, direct, is_pair, ir_len)[:, 0, 0]

    err = np.max(np.abs(ir_time - ir_modal))
    print("max |IR_time - IR_modal|:", err)
    assert err < 1e-7


if __name__ == "__main__":
    main()

