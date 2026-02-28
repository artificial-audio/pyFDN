"""
Dedicated FLAMO/autograd DSS2PR example.

This example intentionally uses the FLAMO-first split:
``dss_to_flamo`` then ``flamo_to_pr``.
"""

from __future__ import annotations

import numpy as np
import torch

import pyFDN


def main() -> None:
    np.random.seed(7)

    # Small stable FDN
    n = 4
    delays = np.array([53, 67, 79, 97], dtype=int)
    a_num = 0.7 * pyFDN.random_orthogonal(n)
    b = np.eye(n, 1)
    c = np.eye(1, n)
    d = np.zeros((1, 1))

    # Preferred split: DSS -> FLAMO model, then FLAMO -> PR.
    # Use float64 in FLAMO for tighter numerical agreement.
    model = pyFDN.dss_to_flamo(
        A=a_num,
        B=b,
        C=c,
        D=d,
        m=delays,
        Fs=1.0,
        nfft=2**12,
        shell=False,
        dtype=torch.float64,
    )
    core = model.get_core() if callable(getattr(model, "get_core", None)) else model
    recursion_module = list(core.branchA)[1]
    residues, poles, direct, is_pair, _ = pyFDN.flamo_to_pr(
        model=model,
        delays=delays,
        recursion_module=recursion_module,
        feedback_delay_units=0,
        quality_threshold=1e-10,
        refinement_tol=1e-7,
        maximum_iterations=40,
        verbose=False,
    )

    ir_len = 1024
    ir_time = pyFDN.dss_to_impz(ir_len, delays, a_num, b, c, d)[:, 0, 0]
    ir_modal = pyFDN.pr_to_impz(residues, poles, direct, is_pair, ir_len)[:, 0, 0]

    err = np.max(np.abs(ir_time - ir_modal))
    print("max |IR_time - IR_modal|:", err)
    assert err < 1e-5


if __name__ == "__main__":
    main()

