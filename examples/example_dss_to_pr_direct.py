"""
Dedicated direct DSS2PR example.

Uses ``dss_to_pr_direct`` (numeric DSS-only path).
"""

from __future__ import annotations

import numpy as np

import pyFDN

import matplotlib.pyplot as plt

def main() -> None:
    np.random.seed(11)

    n = 4
    delays = np.array([41, 53, 67, 79], dtype=int)
    A = 0.65 * pyFDN.random_orthogonal(n)
    b = np.eye(n, 1)
    c = np.eye(1, n)
    d = np.ones((1, 1))

    ir_len = 1024
    ir_time = pyFDN.dss_to_impz(ir_len, delays, A, b, c, d)[:, 0, 0]
    ir_modals = {}
    modes = ["eig", "roots", "polyeig"]
    for mode in modes:
        residues, poles, direct, is_pair, _ = pyFDN.dss_to_pr_direct(
            delays,
            A,
            b,
            c,
            d,
            mode=mode,
        )
        ir_modals[mode] = pyFDN.pr_to_impz(residues, poles, direct, is_pair, ir_len)[:, 0, 0]
        err = np.max(np.abs(ir_time - ir_modals[mode]))
        print("max |IR_time - IR_modal|:", err)
        assert err < 1e-7

    plt.figure(figsize=(10, 4))
    plt.plot(pyFDN.mulaw_encode(ir_time), label="IR from dss_to_impz", linewidth=1.2)
    for mode in modes:
        plt.plot(pyFDN.mulaw_encode(ir_modals[mode]), "--", label=f"IR from {mode}", linewidth=1.2)
    plt.title("DSS time response vs modal reconstruction")
    plt.xlabel("Time [samples]")
    plt.ylabel("Amplitude [mu-law]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

