"""Reproducible modal-decomposition trial; timings are not CI assertions.

Run from the checkout with a fixed BLAS thread count, for example::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_lindsey_fox.py

Use --scales 1 4 10 for larger systems, --modes eig roots lindsey_fox eai
to include EAI, and --repeat 3 for median timings. Imports are warmed up before
timing; every EAI call includes model construction. Timed calls include pole
search, conjugate pairing, and residue extraction. The first mode is the
numerical reference.
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from pyFDN.translate.dss_to_impz import dss_to_impz
from pyFDN.translate.dss_to_pr import dss_to_pr
from pyFDN.translate.pr_to_impz import pr_to_impz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=["eig", "roots", "lindsey_fox"])
    args = parser.parse_args()
    if args.repeat < 1 or min(args.scales) < 1:
        parser.error("repeat and scales must be positive")
    rng = np.random.default_rng(53)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    a = 0.95 * q
    b = rng.normal(size=(3, 1))
    c = rng.normal(size=(1, 3))
    d = np.zeros((1, 1))
    print("degree,mode,median_seconds,max_pole_error,max_ir_error")
    for mode in args.modes:
        dss_to_pr([3, 4, 5], a, b, c, d, mode=mode)
    failed = False
    for scale in args.scales:
        delays = np.array([31, 37, 43]) * scale + [0, 2, 4]
        ir_time = dss_to_impz(delays, a, b, c, d, 512)
        reference = None
        for mode in args.modes:
            elapsed = []
            try:
                for _ in range(args.repeat):
                    start = perf_counter()
                    result = dss_to_pr(delays, a, b, c, d, mode=mode)
                    elapsed.append(perf_counter() - start)
                residues, poles, direct, pairs, meta = result
                assert np.sum(1 + pairs) == delays.sum(), "missing poles"
                assert meta["nonPairedPoles"].size == 0, "unpaired poles"
                if reference is None:
                    reference = poles
                row, col = linear_sum_assignment(np.abs(reference[:, None] - poles))
                pole_error = np.max(np.abs(reference[row] - poles[col]))
                ir = pr_to_impz(residues, poles, direct, pairs, 512)
                ir_error = np.max(np.abs(ir - ir_time))
                assert pole_error < 1e-8 and ir_error < 1e-7, "accuracy check failed"
                print(
                    f"{delays.sum()},{mode},{np.median(elapsed):.6f},"
                    f"{pole_error:.3e},{ir_error:.3e}",
                    flush=True,
                )
            except (RuntimeError, AssertionError) as error:
                failed = True
                print(f"{delays.sum()},{mode},FAILED,{error}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
