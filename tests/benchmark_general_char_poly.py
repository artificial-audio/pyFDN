"""Compare one-time polynomial construction as the number of delay lines grows.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 from an editable installation::

    python tests/benchmark_general_char_poly.py --lines 4 8 12 16 --repeat 3

Checks coefficient agreement before reporting median construction times. The
principal-minor reference becomes exponential; choose --lines accordingly.
"""

import argparse
from time import perf_counter

import numpy as np

from pyFDN.auxiliary.math import general_char_poly, general_char_poly_fft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", nargs="+", type=int, default=[4, 8, 12, 16])
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if min(args.lines) < 1 or args.repeat < 1:
        parser.error("lines and repeat must be positive")
    print(
        "lines,degree,principal_seconds,fft_seconds,max_coefficient_error", flush=True
    )
    for n in args.lines:
        rng = np.random.default_rng(73)
        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        a = 0.9 * q
        delays = 31 + 2 * np.arange(n)
        results = []
        timings = []
        for function in (general_char_poly, general_char_poly_fft):
            elapsed = []
            for _ in range(args.repeat):
                start = perf_counter()
                coefficients = function(delays, a)
                elapsed.append(perf_counter() - start)
            results.append(coefficients)
            timings.append(np.median(elapsed))
        error = np.max(np.abs(results[0] - results[1]))
        np.testing.assert_allclose(results[0], results[1], atol=1e-10, rtol=1e-10)
        print(
            f"{n},{delays.sum()},{timings[0]:.6f},{timings[1]:.6f},{error:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
