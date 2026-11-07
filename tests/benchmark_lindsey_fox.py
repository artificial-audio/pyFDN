"""Exercise the public validated LF route on all 30 modal example configurations.

Run from the checkout root with BLAS threads limited to one::

    python tests/benchmark_lindsey_fox.py

The two absorption examples explicitly opt into approximation. All other
examples require complete pole counts. Assertions check accuracy, not timings.
"""

import contextlib
import json
import sys
from time import perf_counter

import modal_examples as bench
import numpy as np

import pyFDN


def main():
    with contextlib.redirect_stdout(sys.stderr):
        cases = bench.cases()
    failures = []
    for case in cases:
        start = perf_counter()
        try:
            if "model" in case:
                approximate = case["build"].post_delay is not None
                result = pyFDN.flamo_to_pr(
                    case["model"],
                    mode="lindsey_fox",
                    num_poles=case["degree"],
                    allow_approximate=approximate,
                    maximum_iterations=80,
                )
                reference = pyFDN.build_to_impz(case["build"], 48000)
            else:
                args = [case[k] for k in ("delays", "A", "B", "C", "D")]
                result = pyFDN.dss_to_pr(
                    *args, mode="lindsey_fox", maximum_iterations=80
                )
                reference = pyFDN.dss_to_impz(*args, 48000)
            elapsed = perf_counter() - start
            ir = pyFDN.pr_to_impz(*result[:4], 48000, mode="lowMemory")
            error = float(np.linalg.norm(ir - reference) / np.linalg.norm(reference))
            meta = result[4]
            assert error < 1e-8, error
            assert meta["complete"] or ("model" in case and approximate)
            print(
                json.dumps(
                    {
                        "example": case["name"],
                        "status": "ok",
                        "decomposition_and_reference_seconds": elapsed,
                        "ir_relative_l2": error,
                        **{
                            key: meta[key]
                            for key in (
                                "solver",
                                "complete",
                                "approximate",
                                "expectedPoles",
                                "returnedPoles",
                                "responseValidation",
                            )
                        },
                        "fallbackReason": meta.get("fallbackReason"),
                    }
                ),
                flush=True,
            )
        except Exception as error:
            failures.append(case["name"])
            print(
                json.dumps(
                    {
                        "example": case["name"],
                        "status": type(error).__name__,
                        "error": str(error),
                    }
                ),
                flush=True,
            )
    assert not failures, failures


if __name__ == "__main__":
    main()
