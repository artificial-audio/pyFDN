"""Exercise the public validated LF route on all 30 modal example configurations.

Run from the checkout root with BLAS threads limited to one::

    python tests/benchmark_lindsey_fox.py

All successful examples require complete LF pole counts. The two absorption
examples may report search failure; no other solver is run. Assertions check
accuracy and known failure cases, not timings.
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
        absorption = "model" in case and case["filters"].get("post_delay") is not None
        try:
            if "model" in case:
                result = pyFDN.flamo_to_pr(
                    case["model"],
                    mode="lindsey_fox",
                    num_poles=case["degree"],
                )
                dss: pyFDN.DSS = case["dss"]
                graph = pyFDN.dss_to_td(*dss, **case["filters"])
                reference = graph.impulse_response(48000)
            else:
                args = pyFDN.DSS(*(case[k] for k in ("delays", "A", "B", "C", "D")))
                result = pyFDN.dss_to_pr(*args, mode="lindsey_fox")
                reference = pyFDN.dss_to_impz(*args, 48000)
            elapsed = perf_counter() - start
            ir = pyFDN.pr_to_impz(*result[:4], 48000, mode="lowMemory")
            error = float(np.linalg.norm(ir - reference) / np.linalg.norm(reference))
            meta = result[4]
            assert error < 1e-8, error
            assert meta["complete"] and meta["solver"] == "lindsey_fox"
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
                    }
                ),
                flush=True,
            )
        except Exception as error:
            expected_failure = (
                absorption
                and isinstance(error, RuntimeError)
                and ("Lindsey-Fox search" in str(error))
            )
            if not expected_failure:
                failures.append(case["name"])
            print(
                json.dumps(
                    {
                        "example": case["name"],
                        "status": "expected_failure"
                        if expected_failure
                        else type(error).__name__,
                        "decomposition_seconds": perf_counter() - start,
                        "error": str(error),
                    }
                ),
                flush=True,
            )
    assert not failures, failures


if __name__ == "__main__":
    main()
