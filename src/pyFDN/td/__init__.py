"""Time-domain rendering of FLAMO-style processing graphs.

This subpackage mirrors a FLAMO model's Shell/Series/Parallel/Recursion/leaf
structure as a tree of stateful NumPy operators and runs it block by block in
the time domain -- no torch, no FFT. It is the time-domain analogue of
:func:`pyFDN.process_fdn`, but takes its structure from an arbitrary FLAMO graph
rather than a fixed FDN topology.

Typical use::

    import pyFDN
    from pyFDN import td

    model = pyFDN.dss_to_flamo(A, B, C, D, delays, fs, sos_filter=absorption)
    ir = td.process(model, impulse)

The feedback ``Recursion`` is the only non-trivial piece; see
:class:`pyFDN.td.operators.Recursion`.
"""

from __future__ import annotations

from pyFDN.td.compiler import compile_flamo_graph, process
from pyFDN.td.operators import (
    Delay,
    Gain,
    Identity,
    MatrixFIR,
    Parallel,
    Recursion,
    Series,
    SOSBank,
    TimeOperator,
    TimeVaryingMatrix,
)

__all__ = [
    "process",
    "compile_flamo_graph",
    "TimeOperator",
    "Identity",
    "Gain",
    "Delay",
    "SOSBank",
    "MatrixFIR",
    "TimeVaryingMatrix",
    "Series",
    "Parallel",
    "Recursion",
]
