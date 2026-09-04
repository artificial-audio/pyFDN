"""Time-domain rendering of block-based processing graphs.

This subpackage builds a processing graph as a tree of stateful NumPy
operators and runs it block by block in the time domain -- no torch, no FFT.
Leaves (:class:`Gain`, :class:`Delay`, :class:`SOSBank`, ...) are wired
together by the connectors :class:`Series`, :class:`Parallel` and
:class:`Recursion`.

Typical use::

    import numpy as np
    from pyFDN import td

    forward = td.Series([td.Delay(delays - block_size), td.SOSBank(absorption)])
    fdn = td.Series([
        td.Gain(B),
        td.Recursion(forward, td.Gain(A), block_size=block_size),
        td.Gain(C),
    ])
    ir = fdn.process_signal(impulse)

The feedback :class:`Recursion` is the only non-trivial piece: it processes in
blocks and therefore inserts ``block_size`` samples of delay into the loop,
which the caller compensates for. See
:class:`pyFDN.td.connectors.Recursion`.
"""

from __future__ import annotations

from pyFDN.td.connectors import Parallel, Recursion, Series
from pyFDN.td.operators import (
    AbsoluteValue,
    Delay,
    Gain,
    Identity,
    MatrixConvolver,
    MatrixFIR,
    RecursionState,
    SOSBank,
    TimeOperator,
    TimeVaryingMatrix,
)
__all__ = [
    "TimeOperator",
    "Identity",
    "Gain",
    "Delay",
    "AbsoluteValue",
    "SOSBank",
    "MatrixFIR",
    "MatrixConvolver",
    "TimeVaryingMatrix",
    "RecursionState",
    "Series",
    "Parallel",
    "Recursion",
]
