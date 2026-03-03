"""
Recursive DSP Framework
========================

A modular, block-based framework for recursive DSP systems using PyTorch tensors.

This module provides:

- Core abstractions: Stage (base class) and RecursionCore (coordinator)
- Concrete stages for building FDN-like recursive systems:

  * DelayRead/DelayWrite: Circular delay buffer management
  * Biquads: IIR filter bank
  * FeedbackMix: Feedback matrix application
  * InputTap: External input injection
  * OutputTap: Output summation

Example usage:
    >>> from pyFDN.recursive import *
    >>> stages = [
    ...     DelayRead(delay_length=1024, num_lines=4),
    ...     FeedbackMix(feedback_matrix=A),
    ...     Biquads(num_lines=4),
    ...     InputTap(input_matrix=B),
    ...     DelayWrite(),
    ...     OutputTap(output_matrix=C)
    ... ]
    >>> core = RecursionCore(stages, block_size=512)
    >>> output = core.process(input_signal)
"""

from .stage import Stage
from .core import RecursionCore
from .delay_lines import DelayRead, DelayWrite, Delay
from .biquads import Biquads
from .feedback_mix import FeedbackMix
from .input_tap import InputTap
from .output_tap import OutputTap

__all__ = [
    "Stage",
    "RecursionCore",
    "DelayRead",
    "DelayWrite",
    "Delay",
    "Biquads",
    "FeedbackMix",
    "InputTap",
    "OutputTap",
]
