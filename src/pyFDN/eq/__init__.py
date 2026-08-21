"""Equalizer and absorption filter design.

Three designs of the same two filters -- an FDN's in-loop absorption and its
output EQ -- at three levels of detail: a ten-band graphic EQ, a first-order
shelf, and a one-pole. :mod:`.designs` puts them behind one interface,
:class:`~.designs.EQDesign`, which is what :mod:`pyFDN.train` builds its
trainable filters on; each design runs in numpy or in torch from one source
(see :mod:`._backend`).
"""

from .absorption_geq import absorption_geq
from .bandpass_filter import bandpass_filter
from .design_geq import design_geq, geq_design_matrix, geq_sos
from .designs import EQDesign, FirstOrderShelf, GraphicEQ, OnePole, default_design
from .first_order import (
    first_order_absorption,
    first_order_shelf_sos,
    first_order_shelving_eq,
    shelf_crossover_omega,
)
from .graphic_eq import graphic_eq
from .one_pole import one_pole_absorption, one_pole_sos
from .probe_sos import probe_sos
from .shelving_filter import shelving_filter

__all__ = [
    "EQDesign",
    "FirstOrderShelf",
    "GraphicEQ",
    "OnePole",
    "absorption_geq",
    "bandpass_filter",
    "default_design",
    "design_geq",
    "first_order_absorption",
    "first_order_shelf_sos",
    "first_order_shelving_eq",
    "geq_design_matrix",
    "geq_sos",
    "graphic_eq",
    "one_pole_absorption",
    "one_pole_sos",
    "probe_sos",
    "shelf_crossover_omega",
    "shelving_filter",
]
