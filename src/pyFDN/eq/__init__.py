"""Filter sections and target-to-EQ design functions."""

from .design import (
    EQ_DESIGNS,
    EQDesign,
    decay_to_first_order_shelf,
    decay_to_graphic_eq,
    decay_to_one_pole,
    decay_to_two_stage,
    gain_to_first_order_shelf,
    gain_to_one_pole,
    gain_to_two_stage,
)
from .graphic_eq import (
    BANDWIDTH_R,
    CENTER_FREQUENCIES,
    COMMAND_FREQUENCIES,
    SHELVING_CROSSOVER,
    gain_to_bounded_graphic_eq,
    gain_to_graphic_eq,
    graphic_eq_design_matrix,
)
from .probe_sos import probe_sos

__all__ = [
    "BANDWIDTH_R",
    "CENTER_FREQUENCIES",
    "COMMAND_FREQUENCIES",
    "EQDesign",
    "EQ_DESIGNS",
    "SHELVING_CROSSOVER",
    "decay_to_first_order_shelf",
    "decay_to_graphic_eq",
    "decay_to_one_pole",
    "decay_to_two_stage",
    "gain_to_bounded_graphic_eq",
    "gain_to_first_order_shelf",
    "gain_to_graphic_eq",
    "gain_to_one_pole",
    "gain_to_two_stage",
    "graphic_eq_design_matrix",
    "probe_sos",
]
