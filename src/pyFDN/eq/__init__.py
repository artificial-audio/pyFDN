"""Filter sections and target-to-EQ design functions."""

from .biquads import (
    first_order_shelf_biquad,
    highshelf_biquad,
    lowshelf_biquad,
    one_pole_biquad,
    peaking_biquad,
)
from .design import (
    EQDesign,
    decay_to_first_order_shelf,
    decay_to_geq,
    decay_to_one_pole,
    gain_to_first_order_shelf,
    gain_to_one_pole,
)
from .graphic_eq import (
    BANDWIDTH_R,
    CENTER_FREQUENCIES,
    SHELVING_CROSSOVER,
    gain_to_bounded_geq,
    gain_to_geq,
    geq_design_matrix,
)
from .probe_sos import probe_sos

__all__ = [
    "BANDWIDTH_R",
    "CENTER_FREQUENCIES",
    "EQDesign",
    "SHELVING_CROSSOVER",
    "decay_to_first_order_shelf",
    "decay_to_geq",
    "decay_to_one_pole",
    "first_order_shelf_biquad",
    "gain_to_bounded_geq",
    "gain_to_first_order_shelf",
    "gain_to_geq",
    "gain_to_one_pole",
    "geq_design_matrix",
    "highshelf_biquad",
    "lowshelf_biquad",
    "one_pole_biquad",
    "peaking_biquad",
    "probe_sos",
]
