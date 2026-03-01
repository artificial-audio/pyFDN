"""Auxiliary modules (utils, acoustics, allpass, flamo wrappers, etc.)."""
from .utils import skew
from .flamo import gain_module, delay_module, sos_filter_module
from .allpass import (
    poletti_allpass,
    series_allpass,
    nested_allpass,
    is_uniallpass,
    is_allpass,
    is_paraunitary,
)
from .flamo_graph import flamo_model_to_nodes, flamo_nodes_flat, draw_flamo_graph
from .poles import reduce_conjugate_pairs

__all__ = [
    "skew",
    "gain_module",
    "delay_module",
    "sos_filter_module",
    "poletti_allpass",
    "series_allpass",
    "nested_allpass",
    "is_uniallpass",
    "is_allpass",
    "is_paraunitary",
    "flamo_model_to_nodes",
    "flamo_nodes_flat",
    "draw_flamo_graph",
    "reduce_conjugate_pairs",
]
