"""
particle_analysis: Compute J (second adiabatic invariant) for stellarator configurations.

v0.1 - Core J calculation with Boozer coordinates via booz_xform.
"""

from .j_invariant import (
    load_boozer,
    get_global_B_range,
    find_wells,
    compute_j_well,
)

__version__ = "0.1.0"
__all__ = [
    "load_boozer",
    "get_global_B_range",
    "find_wells",
    "compute_j_well",
]
