"""
particle_analysis: Compute J (second adiabatic invariant) for stellarator configurations.

v0.1 - Core J calculation with Boozer coordinates via booz_xform.
v0.2 - Added diagnostic visualization module.
"""

from .j_invariant import (
    load_boozer,
    trace_field_line,
    get_global_B_range,
    find_wells,
    compute_j_well,
)

__version__ = "0.2.0"
__all__ = [
    "load_boozer",
    "trace_field_line",
    "get_global_B_range",
    "find_wells",
    "compute_j_well",
]

try:
    from .visualization import (
        plot_field_line,
        plot_well_crossings,
        plot_well_integrand,
        plot_j_diagnostic,
        diagnose_j_calculation,
    )
    __all__.extend([
        "plot_field_line",
        "plot_well_crossings",
        "plot_well_integrand",
        "plot_j_diagnostic",
        "diagnose_j_calculation",
    ])
except ImportError:
    pass
