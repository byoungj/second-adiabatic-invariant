"""
Basic example: compute J for a single field line.
"""

import numpy as np
from pathlib import Path
from particle_analysis import load_boozer, find_wells, compute_j_well

# Load VMEC equilibrium and transform to Boozer coordinates
examples_dir = Path(__file__).parent
wout_path = examples_dir / "configs" / "wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
boozer = load_boozer(str(wout_path))

print(f"nfp = {boozer['nfp']}")
print(f"ns = {boozer['ns']}")
print(f"Number of Boozer modes: {len(boozer['xm_b'])}")

# You need to provide B(zeta) from a field line tracer
# This example assumes you have zeta and B arrays
# zeta = ...  # toroidal angle array
# B = ...     # |B| along field line

# Global B range (should be computed across all surfaces)
B_min_global = 2.29
B_max_global = 3.31

# Pick a pitch angle
lambda_n = 0.3
B_bounce = B_min_global + lambda_n * (B_max_global - B_min_global)

# Compute normalization
s = 0.25  # flux surface
s_idx = int(s * (boozer['ns'] - 1))
iota = boozer['iota'][s_idx]
G = boozer['Boozer_G'][s_idx]
I = boozer['Boozer_I'][s_idx]

# Note: R_major would need to come from VMEC data or be specified
R_major = 5.5  # approximate for W7-X
norm_factor = abs(G + iota * I) / (2 * np.pi * R_major / boozer['nfp'])

print(f"\nAt s = {s}:")
print(f"  iota = {iota:.4f}")
print(f"  G = {G:.4f}")
print(f"  I = {I:.4f}")
print(f"  norm_factor = {norm_factor:.4f}")

# Find wells and compute J for a specific well
# wells = find_wells(zeta, B, B_bounce)
# if wells:
#     well = wells[0]  # or pick the well containing the particle
#     J = compute_j_well(zeta, B, well, B_bounce, norm_factor)
#     print(f"J = {J:.4f}")
