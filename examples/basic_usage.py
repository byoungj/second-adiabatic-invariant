"""
Basic example: compute J for field line wells.

Demonstrates the full workflow:
1. Load equilibrium (VMEC -> Boozer)
2. Trace field line
3. Find bounce points / wells
4. Compute J for each well
"""

import numpy as np
from pathlib import Path
from particle_analysis import (
    load_boozer,
    trace_field_line,
    find_wells,
    compute_j_well,
)

# Load VMEC equilibrium and transform to Boozer coordinates
examples_dir = Path(__file__).parent
wout_path = examples_dir / "configs" / "wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
boozer = load_boozer(str(wout_path))

print(f"nfp = {boozer['nfp']}")
print(f"ns = {boozer['ns']}")
print(f"Number of Boozer modes: {len(boozer['xm_b'])}")

# Trace field line over 20 periods centered on zeta=0
s_idx = 100  # mid-radius flux surface
alpha = 0.0  # field line label
zeta, B = trace_field_line(boozer, s_idx=s_idx, alpha=alpha, n_periods=20)

print(f"\nField line traced:")
print(f"  s_idx = {s_idx}")
print(f"  alpha = {alpha}")
print(f"  zeta range: [{zeta[0]:.3f}, {zeta[-1]:.3f}]")
print(f"  {len(zeta)} points")

# Get B range on this field line
B_min, B_max = B.min(), B.max()
print(f"\nB range on field line:")
print(f"  B_min = {B_min:.4f}")
print(f"  B_max = {B_max:.4f}")

# Set pitch angle (lambda_n = 0 -> deeply trapped, lambda_n = 1 -> passing boundary)
lambda_n = 0.5
B_bounce = B_min + lambda_n * (B_max - B_min)
print(f"\nPitch angle:")
print(f"  lambda_n = {lambda_n}")
print(f"  B_bounce = {B_bounce:.4f}")

# Find all wells (bounce points)
wells = find_wells(zeta, B, B_bounce)
print(f"\nFound {len(wells)} wells")

# Compute normalization factor
iota = boozer['iota'][s_idx]
G = boozer['Boozer_G'][s_idx]
I = boozer['Boozer_I'][s_idx]
R_major = 5.5  # approximate for W7-X
norm_factor = abs(G + iota * I) / (2 * np.pi * R_major / boozer['nfp'])

print(f"\nNormalization:")
print(f"  iota = {iota:.4f}")
print(f"  G = {G:.4f}")
print(f"  I = {I:.4f}")
print(f"  norm_factor = {norm_factor:.4f}")

# Compute J for each well
print(f"\nJ values for first 10 wells:")
for i, well in enumerate(wells[:10]):
    J = compute_j_well(zeta, B, well, B_bounce, norm_factor)
    width = well[1] - well[0]
    print(f"  Well {i:2d}: zeta=[{well[0]:7.3f}, {well[1]:7.3f}], width={width:.3f}, J={J:.6f}")
