# particle_analysis

Compute J (second adiabatic invariant) for stellarator configurations.

## Installation

Requires the `particle-analysis` conda environment with `booz_xform` installed.

```bash
conda activate particle-analysis
pip install -e .
```

## Quick Start

```python
from particle_analysis import load_boozer, find_wells, compute_j_well
import numpy as np

# Load VMEC equilibrium and transform to Boozer coordinates
boozer = load_boozer("wout_example.nc")

# Set up pitch angle
B_min_global, B_max_global = 2.29, 3.31
lambda_n = 0.3
B_bounce = B_min_global + lambda_n * (B_max_global - B_min_global)

# Compute normalization factor
s_idx = 25  # flux surface index
iota = boozer['iota'][s_idx]
G = boozer['Boozer_G'][s_idx]
I = boozer['Boozer_I'][s_idx]
norm_factor = abs(G + iota * I) / (2 * np.pi * 5.5 / boozer['nfp'])

# Find wells and compute J (requires B(zeta) from field line trace)
# wells = find_wells(zeta, B, B_bounce)
# J = compute_j_well(zeta, B, wells[0], B_bounce, norm_factor)
```

## Documentation

See [docs/j_calculation.md](docs/j_calculation.md) for the physics design document.

## License

BSD-3-Clause
