# Computing J (Second Adiabatic Invariant)

## 1. Physical Background

J is the second adiabatic invariant. It describes the bounce motion
of trapped particles in a magnetic mirror:

    J = ∮ v_∥ dl

where the integral is over one complete bounce (turning point to
turning point and back).

**Goal**: Compute J across a magnetic configuration and verify whether
J IS invariant (constant along particle drift orbits) - a key property
for good particle confinement.

## 2. Coordinate Transformation

VMEC outputs physical space coordinates, not straight field line (SFL) coordinates.
The J calculation requires Boozer coordinates.

We use `booz_xform` to transform VMEC → Boozer:

```python
import booz_xform as bx
b = bx.Booz_xform()
b.read_wout("wout_*.nc")
b.mboz = 40  # poloidal modes
b.nboz = 40  # toroidal modes
b.run()
# Access: b.bmnc_b (B in Boozer), b.write_boozmn("boozmn_*.nc")
```

## 3. Working Formula

For stellarator calculations in Boozer coordinates:

    J = [|G + ι·I| / (2π · R_major / n_fp)] × ∫ dζ (1/B) √(1 - B/B_bounce)

where:
- G, I = toroidal and poloidal currents (Boozer_G, Boozer_I from booz_xform)
- ι = rotational transform (iota)
- R_major = major radius
- n_fp = number of field periods
- B_bounce = mirror field strength for this pitch angle
- ζ = toroidal angle (Boozer coordinate)

## 4. Pitch Angle Convention

    B_bounce = B_min_global + λ_n × (B_max_global - B_min_global)

where:
- B_min_global, B_max_global = min/max |B| across ALL flux surfaces
- λ_n ∈ [0, 1]: normalized pitch parameter

**Physical interpretation:**
- **λ_n = 0**: B_bounce = B_min → deeply trapped
  - Wells are small (only where B ≈ B_min)
  - Integrand √(1 - B/B_bounce) → 0 as B → B_bounce
  - Both the integration domain and the integrand magnitude vanish, so J → 0
- **λ_n = 1**: B_bounce = B_max → trapped-passing boundary
  - Wells span nearly the entire field line
  - Large integration domain → J is large

**Passing/Prohibited regions (using GLOBAL B range):**
- If local_B_max < B_bounce: particle is PASSING (never bounces)
  - Happens when λ_n is large at inner radii where local_B_max < global_B_max
- If local_B_min ≥ B_bounce: PROHIBITED (always above mirror)
  - Can occur when λ_n is small and local_B_min > global_B_min

## 5. Magnetic Wells

A "well" is a region where B < B_bounce. Particles bounce between
the well boundaries (where B = B_bounce).

Detection algorithm:
1. Find crossings where B crosses B_bounce
2. Pair up crossings: entry → exit = one well
3. Integrate over well region only

**Multiple wells**: A field line often has multiple wells. For single
particle analysis, we only care about wells the particle actually visits
during its drift orbit. A particle stays trapped in connected wells as
it drifts - it doesn't hop arbitrarily between disconnected wells.

**Approaches:**
- **Single particle**: Compute J for the specific well the particle occupies
- **Configuration average**: Compute J across multiple wells for statistical analysis

For the v0.1 implementation, we compute J for a **single well**
(the one the particle is in). The particle's starting position determines
which well it occupies.

## 6. Calculation Steps

Step 1: Load equilibrium
  - Read wout file (VMEC output)
  - Transform to Boozer coordinates using booz_xform
  - Extract: nfp, Boozer_G, Boozer_I, iota

Step 2: Trace field line
  - At given (s, α), trace B(ζ) over several field periods
  - s = normalized flux, α = field line label

Step 3: Find GLOBAL B range
  - Trace field lines at multiple (s, α) across ALL surfaces
  - B_min_global = min of all B values
  - B_max_global = max of all B values
  - This ensures consistent λ_n interpretation everywhere

Step 4: For each λ_n (in range [0, 1]):
  - Compute B_bounce = B_min + λ_n × (B_max - B_min)
  - Check: passing if local_B_max < B_bounce
  - Check: prohibited if local_B_min ≥ B_bounce
  - Find wells and integrate

Step 5: Compute J
  - Normalization factor: |G + ι·I| / (2π · R / n_fp)
  - Integrand: (1/B) × √(1 - B/B_bounce)
  - Integrate over well (or one period, depending on method)
