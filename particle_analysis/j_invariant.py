"""
Minimal J-invariant calculation.
Core functionality for computing J (second adiabatic invariant).
"""

import numpy as np
import booz_xform as bx


def load_boozer(wout_path, mboz=40, nboz=40):
    """Load VMEC equilibrium and transform to Boozer coordinates.

    Parameters
    ----------
    wout_path : str
        Path to VMEC wout_*.nc file
    mboz : int
        Number of poloidal Fourier modes for Boozer transform
    nboz : int
        Number of toroidal Fourier modes for Boozer transform

    Returns
    -------
    dict
        Equilibrium data in Boozer coordinates
    """
    b = bx.Booz_xform()
    b.read_wout(wout_path)
    b.mboz = mboz
    b.nboz = nboz
    b.run()

    return {
        'nfp': b.nfp,
        'ns': b.ns_in,
        'bmnc_b': b.bmnc_b,
        'xm_b': b.xm_b,
        'xn_b': b.xn_b,
        'iota': b.iota,
        'Boozer_I': b.Boozer_I,
        'Boozer_G': b.Boozer_G,
        'booz': b,
    }


def get_global_B_range(B_arrays):
    """Get global B_min, B_max across all traced field lines.

    Parameters
    ----------
    B_arrays : list of ndarray
        List of B arrays from multiple (s, alpha) traces

    Returns
    -------
    tuple
        (B_min_global, B_max_global)
    """
    all_B = np.concatenate(B_arrays)
    return all_B.min(), all_B.max()


def find_wells(zeta, B, B_bounce):
    """Find magnetic wells where B < B_bounce.

    Parameters
    ----------
    zeta : ndarray
        Toroidal angle array
    B : ndarray
        Magnetic field strength along field line
    B_bounce : float
        Mirror field strength for this pitch angle

    Returns
    -------
    list of tuple
        List of (zeta_start, zeta_end) pairs defining wells
    """
    above = B >= B_bounce
    crossings = np.where(np.diff(above.astype(int)) != 0)[0]

    wells = []
    i = 1 if not above[0] else 0
    while i + 1 < len(crossings):
        wells.append((zeta[crossings[i]], zeta[crossings[i + 1]]))
        i += 2
    return wells


def compute_j_well(zeta, B, well, B_bounce, norm_factor):
    """Compute J for a single magnetic well.

    Parameters
    ----------
    zeta : ndarray
        Toroidal angle array
    B : ndarray
        Magnetic field strength along field line
    well : tuple
        (zeta_start, zeta_end) defining the well boundaries
    B_bounce : float
        Mirror field strength for this pitch angle
    norm_factor : float
        Normalization factor |G + iota*I| / (2*pi*R/nfp)

    Returns
    -------
    float
        J value for this well
    """
    zeta_start, zeta_end = well
    mask = (zeta >= zeta_start) & (zeta <= zeta_end)

    zeta_w = zeta[mask]
    B_w = B[mask]

    if len(zeta_w) < 3:
        return 0.0

    integrand = np.where(
        B_w < B_bounce,
        (1 / B_w) * np.sqrt(1 - B_w / B_bounce),
        0
    )

    return norm_factor * np.trapezoid(integrand, zeta_w)
