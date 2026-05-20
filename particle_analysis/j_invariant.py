"""
Minimal J-invariant calculation.
Core functionality for computing J (second adiabatic invariant).
"""

import numpy as np
import booz_xform as bx
from pathlib import Path


def get_example_wout():
    """Get path to the example W7-X wout file.

    Returns
    -------
    Path
        Path to wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc
    """
    package_dir = Path(__file__).parent.parent
    return package_dir / "examples" / "configs" / "wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"


def get_example_boozmn():
    """Get path to the pre-computed W7-X boozmn file.

    Returns
    -------
    Path
        Path to boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc
    """
    package_dir = Path(__file__).parent.parent
    return package_dir / "examples" / "configs" / "boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"


def load_boozmn(boozmn_path):
    """Load pre-computed Boozer coordinates from a boozmn file.

    This is faster than load_boozer() since it skips the transform computation.

    Parameters
    ----------
    boozmn_path : str or Path
        Path to boozmn_*.nc file (output from booz_xform)

    Returns
    -------
    dict
        Equilibrium data in Boozer coordinates
    """
    from scipy.io import netcdf_file

    f = netcdf_file(str(boozmn_path), 'r')

    nfp = int(f.variables['nfp_b'].data)
    bmnc_b = f.variables['bmnc_b'][:].T.copy()
    xm_b = f.variables['ixm_b'][:].copy()
    xn_b = f.variables['ixn_b'][:].copy()
    iota = f.variables['iota_b'][1:].copy()
    Boozer_G = f.variables['bvco_b'][1:].copy()
    Boozer_I = f.variables['buco_b'][1:].copy()
    rmnc_b = f.variables['rmnc_b'][:].copy()
    ns = bmnc_b.shape[1]

    f.close()

    i00 = np.where((xm_b == 0) & (xn_b == 0))[0][0]
    R_major = float(rmnc_b[-1, i00])

    return {
        'nfp': nfp,
        'ns': ns,
        'bmnc_b': bmnc_b,
        'xm_b': xm_b,
        'xn_b': xn_b,
        'iota': iota,
        'Boozer_I': Boozer_I,
        'Boozer_G': Boozer_G,
        'R_major': R_major,
        'booz': None,
    }


def load_boozer(wout_path, mboz=40, nboz=40):
    """Load VMEC equilibrium and transform to Boozer coordinates.

    Parameters
    ----------
    wout_path : str or Path
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
    b.read_wout(str(wout_path))
    b.mboz = mboz
    b.nboz = nboz
    b.run()

    i00 = np.where((b.xm_b == 0) & (b.xn_b == 0))[0][0]
    R_major = float(b.rmnc_b[i00, -1])

    return {
        'nfp': b.nfp,
        'ns': b.ns_in,
        'bmnc_b': b.bmnc_b,
        'xm_b': b.xm_b,
        'xn_b': b.xn_b,
        'iota': b.iota,
        'Boozer_I': b.Boozer_I,
        'Boozer_G': b.Boozer_G,
        'R_major': R_major,
        'booz': b,
    }


def trace_field_line(boozer, s_idx, alpha, n_zeta=2**15, n_periods=20):
    """Trace B along a field line in Boozer coordinates.

    In Boozer coordinates, field lines are straight:
        theta_Boozer = alpha + iota * zeta_Boozer

    Parameters
    ----------
    boozer : dict
        Output from load_boozer()
    s_idx : int
        Flux surface index (0 to ns-1)
    alpha : float
        Field line label (0 to 2*pi)
    n_zeta : int
        Number of points along field line
    n_periods : int
        Total number of field periods to trace (centered on zeta=0)
        Default 20 gives range [-10, +10] periods

    Returns
    -------
    zeta : ndarray
        Toroidal angle array (Boozer coordinate)
    B : ndarray
        |B| along field line
    """
    nfp = boozer['nfp']
    iota = boozer['iota'][s_idx]
    bmnc_b = boozer['bmnc_b']
    xm_b = boozer['xm_b']
    xn_b = boozer['xn_b']

    period = 2 * np.pi / nfp
    half_range = (n_periods / 2) * period
    zeta = np.linspace(-half_range, half_range, n_zeta)

    theta = alpha + iota * zeta

    phases = xm_b[:, None] * theta[None, :] - xn_b[:, None] * zeta[None, :]
    B = np.dot(bmnc_b[:, s_idx], np.cos(phases))

    return zeta, B


def get_global_B_range(boozer, n_rho=11, n_alpha=8, n_periods=20, n_zeta=1024):
    """Compute global B_min, B_max across the entire equilibrium.

    Traces field lines across multiple flux surfaces and field line
    labels to determine the global B range. This is essential for
    consistent pitch-angle normalization across surfaces.

    Parameters
    ----------
    boozer : dict
        Equilibrium data from load_boozer() or load_boozmn()
    n_rho : int, optional
        Number of flux surfaces to sample (evenly spaced in rho space). Default 11.
    n_alpha : int, optional
        Number of field line labels (alpha values) to sample per surface. Default 8.
    n_periods : int, optional
        Number of field periods to trace per field line. Default 20.
    n_zeta : int, optional
        Number of points along each field line. Default 1024 (sufficient for min/max).

    Returns
    -------
    tuple
        (B_min_global, B_max_global)
    """
    ns = boozer['ns']

    rho_values = np.linspace(0, 1, n_rho)
    s_indices = np.round(rho_values**2 * (ns - 1)).astype(int)
    s_indices = np.unique(s_indices)

    alphas = np.linspace(0, 2*np.pi, n_alpha, endpoint=False)

    B_min = np.inf
    B_max = -np.inf
    for s_idx in s_indices:
        for alpha in alphas:
            _, B = trace_field_line(boozer, s_idx, alpha, n_zeta=n_zeta, n_periods=n_periods)
            B_min = min(B_min, B.min())
            B_max = max(B_max, B.max())

    return B_min, B_max


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
    start_idx = np.searchsorted(zeta, zeta_start)
    end_idx = np.searchsorted(zeta, zeta_end, side='right')

    zeta_w = zeta[start_idx:end_idx]
    B_w = B[start_idx:end_idx]

    if len(zeta_w) < 3:
        return 0.0

    integrand = np.where(
        B_w < B_bounce,
        (1 / B_w) * np.sqrt(np.maximum(0, 1 - B_w / B_bounce)),
        0
    )

    return norm_factor * np.trapezoid(integrand, zeta_w)
