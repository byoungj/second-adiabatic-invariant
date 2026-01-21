"""
Visualization module for J calculation diagnostics.

Provides plotting functions to diagnose each step of the J calculation:
1. Field line tracing
2. Well detection (B_bounce crossings)
3. Integration over wells

All functions accept optional axes for embedding in larger figures.
"""

import numpy as np


def _get_plt():
    """Lazy import of matplotlib.pyplot."""
    import matplotlib.pyplot as plt
    return plt


COLORS = {
    'B_curve': 'k',
    'B_bounce': 'r',
    'well_fill': '#cce5ff',
    'trapped_bg': '#ccffcc',
    'passing_bg': '#ffcccc',
    'entry_crossing': 'g',
    'exit_crossing': 'orange',
    'integrand': '#1f77b4',
    'well_boundary': 'gray',
}


def _compute_integrand(B, B_bounce):
    """Compute J integrand: (1/B) * sqrt(1 - B/B_bounce)."""
    return np.where(
        B < B_bounce,
        (1 / B) * np.sqrt(1 - B / B_bounce),
        0
    )


def _find_crossings(B, B_bounce):
    """Find indices where B crosses B_bounce.

    Returns
    -------
    indices : ndarray
        Indices where crossing occurs
    directions : ndarray
        +1 for upward crossing (exiting well), -1 for downward (entering)
    """
    above = B >= B_bounce
    indices = np.where(np.diff(above.astype(int)) != 0)[0]
    if len(indices) == 0:
        return indices, np.array([])
    directions = np.diff(above.astype(int))[indices]
    return indices, directions


def plot_field_line(zeta, B, B_bounce=None, wells=None, ax=None):
    """Plot magnetic field strength along a field line.

    Parameters
    ----------
    zeta : ndarray
        Toroidal angle array
    B : ndarray
        Magnetic field strength along field line
    B_bounce : float, optional
        Mirror field strength threshold. If provided, draws horizontal line.
    wells : list of tuple, optional
        List of (zeta_start, zeta_end) pairs. If provided, shades well regions.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot
    """
    plt = _get_plt()

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(zeta, B, color=COLORS['B_curve'], linewidth=1)

    if B_bounce is not None:
        ax.axhline(B_bounce, color=COLORS['B_bounce'], linestyle='--',
                   linewidth=1, label=f'B_bounce = {B_bounce:.4f}')

        indices, _ = _find_crossings(B, B_bounce)
        if len(indices) > 0:
            ax.plot(zeta[indices], B[indices], 'o', color=COLORS['B_bounce'],
                    markersize=4, label='crossings')

    if wells is not None:
        for i, (z_start, z_end) in enumerate(wells):
            ax.axvspan(z_start, z_end, alpha=0.3, color=COLORS['well_fill'],
                       label='wells' if i == 0 else None)

    ax.set_xlabel('zeta (rad)')
    ax.set_ylabel('|B| (T)')
    ax.set_title('Field Line: B(zeta)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    return ax


def plot_well_crossings(zeta, B, B_bounce, ax=None):
    """Visualize B_bounce crossings for well detection diagnosis.

    Parameters
    ----------
    zeta : ndarray
        Toroidal angle array
    B : ndarray
        Magnetic field strength along field line
    B_bounce : float
        Mirror field strength threshold
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot
    crossings : ndarray
        Indices of crossing points
    """
    plt = _get_plt()

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    above = B >= B_bounce
    for i in range(len(zeta) - 1):
        color = COLORS['passing_bg'] if above[i] else COLORS['trapped_bg']
        ax.axvspan(zeta[i], zeta[i+1], alpha=0.3, color=color, linewidth=0)

    ax.plot(zeta, B, color=COLORS['B_curve'], linewidth=1)

    ax.axhline(B_bounce, color=COLORS['B_bounce'], linestyle='--',
               linewidth=1, label=f'B_bounce = {B_bounce:.4f}')

    indices, directions = _find_crossings(B, B_bounce)

    for idx, direction in zip(indices, directions):
        if direction < 0:
            marker = 'v'
            color = COLORS['entry_crossing']
            label = 'enter well' if idx == indices[0] else None
        else:
            marker = '^'
            color = COLORS['exit_crossing']
            label = 'exit well' if idx == indices[0] else None

        ax.plot(zeta[idx], B[idx], marker, color=color, markersize=8, label=label)
        ax.annotate(str(idx), (zeta[idx], B[idx]), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=7)

    ax.set_xlabel('zeta (rad)')
    ax.set_ylabel('|B| (T)')
    ax.set_title('Well Detection: Crossings')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    return ax, indices


def plot_well_integrand(zeta, B, well, B_bounce, ax=None):
    """Visualize the J integrand for a single well.

    Parameters
    ----------
    zeta : ndarray
        Toroidal angle array (full field line)
    B : ndarray
        Magnetic field strength along field line
    well : tuple
        (zeta_start, zeta_end) defining the well boundaries
    B_bounce : float
        Mirror field strength for this pitch angle
    ax : matplotlib.axes.Axes or tuple of axes, optional
        Axes to plot on. If None, creates new figure with 2 subplots.

    Returns
    -------
    axes : tuple
        (ax_B, ax_integrand) - the two axes
    integral : float
        Computed integral value (before normalization)
    """
    plt = _get_plt()

    if ax is None:
        fig, (ax_B, ax_int) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    else:
        ax_B, ax_int = ax

    zeta_start, zeta_end = well
    mask = (zeta >= zeta_start) & (zeta <= zeta_end)
    zeta_w = zeta[mask]
    B_w = B[mask]

    margin = 0.1 * (zeta_end - zeta_start)
    view_mask = (zeta >= zeta_start - margin) & (zeta <= zeta_end + margin)
    zeta_view = zeta[view_mask]
    B_view = B[view_mask]

    ax_B.plot(zeta_view, B_view, color=COLORS['B_curve'], linewidth=1.5)
    ax_B.axhline(B_bounce, color=COLORS['B_bounce'], linestyle='--', linewidth=1)
    ax_B.axvline(zeta_start, color=COLORS['well_boundary'], linestyle='--', alpha=0.7)
    ax_B.axvline(zeta_end, color=COLORS['well_boundary'], linestyle='--', alpha=0.7)
    ax_B.axvspan(zeta_start, zeta_end, alpha=0.2, color=COLORS['well_fill'])
    ax_B.set_ylabel('|B| (T)')
    ax_B.set_title(f'Well: zeta = [{zeta_start:.3f}, {zeta_end:.3f}]')
    ax_B.grid(True, alpha=0.3)

    integrand = _compute_integrand(B_w, B_bounce)
    ax_int.plot(zeta_w, integrand, color=COLORS['integrand'], linewidth=1.5)
    ax_int.fill_between(zeta_w, integrand, alpha=0.3, color=COLORS['integrand'])
    ax_int.axvline(zeta_start, color=COLORS['well_boundary'], linestyle='--', alpha=0.7)
    ax_int.axvline(zeta_end, color=COLORS['well_boundary'], linestyle='--', alpha=0.7)

    integral = np.trapezoid(integrand, zeta_w) if len(zeta_w) >= 2 else 0.0
    ax_int.annotate(f'integral = {integral:.6f}',
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax_int.set_xlabel('zeta (rad)')
    ax_int.set_ylabel('Integrand')
    ax_int.set_title('Integrand: (1/B) * sqrt(1 - B/B_bounce)')
    ax_int.grid(True, alpha=0.3)

    plt.tight_layout()

    return (ax_B, ax_int), integral


def plot_j_diagnostic(zeta, B, B_bounce, wells, norm_factor, ax=None):
    """Comprehensive diagnostic plot showing all wells with J values.

    Parameters
    ----------
    zeta : ndarray
        Toroidal angle array
    B : ndarray
        Magnetic field strength along field line
    B_bounce : float
        Mirror field strength for this pitch angle
    wells : list of tuple
        List of (zeta_start, zeta_end) pairs defining wells
    norm_factor : float
        Normalization factor |G + iota*I| / (2*pi*R/nfp)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot
    j_values : list
        J value for each well
    """
    plt = _get_plt()

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(zeta, B, color=COLORS['B_curve'], linewidth=1)

    ax.axhline(B_bounce, color=COLORS['B_bounce'], linestyle='--',
               linewidth=1, label=f'B_bounce = {B_bounce:.4f}')

    j_values = []
    colors = [COLORS['well_fill'], '#ffe6cc']

    for i, (zeta_start, zeta_end) in enumerate(wells):
        color = colors[i % 2]
        ax.axvspan(zeta_start, zeta_end, alpha=0.4, color=color)

        mask = (zeta >= zeta_start) & (zeta <= zeta_end)
        zeta_w = zeta[mask]
        B_w = B[mask]
        if len(zeta_w) >= 2:
            integrand = _compute_integrand(B_w, B_bounce)
            J = norm_factor * np.trapezoid(integrand, zeta_w)
        else:
            J = 0.0
        j_values.append(J)

        zeta_mid = (zeta_start + zeta_end) / 2
        B_mid = B[mask].min() if np.any(mask) else B_bounce
        ax.annotate(f'{i}', (zeta_mid, B_mid - 0.02),
                    ha='center', fontsize=8, color='blue')

    stats_text = (f'Wells: {len(wells)}\n'
                  f'Mean J: {np.mean(j_values):.6f}\n'
                  f'J range: [{min(j_values):.4f}, {max(j_values):.4f}]')
    ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('zeta (rad)')
    ax.set_ylabel('|B| (T)')
    ax.set_title('J Diagnostic: All Wells')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    return ax, j_values


def diagnose_j_calculation(zeta, B, B_bounce, wells, norm_factor, well_idx=0):
    """Create comprehensive diagnostic figure for J calculation.

    Parameters
    ----------
    zeta : ndarray
        Toroidal angle array
    B : ndarray
        Magnetic field strength along field line
    B_bounce : float
        Mirror field strength for this pitch angle
    wells : list of tuple
        List of (zeta_start, zeta_end) pairs
    norm_factor : float
        Normalization factor
    well_idx : int
        Index of well to diagnose in detail (default: 0)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The diagnostic figure
    j_value : float
        Computed J for the selected well
    """
    plt = _get_plt()

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    plot_field_line(zeta, B, B_bounce=B_bounce, wells=wells, ax=ax1)
    ax1.set_title('Overview: All Wells')

    if well_idx < len(wells):
        well = wells[well_idx]
        ax1.axvspan(well[0], well[1], alpha=0.5, color='yellow',
                    label=f'Well {well_idx} (selected)')
        ax1.legend(loc='upper right')

    ax2 = fig.add_subplot(2, 2, 2)
    if well_idx < len(wells):
        well = wells[well_idx]
        zeta_start, zeta_end = well
        margin = 0.3 * (zeta_end - zeta_start)
        view_mask = (zeta >= zeta_start - margin) & (zeta <= zeta_end + margin)

        ax2.plot(zeta[view_mask], B[view_mask], color=COLORS['B_curve'], linewidth=1.5)
        ax2.axhline(B_bounce, color=COLORS['B_bounce'], linestyle='--', linewidth=1)
        ax2.axvspan(zeta_start, zeta_end, alpha=0.3, color=COLORS['well_fill'])

        indices, directions = _find_crossings(B, B_bounce)
        for idx, direction in zip(indices, directions):
            if zeta_start - margin <= zeta[idx] <= zeta_end + margin:
                marker = 'v' if direction < 0 else '^'
                color = COLORS['entry_crossing'] if direction < 0 else COLORS['exit_crossing']
                ax2.plot(zeta[idx], B[idx], marker, color=color, markersize=10)

        ax2.set_title(f'Well {well_idx}: B(zeta)')
    ax2.set_xlabel('zeta (rad)')
    ax2.set_ylabel('|B| (T)')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    if well_idx < len(wells):
        well = wells[well_idx]
        zeta_start, zeta_end = well
        margin = 0.5 * (zeta_end - zeta_start)
        view_mask = (zeta >= zeta_start - margin) & (zeta <= zeta_end + margin)
        zeta_view = zeta[view_mask]
        B_view = B[view_mask]

        above = B_view >= B_bounce
        for i in range(len(zeta_view) - 1):
            color = COLORS['passing_bg'] if above[i] else COLORS['trapped_bg']
            ax3.axvspan(zeta_view[i], zeta_view[i+1], alpha=0.4, color=color, linewidth=0)

        ax3.plot(zeta_view, B_view, color=COLORS['B_curve'], linewidth=1.5)
        ax3.axhline(B_bounce, color=COLORS['B_bounce'], linestyle='--', linewidth=1)

        ax3.set_title(f'Well {well_idx}: Crossing Detection')
    ax3.set_xlabel('zeta (rad)')
    ax3.set_ylabel('|B| (T)')
    ax3.grid(True, alpha=0.3)

    ax4_B = fig.add_subplot(2, 2, 4)
    j_value = 0.0
    if well_idx < len(wells):
        well = wells[well_idx]
        zeta_start, zeta_end = well
        mask = (zeta >= zeta_start) & (zeta <= zeta_end)
        zeta_w = zeta[mask]
        B_w = B[mask]

        if len(zeta_w) >= 2:
            integrand = _compute_integrand(B_w, B_bounce)
            integral = np.trapezoid(integrand, zeta_w)
            j_value = norm_factor * integral

            ax4_B.plot(zeta_w, integrand, color=COLORS['integrand'], linewidth=1.5)
            ax4_B.fill_between(zeta_w, integrand, alpha=0.3, color=COLORS['integrand'])
            ax4_B.annotate(f'J = {j_value:.6f}\n(integral = {integral:.6f})',
                           xy=(0.02, 0.95), xycoords='axes fraction',
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax4_B.set_title(f'Well {well_idx}: Integrand')
    ax4_B.set_xlabel('zeta (rad)')
    ax4_B.set_ylabel('Integrand')
    ax4_B.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, j_value
