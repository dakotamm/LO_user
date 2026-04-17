"""
Run SWIRL vortex identification on ROMS model output.

SWIRL (Canivete Cuissa & Steiner, 2022) identifies vortices in 2D velocity
fields using the Estimated Vortex Center (EVC) method.

Designed for wb1_t0_xn11ab (Whidbey Basin / Penn Cove) but configurable
for any LO ROMS output. Run on apogee where history files are stored.

Requires: pip install swirl-code

Usage examples:
    # Single date, surface velocity from history files
    python run_swirl_roms.py -gtx wb1_t0_xn11ab -0 2017.09.10 -vel surface

    # Use average files instead of history files
    python run_swirl_roms.py -gtx wb1_t0_xn11ab -0 2017.09.10 -vel surface -ftype avg

    # Date range (processes every file per day)
    python run_swirl_roms.py -gtx wb1_t0_xn11ab -0 2017.09.05 -1 2017.09.18 -vel surface -save True

    # Single date, specific file number
    python run_swirl_roms.py -gtx wb1_t0_xn11ab -0 2017.09.10 -fnum 1 -vel depth_avg

    # Date range with avg files, depth-averaged, skip plots
    python run_swirl_roms.py -gtx wb1_t0_xn11ab -0 2017.09.05 -1 2017.09.10 -ftype avg -vel depth_avg -no_plot True -save True

    # Velocity at a specific s-level
    python run_swirl_roms.py -gtx wb1_t0_xn11ab -0 2017.09.10 -vel depth_level -s_lev -1

    # Penn Cove subset
    python run_swirl_roms.py -gtx wb1_t0_xn11ab -0 2017.09.01 -1 2017.09.30 -vel surface -penn_cove True -save True

    # Custom bounding box
    python run_swirl_roms.py -gtx wb1_t0_xn11ab -0 2017.09.10 -vel surface -lon0 -122.74 -lon1 -122.56 -lat0 48.21 -lat1 48.26

    # Local run with downloaded files (no LO framework needed)
    python run_swirl_roms.py -0 2017.09.10 -vel surface -save True \
        -roms_dir /path/to/roms_output -grid_file /path/to/grid.nc \
        -out_dir /path/to/output

Outputs (with -save True):
    - <out_dir>/swirl_vortices_<ds0>_<ds1>_<ftype>_<vel>.nc  (NetCDF)
    - <out_dir>/swirl_vortices_<ds0>_<ds1>_<ftype>_<vel>.csv (CSV)
    - <out_dir>/swirl_summary_<ds0>_<ds1>_<ftype>_<vel>.csv  (per-snapshot counts)
    - Per-snapshot map plots as PNG
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.dates as mdates
from scipy import ndimage
from scipy.ndimage import gaussian_filter

import swirl

try:
    from lo_tools import Lfun, zrfun
    from lo_tools import plotting_functions as pfun
    _HAS_LO_TOOLS = True
except ImportError:
    _HAS_LO_TOOLS = False

    # Minimal fallbacks for pfun functions when lo_tools is not available
    class _pfun_fallback:
        @staticmethod
        def get_plon_plat(lon, lat):
            """Compute pcolormesh corner coordinates from cell centers."""
            plon = np.zeros((lon.shape[0] + 1, lon.shape[1] + 1))
            plat = np.zeros_like(plon)
            # Interior
            plon[1:-1, 1:-1] = 0.25 * (lon[:-1, :-1] + lon[:-1, 1:]
                                         + lon[1:, :-1] + lon[1:, 1:])
            plat[1:-1, 1:-1] = 0.25 * (lat[:-1, :-1] + lat[:-1, 1:]
                                         + lat[1:, :-1] + lat[1:, 1:])
            # Edges by extrapolation
            plon[0, :] = 2 * plon[1, :] - plon[2, :]
            plon[-1, :] = 2 * plon[-2, :] - plon[-3, :]
            plon[:, 0] = 2 * plon[:, 1] - plon[:, 2]
            plon[:, -1] = 2 * plon[:, -2] - plon[:, -3]
            plat[0, :] = 2 * plat[1, :] - plat[2, :]
            plat[-1, :] = 2 * plat[-2, :] - plat[-3, :]
            plat[:, 0] = 2 * plat[:, 1] - plat[:, 2]
            plat[:, -1] = 2 * plat[:, -2] - plat[:, -3]
            return plon, plat

        @staticmethod
        def dar(ax):
            """Set equal aspect ratio for lat/lon plots."""
            ax.set_aspect('equal', adjustable='box')

    pfun = _pfun_fallback()

# ============================================================================
# Argument parsing (LO style: single-dash short flags)
# ============================================================================

# Boolean string converter matching LO convention
def _bool_str(s):
    if s in ('True', 'true', 'T', 't', '1'):
        return True
    elif s in ('False', 'false', 'F', 'f', '0'):
        return False
    raise ValueError(f'Cannot parse {s!r} as boolean')

# Use Lfun.boolean_string if available, else local fallback
_boolean_string = Lfun.boolean_string if _HAS_LO_TOOLS else _bool_str

parser = argparse.ArgumentParser(
    description='Run SWIRL vortex identification on ROMS output.')

# --- which run (matches exfun.intro style) ---
parser.add_argument('-gtx', '--gtagex', type=str, default=None,
                    help='gridname_tag_exname, e.g. wb1_t0_xn11ab')
parser.add_argument('-ro', '--roms_out_num', type=int, default=0,
                    help='ROMS output number (0=default, 1=roms_out1, etc.)')

# --- time selection ---
parser.add_argument('-0', '--ds0', type=str, default=None,
                    help='Start date YYYY.MM.DD (or only date if -1 omitted).')
parser.add_argument('-1', '--ds1', type=str, default=None,
                    help='End date YYYY.MM.DD.')

# --- velocity options ---
parser.add_argument('-vel', '--vel_type', type=str, default='surface',
                    choices=['surface', 'depth_avg', 'depth_level'],
                    help='Type of 2D velocity field to analyze.')
parser.add_argument('-s_lev', '--s_level', type=int, default=-1,
                    help='S-level index for depth_level (0=bottom, -1=surface).')

# --- file selection ---
parser.add_argument('-ftype', '--file_type', type=str, default='his',
                    choices=['his', 'avg'],
                    help="ROMS output type: 'his' or 'avg'.")
parser.add_argument('-fnum', '--file_num', type=int, default=None,
                    help='File number (0-based). Omit to process all per day.')
parser.add_argument('-his_num', type=int, default=None,
                    help='Alias for -fnum (backward compat).')

# --- output control ---
parser.add_argument('-save', type=_boolean_string, default=False,
                    help='Save vortex properties to NetCDF and CSV.')
parser.add_argument('-out_dir', type=str, default=None,
                    help='Output directory for results and plots.')
parser.add_argument('-no_plot', type=_boolean_string, default=False,
                    help='Skip plotting.')
parser.add_argument('-verbose', type=_boolean_string, default=False,
                    help='Verbose SWIRL output.')

# --- detection method ---
parser.add_argument('-method', type=str, default='ow',
                    choices=['swirl', 'vorticity', 'ow'],
                    help="Detection method: 'ow' (Okubo-Weiss, default), "
                         "'vorticity' (relative vorticity), or "
                         "'swirl' (EVC-based).")
parser.add_argument('-vort_thresh', type=float, default=None,
                    help='Vorticity threshold (1/s). Default: 0.5*std of '
                         'water-point vorticity.')
parser.add_argument('-ow_thresh', type=float, default=None,
                    help='Okubo-Weiss threshold (1/s^2). Default: '
                         '-0.2*std of water-point OW (negative = rotation).')
parser.add_argument('-min_cells', type=int, default=9,
                    help='Min grid cells for a detected feature (default 9).')
parser.add_argument('-smooth', type=float, default=2.0,
                    help='Gaussian smoothing sigma in grid cells before '
                         'thresholding (default 2.0, 0=no smoothing).')

# --- SWIRL parameters ---
parser.add_argument('-param_file', type=str, default=None,
                    help='Path to SWIRL .param file.')

# --- spatial subset ---
parser.add_argument('-lon0', type=float, default=None)
parser.add_argument('-lon1', type=float, default=None)
parser.add_argument('-lat0', type=float, default=None)
parser.add_argument('-lat1', type=float, default=None)
parser.add_argument('-penn_cove', type=_boolean_string, default=False,
                    help='Subset to Penn Cove (~-122.74:-122.56, 48.21:48.26).')

# --- local mode (bypasses LO framework) ---
parser.add_argument('-roms_dir', type=str, default=None,
                    help='Path to dir containing fYYYY.MM.DD/ folders.')
parser.add_argument('-grid_file', type=str, default=None,
                    help='Path to ROMS grid.nc file.')

args = parser.parse_args()

# ============================================================================
# Helper functions
# ============================================================================

def get_date_list(ds0_str, ds1_str):
    """Generate list of date strings from ds0 to ds1 inclusive."""
    ds0 = datetime.strptime(ds0_str, '%Y.%m.%d')
    ds1 = datetime.strptime(ds1_str, '%Y.%m.%d')
    date_list = []
    d = ds0
    while d <= ds1:
        date_list.append(d.strftime('%Y.%m.%d'))
        d += timedelta(days=1)
    return date_list


def find_date_dir(date_str, Ldir, gtagex):
    """Search roms_out paths for a date directory. Returns Path or None."""
    for key in ['roms_out', 'roms_out2', 'roms_out1']:
        candidate = Ldir[key] / gtagex / f'f{date_str}'
        if candidate.exists():
            return candidate
    return None


def interp_u_to_rho(u_field):
    """Interpolate u-grid field to rho-grid by averaging neighbors in xi."""
    return 0.5 * (u_field[:, :-1] + u_field[:, 1:]) if u_field.ndim == 2 else \
           0.5 * (u_field[:, :, :-1] + u_field[:, :, 1:])


def interp_v_to_rho(v_field):
    """Interpolate v-grid field to rho-grid by averaging neighbors in eta."""
    return 0.5 * (v_field[:-1, :] + v_field[1:, :]) if v_field.ndim == 2 else \
           0.5 * (v_field[:, :-1, :] + v_field[:, 1:, :])


def get_velocity_2d(ds, dsg, vel_type, s_level=-1):
    """
    Extract a 2D velocity field on the rho-grid.

    Parameters
    ----------
    ds : xarray.Dataset
        Opened ROMS history file.
    dsg : xarray.Dataset
        Opened ROMS grid file.
    vel_type : str
        'surface', 'depth_avg', or 'depth_level'.
    s_level : int
        S-level index (only used for 'depth_level').

    Returns
    -------
    vx : np.ndarray (eta_rho, xi_rho)
        Zonal velocity on rho-grid [m/s].
    vy : np.ndarray (eta_rho, xi_rho)
        Meridional velocity on rho-grid [m/s].
    title_str : str
        Description string for plot titles.
    """
    mask_rho = dsg.mask_rho.values  # 1=water, 0=land

    if vel_type == 'surface':
        # Surface = top s-level of 3D u, v
        u_raw = ds.u.values[0, -1, :, :]   # (eta_u, xi_u)
        v_raw = ds.v.values[0, -1, :, :]   # (eta_v, xi_v)
        title_str = 'Surface velocity'

    elif vel_type == 'depth_avg':
        # Depth-averaged: ubar, vbar
        u_raw = ds.ubar.values[0, :, :]    # (eta_u, xi_u)
        v_raw = ds.vbar.values[0, :, :]    # (eta_v, xi_v)
        title_str = 'Depth-averaged velocity'

    elif vel_type == 'depth_level':
        u_raw = ds.u.values[0, s_level, :, :]
        v_raw = ds.v.values[0, s_level, :, :]
        title_str = f'Velocity at s-level {s_level}'

    else:
        raise ValueError(f'Unknown vel_type: {vel_type}')

    # Mask u/v on their native grids before interpolation to avoid
    # boundary contamination (averaging a valid cell with a land cell)
    u_raw = np.nan_to_num(u_raw, nan=0.0)
    v_raw = np.nan_to_num(v_raw, nan=0.0)
    mask_u = dsg.mask_u.values
    mask_v = dsg.mask_v.values
    u_raw[mask_u == 0] = 0.0
    v_raw[mask_v == 0] = 0.0

    # Interpolate from staggered u/v grids to rho-grid
    # u is on (eta_rho, xi_u) where xi_u has one fewer point than xi_rho
    # v is on (eta_v, xi_v) where eta_v has one fewer point than eta_rho
    vx = interp_u_to_rho(u_raw)  # now (eta_rho, xi_rho)
    vy = interp_v_to_rho(v_raw)  # now (eta_rho, xi_rho)

    # Trim to common shape (interpolation may leave different sizes)
    ny = min(vx.shape[0], vy.shape[0])
    nx = min(vx.shape[1], vy.shape[1])
    vx = vx[:ny, :nx]
    vy = vy[:ny, :nx]

    # Apply land mask (set land to zero to avoid spurious vortices at boundaries)
    mask = mask_rho[:ny, :nx]
    vx = np.where(mask == 1, vx, 0.0)
    vy = np.where(mask == 1, vy, 0.0)

    # Replace NaNs with zero for SWIRL
    vx = np.nan_to_num(vx, nan=0.0)
    vy = np.nan_to_num(vy, nan=0.0)

    return vx, vy, title_str


def get_grid_spacing(dsg):
    """
    Compute representative grid spacing in meters.

    ROMS stores pm = 1/DX and pn = 1/DY at rho-points.
    SWIRL expects scalar dx, dy. We use the domain-average.

    Returns
    -------
    dx_m : float
        Average grid spacing in xi-direction [m].
    dy_m : float
        Average grid spacing in eta-direction [m].
    """
    mask_rho = dsg.mask_rho.values
    pm = dsg.pm.values  # 1/DX
    pn = dsg.pn.values  # 1/DY

    # Average over water points only
    dx_all = 1.0 / pm[mask_rho == 1]
    dy_all = 1.0 / pn[mask_rho == 1]

    dx_m = float(np.mean(dx_all))
    dy_m = float(np.mean(dy_all))

    return dx_m, dy_m


def subset_to_bbox(vx, vy, dsg, lon0, lon1, lat0, lat1):
    """
    Crop velocity fields and grid dataset to a lon/lat bounding box.

    Returns
    -------
    vx_sub, vy_sub : np.ndarray
        Cropped velocity arrays.
    dsg_sub : xr.Dataset
        Cropped grid dataset (lon_rho, lat_rho, mask_rho, h, pm, pn).
    eta_slice, xi_slice : slice
        Index slices used (for reference).
    """
    lon_rho = dsg.lon_rho.values
    lat_rho = dsg.lat_rho.values

    ny, nx = vx.shape
    lon_sub = lon_rho[:ny, :nx]
    lat_sub = lat_rho[:ny, :nx]

    # Find indices inside bounding box
    mask_bbox = ((lon_sub >= lon0) & (lon_sub <= lon1) &
                 (lat_sub >= lat0) & (lat_sub <= lat1))

    if not mask_bbox.any():
        raise ValueError(
            f'No grid points found in bbox '
            f'[{lon0}, {lon1}] x [{lat0}, {lat1}].')

    eta_idx, xi_idx = np.where(mask_bbox)
    e0, e1 = int(eta_idx.min()), int(eta_idx.max()) + 1
    x0, x1 = int(xi_idx.min()), int(xi_idx.max()) + 1

    vx_sub = vx[e0:e1, x0:x1]
    vy_sub = vy[e0:e1, x0:x1]

    # Build a subsetted grid dataset with the same variable names
    dsg_sub = xr.Dataset({
        'lon_rho': (['eta_rho', 'xi_rho'], lon_rho[e0:e1, x0:x1]),
        'lat_rho': (['eta_rho', 'xi_rho'], lat_rho[e0:e1, x0:x1]),
        'mask_rho': (['eta_rho', 'xi_rho'],
                     dsg.mask_rho.values[e0:e1, x0:x1]),
        'h': (['eta_rho', 'xi_rho'], dsg.h.values[e0:e1, x0:x1]),
        'pm': (['eta_rho', 'xi_rho'], dsg.pm.values[e0:e1, x0:x1]),
        'pn': (['eta_rho', 'xi_rho'], dsg.pn.values[e0:e1, x0:x1]),
    })

    return vx_sub, vy_sub, dsg_sub, slice(e0, e1), slice(x0, x1)


def _add_ssh_panel(fig, ax, ssh_times, ssh_values, current_idx):
    """
    Draw SSH time series on the given axes with a marker at the current step.
    """
    ax.plot(ssh_times, ssh_values, 'k-', linewidth=1.0, alpha=0.7)
    ax.plot(ssh_times[current_idx], ssh_values[current_idx],
            'ro', markersize=8, zorder=5, label='current')
    ax.set_ylabel('Mean SSH [m]')
    ax.set_xlabel('Time')
    ax.set_title('Sea surface height (domain mean)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax.tick_params(axis='x', rotation=30)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_vortices_on_map(vortices_obj, vx, vy, dsg, vel_type_str, date_str,
                         out_path=None, ssh_times=None, ssh_values=None,
                         ssh_idx=None):
    """
    Plot identified vortices overlaid on the ROMS domain with velocity vectors.
    """
    lon_rho = dsg.lon_rho.values
    lat_rho = dsg.lat_rho.values
    mask_rho = dsg.mask_rho.values
    h = dsg.h.values
    h[mask_rho == 0] = np.nan

    # Trim coordinates to match velocity field shape
    ny, nx = vx.shape
    lon = lon_rho[:ny, :nx]
    lat = lat_rho[:ny, :nx]
    depth = h[:ny, :nx]

    # Compute speed
    speed = np.sqrt(vx**2 + vy**2)
    speed[mask_rho[:ny, :nx] == 0] = np.nan

    plon, plat = pfun.get_plon_plat(lon, lat)

    has_ssh = (ssh_times is not None and ssh_values is not None
               and ssh_idx is not None)
    if has_ssh:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax_ssh = fig.add_subplot(gs[1, :])
        axes = [ax0, ax1]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- Left panel: speed with velocity vectors ---
    ax = axes[0]
    cs = ax.pcolormesh(plon, plat, speed, cmap='viridis', shading='flat')
    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Speed [m/s]')

    # Quiver (subsample for readability)
    skip = max(1, ny // 30)
    ax.quiver(lon[::skip, ::skip], lat[::skip, ::skip],
              vx[::skip, ::skip], vy[::skip, ::skip],
              scale=3.0, scale_units='width', color='white', alpha=0.7,
              width=0.002)

    pfun.dar(ax)
    ax.set_title(f'{vel_type_str}\n{date_str}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # --- Right panel: vortices overlaid on bathymetry ---
    ax = axes[1]
    cs = ax.pcolormesh(plon, plat, -depth, cmap='Blues_r', shading='flat')
    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Depth [m]')

    n_vort = len(vortices_obj)
    if n_vort > 0:
        # Access vortex properties
        dx_m, dy_m = get_grid_spacing(dsg)
        centers = vortices_obj.centers  # (n_vortices, 2) in grid index coords
        radii = vortices_obj.radii      # in grid units
        orientations = vortices_obj.orientations

        for i in range(n_vort):
            # Convert grid-index center to lon/lat
            cy_idx, cx_idx = centers[i]
            cy_int = int(round(cy_idx))
            cx_int = int(round(cx_idx))

            # Clamp to valid range
            cy_int = max(0, min(cy_int, ny - 1))
            cx_int = max(0, min(cx_int, nx - 1))

            center_lon = lon[cy_int, cx_int]
            center_lat = lat[cy_int, cx_int]

            # Convert radius from grid units to approximate degrees
            radius_m = radii[i] * np.mean([dx_m, dy_m])
            radius_deg = radius_m / 111000.0  # approximate

            orient = orientations[i]
            color = 'red' if orient < 0 else 'blue'  # CW=red, CCW=blue
            label = 'CW' if orient < 0 else 'CCW'

            circle = mpatches.Circle(
                (center_lon, center_lat), radius_deg,
                fill=False, edgecolor=color, linewidth=2, linestyle='--')
            ax.add_patch(circle)
            ax.plot(center_lon, center_lat, marker='+', color=color,
                    markersize=10, markeredgewidth=2)
            ax.annotate(f'V{i} ({label})\nr={radii[i]:.1f}',
                        (center_lon, center_lat),
                        textcoords='offset points', xytext=(10, 10),
                        fontsize=8, color=color,
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', alpha=0.7))

    # Legend
    cw_patch = mpatches.Patch(edgecolor='red', facecolor='none',
                               label='Clockwise', linewidth=2)
    ccw_patch = mpatches.Patch(edgecolor='blue', facecolor='none',
                                label='Counter-clockwise', linewidth=2)
    ax.legend(handles=[cw_patch, ccw_patch], loc='lower left', fontsize=9)

    pfun.dar(ax)
    ax.set_title(f'Identified vortices: {n_vort}\n{date_str}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    if has_ssh:
        _add_ssh_panel(fig, ax_ssh, ssh_times, ssh_values, ssh_idx)

    plt.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f'Saved plot to {out_path}')

    plt.close(fig)

    return fig


def plot_swirl_diagnostics(vortices_obj, vx, vy, out_dir=None):
    """Run SWIRL's built-in diagnostic plots."""
    swirl.plot_rortex(vortices_obj, f_quiver=6, save=(out_dir is not None))
    swirl.plot_gevc_map(vortices_obj, f_quiver=6, save=(out_dir is not None))
    swirl.plot_decision(vortices_obj, save=(out_dir is not None))
    swirl.plot_vortices(vortices_obj, f_quiver=6, save=(out_dir is not None))


def compute_relative_vorticity(vx, vy, dx_m, dy_m):
    """
    Compute relative vorticity zeta = dv/dx - du/dy on the rho grid.

    Returns
    -------
    zeta : np.ndarray  (same shape as vx)
        Relative vorticity [1/s].
    """
    # np.gradient returns derivative along axis; axis=1 is xi (x), axis=0 is eta (y)
    dvdx = np.gradient(vy, dx_m, axis=1)
    dudy = np.gradient(vx, dy_m, axis=0)
    zeta = dvdx - dudy
    return zeta


def detect_vorticity_features(vx, vy, dsg, dx_m, dy_m, vort_thresh=None,
                              min_cells=9):
    """
    Identify recirculation features using relative vorticity thresholding.

    This is a standard oceanographic approach suitable for large, shallow-water
    features where SWIRL's EVC method may not detect complex eigenvalues.

    Parameters
    ----------
    vx, vy : np.ndarray
        Velocity components on rho grid.
    dsg : xr.Dataset
        Grid dataset (must have mask_rho).
    dx_m, dy_m : float
        Grid spacing in meters.
    vort_thresh : float or None
        Absolute vorticity threshold [1/s]. If None, uses 0.5 * std
        of water-point vorticity.
    min_cells : int
        Minimum number of grid cells for a feature.

    Returns
    -------
    features : list of dict
        Each dict has: center_eta, center_xi, center_lon, center_lat,
        radius_grid, radius_m, orientation (+1 CCW, -1 CW),
        mean_vorticity, max_vorticity, n_cells.
    zeta : np.ndarray
        The vorticity field.
    """
    ny, nx = vx.shape
    mask_rho = dsg.mask_rho.values[:ny, :nx]
    lon_rho = dsg.lon_rho.values[:ny, :nx]
    lat_rho = dsg.lat_rho.values[:ny, :nx]

    zeta = compute_relative_vorticity(vx, vy, dx_m, dy_m)
    # Zero out land
    zeta[mask_rho == 0] = 0.0

    # Determine threshold
    water_zeta = zeta[mask_rho == 1]
    zeta_std = float(np.std(water_zeta))
    zeta_mean = float(np.mean(np.abs(water_zeta)))
    if vort_thresh is None:
        vort_thresh = 0.5 * zeta_std

    print(f'  Vorticity stats: mean|zeta|={zeta_mean:.2e}, '
          f'std={zeta_std:.2e}, thresh={vort_thresh:.2e} 1/s')

    features = []

    # Detect both CW (zeta < -thresh) and CCW (zeta > +thresh)
    for sign, orient, label in [(-1, -1.0, 'CW'), (1, 1.0, 'CCW')]:
        if sign < 0:
            binary = (zeta < -vort_thresh) & (mask_rho == 1)
        else:
            binary = (zeta > vort_thresh) & (mask_rho == 1)

        labeled, n_labels = ndimage.label(binary)

        for lab in range(1, n_labels + 1):
            cells = np.where(labeled == lab)
            n_cells = len(cells[0])
            if n_cells < min_cells:
                continue

            eta_idx = cells[0]
            xi_idx = cells[1]

            # Centroid (weighted by |vorticity|)
            weights = np.abs(zeta[eta_idx, xi_idx])
            w_sum = weights.sum()
            if w_sum == 0:
                continue
            center_eta = float(np.average(eta_idx, weights=weights))
            center_xi = float(np.average(xi_idx, weights=weights))

            # Convert to lon/lat
            ce_int = max(0, min(int(round(center_eta)), ny - 1))
            cx_int = max(0, min(int(round(center_xi)), nx - 1))
            center_lon = float(lon_rho[ce_int, cx_int])
            center_lat = float(lat_rho[ce_int, cx_int])

            # Effective radius
            radius_grid = float(np.sqrt(n_cells / np.pi))
            radius_m = radius_grid * np.mean([dx_m, dy_m])

            mean_vort = float(np.mean(zeta[eta_idx, xi_idx]))
            max_vort = float(zeta[eta_idx, xi_idx][np.argmax(np.abs(
                zeta[eta_idx, xi_idx]))])

            features.append(dict(
                center_eta=center_eta,
                center_xi=center_xi,
                center_lon=center_lon,
                center_lat=center_lat,
                radius_grid=radius_grid,
                radius_m=radius_m,
                orientation=orient,
                rotation=label,
                mean_vorticity=mean_vort,
                max_vorticity=max_vort,
                n_cells=n_cells,
            ))

    # Sort by radius (largest first)
    features.sort(key=lambda f: f['radius_m'], reverse=True)

    return features, zeta


def plot_vorticity_features(features, zeta, vx, vy, dsg, vel_type_str,
                            date_str, dx_m, dy_m, out_path=None,
                            ssh_times=None, ssh_values=None, ssh_idx=None):
    """
    Plot vorticity-detected features overlaid on velocity and vorticity fields.
    """
    ny, nx = vx.shape
    lon_rho = dsg.lon_rho.values[:ny, :nx]
    lat_rho = dsg.lat_rho.values[:ny, :nx]
    mask_rho = dsg.mask_rho.values[:ny, :nx]
    h = dsg.h.values[:ny, :nx].copy()
    h[mask_rho == 0] = np.nan

    speed = np.sqrt(vx**2 + vy**2)
    speed[mask_rho == 0] = np.nan
    zeta_plot = zeta.copy()
    zeta_plot[mask_rho == 0] = np.nan

    plon, plat = pfun.get_plon_plat(lon_rho, lat_rho)

    has_ssh = (ssh_times is not None and ssh_values is not None
               and ssh_idx is not None)
    if has_ssh:
        fig = plt.figure(figsize=(22, 11))
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1])
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        ax_ssh = fig.add_subplot(gs[1, :])
    else:
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # --- Left: speed + quiver ---
    ax = axes[0]
    cs = ax.pcolormesh(plon, plat, speed, cmap='viridis', shading='flat')
    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Speed [m/s]')
    skip = max(1, ny // 30)
    ax.quiver(lon_rho[::skip, ::skip], lat_rho[::skip, ::skip],
              vx[::skip, ::skip], vy[::skip, ::skip],
              scale=3.0, scale_units='width', color='white', alpha=0.7,
              width=0.002)
    pfun.dar(ax)
    ax.set_title(f'{vel_type_str}\n{date_str}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # --- Center: vorticity field ---
    ax = axes[1]
    vmax = np.nanpercentile(np.abs(zeta_plot[mask_rho == 1]), 98)
    cs = ax.pcolormesh(plon, plat, zeta_plot, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, shading='flat')
    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Vorticity [1/s]')
    pfun.dar(ax)
    ax.set_title(f'Relative vorticity\n{date_str}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # --- Right: detected features on bathymetry ---
    ax = axes[2]
    cs = ax.pcolormesh(plon, plat, -h, cmap='Blues_r', shading='flat')
    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Depth [m]')

    for i, feat in enumerate(features):
        color = 'red' if feat['orientation'] < 0 else 'blue'
        label = feat['rotation']
        radius_deg = feat['radius_m'] / 111000.0

        circle = mpatches.Circle(
            (feat['center_lon'], feat['center_lat']), radius_deg,
            fill=False, edgecolor=color, linewidth=2, linestyle='--')
        ax.add_patch(circle)
        ax.plot(feat['center_lon'], feat['center_lat'], marker='+',
                color=color, markersize=10, markeredgewidth=2)
        ax.annotate(
            f'V{i} ({label})\n'
            f'r={feat["radius_m"]:.0f}m\n'
            f'ζ={feat["mean_vorticity"]:.1e}',
            (feat['center_lon'], feat['center_lat']),
            textcoords='offset points', xytext=(10, 10),
            fontsize=7, color=color,
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', alpha=0.7))

    cw_patch = mpatches.Patch(edgecolor='red', facecolor='none',
                               label='Clockwise (ζ<0)', linewidth=2)
    ccw_patch = mpatches.Patch(edgecolor='blue', facecolor='none',
                                label='Counter-CW (ζ>0)', linewidth=2)
    ax.legend(handles=[cw_patch, ccw_patch], loc='lower left', fontsize=8)
    pfun.dar(ax)
    ax.set_title(f'Detected features: {len(features)}\n{date_str}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    if has_ssh:
        _add_ssh_panel(fig, ax_ssh, ssh_times, ssh_values, ssh_idx)

    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f'Saved plot to {out_path}')
    plt.close(fig)
    return fig


def compute_okubo_weiss(vx, vy, dx_m, dy_m):
    """
    Compute the Okubo-Weiss parameter: W = s_n^2 + s_s^2 - zeta^2.

    W < 0 means rotation dominates strain (vortex core).
    W > 0 means strain dominates (filaments, jets).

    Returns
    -------
    OW : np.ndarray
        Okubo-Weiss parameter [1/s^2].
    zeta : np.ndarray
        Relative vorticity [1/s].
    """
    dudx = np.gradient(vx, dx_m, axis=1)
    dudy = np.gradient(vx, dy_m, axis=0)
    dvdx = np.gradient(vy, dx_m, axis=1)
    dvdy = np.gradient(vy, dy_m, axis=0)

    # Relative vorticity
    zeta = dvdx - dudy
    # Normal strain
    s_n = dudx - dvdy
    # Shear strain
    s_s = dvdx + dudy

    OW = s_n**2 + s_s**2 - zeta**2
    return OW, zeta


def detect_ow_features(vx, vy, dsg, dx_m, dy_m, ow_thresh=None,
                       min_cells=9, smooth_sigma=2.0):
    """
    Identify recirculation features using the Okubo-Weiss parameter.

    OW < 0 indicates rotation-dominated flow (vortex cores). This is the
    standard oceanographic approach for mesoscale/submesoscale eddies and
    works well for large, shallow-water recirculations where SWIRL fails.

    Parameters
    ----------
    vx, vy : np.ndarray
        Velocity components on rho grid.
    dsg : xr.Dataset
        Grid dataset (must have mask_rho).
    dx_m, dy_m : float
        Grid spacing in meters.
    ow_thresh : float or None
        OW threshold [1/s^2]. Features have OW < ow_thresh.
        If None, uses -0.2 * std(OW) over water points.
    min_cells : int
        Minimum number of grid cells for a feature.
    smooth_sigma : float
        Gaussian smoothing sigma in grid cells (0 = no smoothing).

    Returns
    -------
    features : list of dict
    OW : np.ndarray
        The Okubo-Weiss field.
    zeta : np.ndarray
        The vorticity field.
    """
    ny, nx = vx.shape
    mask_rho = dsg.mask_rho.values[:ny, :nx]
    lon_rho = dsg.lon_rho.values[:ny, :nx]
    lat_rho = dsg.lat_rho.values[:ny, :nx]

    OW, zeta = compute_okubo_weiss(vx, vy, dx_m, dy_m)

    # Zero out land
    OW[mask_rho == 0] = 0.0
    zeta[mask_rho == 0] = 0.0

    # Smooth to connect fragmented features
    if smooth_sigma > 0:
        OW_smooth = gaussian_filter(OW, sigma=smooth_sigma)
        zeta_smooth = gaussian_filter(zeta, sigma=smooth_sigma)
        # Re-zero land after smoothing
        OW_smooth[mask_rho == 0] = 0.0
        zeta_smooth[mask_rho == 0] = 0.0
    else:
        OW_smooth = OW
        zeta_smooth = zeta

    # Threshold
    water_OW = OW_smooth[mask_rho == 1]
    ow_std = float(np.std(water_OW))
    ow_mean = float(np.mean(water_OW))
    if ow_thresh is None:
        ow_thresh = -0.2 * ow_std

    print(f'  Okubo-Weiss stats: mean={ow_mean:.2e}, '
          f'std={ow_std:.2e}, thresh={ow_thresh:.2e} 1/s^2')
    print(f'  Fraction with OW < thresh: '
          f'{np.sum(OW_smooth[mask_rho == 1] < ow_thresh) / np.sum(mask_rho == 1):.1%}')

    # Rotation-dominated regions: OW < threshold (negative)
    binary = (OW_smooth < ow_thresh) & (mask_rho == 1)

    # Morphological closing to merge nearby fragments
    struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    binary = ndimage.binary_closing(binary, structure=struct, iterations=2)
    binary = binary & (mask_rho == 1)  # re-mask land after closing

    labeled, n_labels = ndimage.label(binary, structure=struct)

    features = []
    for lab in range(1, n_labels + 1):
        cells = np.where(labeled == lab)
        n_cells = len(cells[0])
        if n_cells < min_cells:
            continue

        eta_idx = cells[0]
        xi_idx = cells[1]

        # Determine rotation from mean vorticity within feature
        mean_zeta = float(np.mean(zeta_smooth[eta_idx, xi_idx]))
        orient = -1.0 if mean_zeta < 0 else 1.0
        rotation = 'CW' if orient < 0 else 'CCW'

        # Centroid weighted by |OW| (stronger rotation = more weight)
        weights = np.abs(OW_smooth[eta_idx, xi_idx])
        w_sum = weights.sum()
        if w_sum == 0:
            continue
        center_eta = float(np.average(eta_idx, weights=weights))
        center_xi = float(np.average(xi_idx, weights=weights))

        ce_int = max(0, min(int(round(center_eta)), ny - 1))
        cx_int = max(0, min(int(round(center_xi)), nx - 1))
        center_lon = float(lon_rho[ce_int, cx_int])
        center_lat = float(lat_rho[ce_int, cx_int])

        radius_grid = float(np.sqrt(n_cells / np.pi))
        radius_m = radius_grid * np.mean([dx_m, dy_m])

        mean_ow = float(np.mean(OW_smooth[eta_idx, xi_idx]))
        max_vort = float(zeta_smooth[eta_idx, xi_idx][
            np.argmax(np.abs(zeta_smooth[eta_idx, xi_idx]))])

        features.append(dict(
            center_eta=center_eta,
            center_xi=center_xi,
            center_lon=center_lon,
            center_lat=center_lat,
            radius_grid=radius_grid,
            radius_m=radius_m,
            orientation=orient,
            rotation=rotation,
            mean_vorticity=mean_zeta,
            max_vorticity=max_vort,
            mean_ow=mean_ow,
            n_cells=n_cells,
        ))

    features.sort(key=lambda f: f['radius_m'], reverse=True)
    return features, OW, zeta


def plot_ow_features(features, OW, zeta, vx, vy, dsg, vel_type_str,
                     date_str, dx_m, dy_m, out_path=None,
                     ssh_times=None, ssh_values=None, ssh_idx=None):
    """
    Plot Okubo-Weiss detected features: velocity, OW field, and features.
    """
    ny, nx = vx.shape
    lon_rho = dsg.lon_rho.values[:ny, :nx]
    lat_rho = dsg.lat_rho.values[:ny, :nx]
    mask_rho = dsg.mask_rho.values[:ny, :nx]
    h = dsg.h.values[:ny, :nx].copy()
    h[mask_rho == 0] = np.nan

    speed = np.sqrt(vx**2 + vy**2)
    speed[mask_rho == 0] = np.nan
    OW_plot = OW.copy()
    OW_plot[mask_rho == 0] = np.nan

    plon, plat = pfun.get_plon_plat(lon_rho, lat_rho)

    has_ssh = (ssh_times is not None and ssh_values is not None
               and ssh_idx is not None)
    if has_ssh:
        fig = plt.figure(figsize=(22, 11))
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1])
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        ax_ssh = fig.add_subplot(gs[1, :])
    else:
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # --- Left: speed + quiver ---
    ax = axes[0]
    cs = ax.pcolormesh(plon, plat, speed, cmap='viridis', shading='flat')
    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Speed [m/s]')
    skip = max(1, ny // 30)
    ax.quiver(lon_rho[::skip, ::skip], lat_rho[::skip, ::skip],
              vx[::skip, ::skip], vy[::skip, ::skip],
              scale=3.0, scale_units='width', color='white', alpha=0.7,
              width=0.002)
    pfun.dar(ax)
    ax.set_title(f'{vel_type_str}\n{date_str}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # --- Center: Okubo-Weiss field ---
    ax = axes[1]
    ow_water = OW_plot[mask_rho == 1]
    vmax = np.nanpercentile(np.abs(ow_water), 95)
    cs = ax.pcolormesh(plon, plat, OW_plot, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, shading='flat')
    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Okubo-Weiss [1/s²]')
    pfun.dar(ax)
    ax.set_title(f'Okubo-Weiss (blue=rotation)\n{date_str}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # --- Right: detected features on bathymetry ---
    ax = axes[2]
    cs = ax.pcolormesh(plon, plat, -h, cmap='Blues_r', shading='flat')
    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Depth [m]')

    # Overlay quiver for context
    ax.quiver(lon_rho[::skip, ::skip], lat_rho[::skip, ::skip],
              vx[::skip, ::skip], vy[::skip, ::skip],
              scale=3.0, scale_units='width', color='gray', alpha=0.4,
              width=0.002)

    for i, feat in enumerate(features):
        color = 'red' if feat['orientation'] < 0 else 'blue'
        label = feat['rotation']
        radius_deg = feat['radius_m'] / 111000.0

        circle = mpatches.Circle(
            (feat['center_lon'], feat['center_lat']), radius_deg,
            fill=False, edgecolor=color, linewidth=2.5, linestyle='--')
        ax.add_patch(circle)
        ax.plot(feat['center_lon'], feat['center_lat'], marker='+',
                color=color, markersize=12, markeredgewidth=2)
        ax.annotate(
            f'V{i} ({label})\n'
            f'r={feat["radius_m"]:.0f}m\n'
            f'{feat["n_cells"]} cells',
            (feat['center_lon'], feat['center_lat']),
            textcoords='offset points', xytext=(12, 12),
            fontsize=7, color=color,
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', alpha=0.8))

    cw_patch = mpatches.Patch(edgecolor='red', facecolor='none',
                               label='CW (\u03b6<0)', linewidth=2)
    ccw_patch = mpatches.Patch(edgecolor='blue', facecolor='none',
                                label='CCW (\u03b6>0)', linewidth=2)
    ax.legend(handles=[cw_patch, ccw_patch], loc='lower left', fontsize=8)
    pfun.dar(ax)
    ax.set_title(f'OW features: {len(features)}\n{date_str}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    if has_ssh:
        _add_ssh_panel(fig, ax_ssh, ssh_times, ssh_values, ssh_idx)

    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f'Saved plot to {out_path}')
    plt.close(fig)
    return fig


def extract_vorticity_records(features, date_str, file_type, file_num,
                              vel_type, s_level):
    """Convert vorticity/OW features to records matching SWIRL output format."""
    records = []
    for i, feat in enumerate(features):
        records.append(dict(
            date=date_str,
            file_type=file_type,
            file_num=file_num,
            vel_type=vel_type,
            s_level=s_level if vel_type == 'depth_level' else np.nan,
            vortex_id=i,
            center_eta_idx=feat['center_eta'],
            center_xi_idx=feat['center_xi'],
            center_lon=feat['center_lon'],
            center_lat=feat['center_lat'],
            radius_grid=feat['radius_grid'],
            radius_m=feat['radius_m'],
            orientation=feat['orientation'],
            rotation=feat['rotation'],
            mean_vorticity=feat.get('mean_vorticity', np.nan),
            max_vorticity=feat.get('max_vorticity', np.nan),
            n_cells=feat.get('n_cells', np.nan),
        ))
    return records


def extract_vortex_records(vortices_obj, dsg, vx_shape, date_str,
                           file_type, file_num, vel_type, s_level,
                           dx_m, dy_m):
    """
    Convert SWIRL vortex output to a list of dicts for DataFrame construction.

    Returns list of dicts, one per vortex (empty list if none found).
    """
    n_vort = len(vortices_obj)
    if n_vort == 0:
        return []

    lon_rho = dsg.lon_rho.values
    lat_rho = dsg.lat_rho.values
    ny, nx = vx_shape

    centers = vortices_obj.centers
    radii = vortices_obj.radii
    orientations = vortices_obj.orientations

    records = []
    for i in range(n_vort):
        cy_idx, cx_idx = centers[i]
        cy_int = max(0, min(int(round(cy_idx)), ny - 1))
        cx_int = max(0, min(int(round(cx_idx)), nx - 1))

        center_lon = float(lon_rho[cy_int, cx_int])
        center_lat = float(lat_rho[cy_int, cx_int])
        radius_m = float(radii[i]) * np.mean([dx_m, dy_m])
        orient = float(orientations[i])

        records.append(dict(
            date=date_str,
            file_type=file_type,
            file_num=file_num,
            vel_type=vel_type,
            s_level=s_level if vel_type == 'depth_level' else np.nan,
            vortex_id=i,
            center_eta_idx=float(cy_idx),
            center_xi_idx=float(cx_idx),
            center_lon=center_lon,
            center_lat=center_lat,
            radius_grid=float(radii[i]),
            radius_m=radius_m,
            orientation=orient,
            rotation='CW' if orient < 0 else 'CCW',
        ))

    return records


def plot_vortex_summary(df_vort, dsg, out_path=None):
    """
    Plot all identified vortex locations on a map, colored by date.
    """
    if len(df_vort) == 0:
        print('  No vortices to plot in summary.')
        return None

    lon_rho = dsg.lon_rho.values
    lat_rho = dsg.lat_rho.values
    mask_rho = dsg.mask_rho.values
    h = dsg.h.values.copy()
    h[mask_rho == 0] = np.nan

    plon, plat = pfun.get_plon_plat(lon_rho, lat_rho)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- Left: vortex locations colored by date ---
    ax = axes[0]
    ax.pcolormesh(plon, plat, -h, cmap='Blues_r', shading='flat', alpha=0.4)

    dates_unique = df_vort['date'].unique()
    cmap = plt.cm.plasma
    colors = {d: cmap(i / max(1, len(dates_unique) - 1))
              for i, d in enumerate(dates_unique)}

    for _, row in df_vort.iterrows():
        c = colors[row['date']]
        marker = 'v' if row['rotation'] == 'CW' else '^'
        ax.plot(row['center_lon'], row['center_lat'],
                marker=marker, color=c, markersize=6, alpha=0.7)

    # Colorbar proxy for dates
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(0, max(1, len(dates_unique) - 1)))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    tick_idx = np.linspace(0, len(dates_unique) - 1,
                           min(len(dates_unique), 8)).astype(int)
    cb.set_ticks(tick_idx)
    cb.set_ticklabels([dates_unique[i] for i in tick_idx])
    cb.set_label('Date')

    # Markers legend
    cw_marker = plt.Line2D([], [], marker='v', color='gray', linestyle='None',
                           markersize=8, label='CW')
    ccw_marker = plt.Line2D([], [], marker='^', color='gray', linestyle='None',
                            markersize=8, label='CCW')
    ax.legend(handles=[cw_marker, ccw_marker], loc='lower left', fontsize=9)

    pfun.dar(ax)
    ax.set_title(f'All vortex locations ({len(df_vort)} total)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # --- Right: radius vs date ---
    ax = axes[1]
    for _, row in df_vort.iterrows():
        c = 'red' if row['rotation'] == 'CW' else 'blue'
        ax.plot(row['date'], row['radius_m'], 'o', color=c,
                markersize=5, alpha=0.6)

    ax.set_ylabel('Radius [m]')
    ax.set_xlabel('Date')
    ax.set_title('Vortex radius over time')
    ax.tick_params(axis='x', rotation=45)

    cw_patch = mpatches.Patch(color='red', label='CW')
    ccw_patch = mpatches.Patch(color='blue', label='CCW')
    ax.legend(handles=[cw_patch, ccw_patch], fontsize=9)

    plt.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f'  Saved summary plot to {out_path}')

    plt.close(fig)
    return fig


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':

    # --- Resolve date range ---
    if args.ds0 is not None and args.ds1 is not None:
        ds0 = args.ds0
        ds1 = args.ds1
    elif args.ds0 is not None:
        # -0 only: single date
        ds0 = args.ds0
        ds1 = args.ds0
    else:
        ds0 = '2017.09.10'
        ds1 = '2017.09.10'

    date_list = get_date_list(ds0, ds1)

    # Resolve file_num (-his_num is alias for -fnum)
    file_num = args.file_num if args.file_num is not None else args.his_num

    file_type = args.file_type  # 'his' or 'avg'
    file_glob = f'ocean_{file_type}_*.nc'
    file_label = 'his' if file_type == 'his' else 'avg'

    # --- Setup: always use Lfun.Lstart() for paths when available ---
    local_mode = (args.roms_dir is not None or args.grid_file is not None)

    if not _HAS_LO_TOOLS and not local_mode:
        raise ImportError(
            'lo_tools is not installed. Use -roms_dir and -grid_file '
            'for local runs without the LO framework.')

    # Parse -gtx into gridname, tag, ex_name (LO convention)
    if args.gtagex is None:
        gtx_str = 'wb1_t0_xn11ab'  # default
    else:
        gtx_str = args.gtagex
    gridname, tag, ex_name = gtx_str.split('_')

    # Get Ldir for machine-aware paths (LOo, grid, roms_out, etc.)
    if _HAS_LO_TOOLS:
        Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
        gtagex = Ldir['gtagex']
        # Set roms_out based on -ro
        if args.roms_out_num > 0:
            Ldir['roms_out'] = Ldir['roms_out' + str(args.roms_out_num)]
    else:
        Ldir = None
        gtagex = gtx_str

    # Resolve ROMS file paths: local overrides take precedence
    if local_mode:
        if args.roms_dir is None or args.grid_file is None:
            raise ValueError(
                '-roms_dir and -grid_file must both be provided '
                'for local runs.')
        roms_dir = Path(args.roms_dir)
        grid_file = Path(args.grid_file)
        if not grid_file.exists():
            raise FileNotFoundError(f'Grid file not found: {grid_file}')
        print(f'Local file override: roms_dir={roms_dir}')
        print(f'                     grid_file={grid_file}')
    else:
        roms_dir = None
        grid_file = Ldir['grid'] / 'grid.nc'

    print(f'Grid: {gtagex}')
    print(f'Date range: {ds0} to {ds1} ({len(date_list)} days)')
    print(f'File type: ocean_{file_type}')
    print(f'Velocity type: {args.vel_type}')

    # --- Load grid (once) ---
    dsg = xr.open_dataset(grid_file)
    print(f'Grid shape: eta_rho={dsg.dims["eta_rho"]}, '
          f'xi_rho={dsg.dims["xi_rho"]}')

    dx_m, dy_m = get_grid_spacing(dsg)
    print(f'Average grid spacing: dx={dx_m:.1f} m, dy={dy_m:.1f} m')

    # --- Resolve spatial bounding box ---
    if args.penn_cove:
        bbox = (-122.74, -122.56, 48.21, 48.26)
        bbox_label = 'Penn Cove'
    elif all(v is not None for v in [args.lon0, args.lon1,
                                      args.lat0, args.lat1]):
        bbox = (args.lon0, args.lon1, args.lat0, args.lat1)
        bbox_label = (f'{args.lon0:.2f}-{args.lon1:.2f}E, '
                      f'{args.lat0:.2f}-{args.lat1:.2f}N')
    elif any(v is not None for v in [args.lon0, args.lon1,
                                      args.lat0, args.lat1]):
        raise ValueError('All four of -lon0/-lon1/-lat0/-lat1 '
                         'must be provided together.')
    else:
        bbox = None
        bbox_label = 'full domain'

    print(f'Spatial subset: {bbox_label}')

    # --- Output directory (always machine-aware via Ldir when possible) ---
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif Ldir is not None:
        out_dir = Ldir['LOo'] / 'swirl' / gtagex
    else:
        raise ValueError(
            'Cannot determine output directory: lo_tools not available '
            'and -out_dir not specified.')
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Collectors ---
    summary_records = []   # one row per snapshot (date/file)
    vortex_records = []    # one row per vortex
    dsg_plot_last = dsg    # track subsetted grid for summary plot

    # --- Pre-scan: collect SSH time series across all snapshots ---
    # ROMS stores free-surface height as 'zeta' (eta_rho, xi_rho)
    print('\nPre-scanning SSH across all snapshots...')
    ssh_times = []   # datetime objects
    ssh_values = []  # mean SSH [m] over bbox water points
    ssh_file_keys = []  # (date_str, fi) tuples for matching

    # Compute bbox index slices once for SSH extraction
    if bbox is not None:
        _lon = dsg.lon_rho.values
        _lat = dsg.lat_rho.values
        _mask_bb = ((_lon >= bbox[0]) & (_lon <= bbox[1]) &
                    (_lat >= bbox[2]) & (_lat <= bbox[3]))
        _eidx, _xidx = np.where(_mask_bb)
        _e0, _e1 = int(_eidx.min()), int(_eidx.max()) + 1
        _x0, _x1 = int(_xidx.min()), int(_xidx.max()) + 1
        _ssh_eslice = slice(_e0, _e1)
        _ssh_xslice = slice(_x0, _x1)
        _ssh_mask = dsg.mask_rho.values[_e0:_e1, _x0:_x1]
    else:
        _ssh_eslice = slice(None)
        _ssh_xslice = slice(None)
        _ssh_mask = dsg.mask_rho.values

    for _ds in date_list:
        if local_mode:
            _dd = roms_dir / f'f{_ds}'
        else:
            _dd = find_date_dir(_ds, Ldir, gtagex)
        if _dd is None or not _dd.exists():
            continue
        _ncs = sorted(_dd.glob(file_glob))
        if file_num is not None:
            _indices = [file_num] if file_num < len(_ncs) else []
        else:
            _indices = list(range(len(_ncs)))
        for _fi in _indices:
            try:
                with xr.open_dataset(_ncs[_fi]) as _dsf:
                    # Extract time
                    _t = pd.Timestamp(_dsf.ocean_time.values[0])
                    # Extract SSH over bbox
                    _ssh = _dsf.zeta.values[0, _ssh_eslice, _ssh_xslice]
                    _ssh = np.where(_ssh_mask == 1, _ssh, np.nan)
                    _mean_ssh = float(np.nanmean(_ssh))
                    ssh_times.append(_t)
                    ssh_values.append(_mean_ssh)
                    ssh_file_keys.append((_ds, _fi))
            except Exception as _e:
                print(f'  SSH scan skip {_ds}/{_fi}: {_e}')

    ssh_times = np.array(ssh_times)
    ssh_values = np.array(ssh_values)
    print(f'  Collected SSH for {len(ssh_times)} snapshots')

    # Counter for tracking current snapshot index into SSH arrays
    _ssh_counter = 0

    # --- Loop over dates ---
    for date_str in date_list:

        if local_mode:
            date_dir = roms_dir / f'f{date_str}'
            if not date_dir.exists():
                print(f'\n*** Skipping {date_str}: {date_dir} not found.')
                continue
        else:
            date_dir = find_date_dir(date_str, Ldir, gtagex)
            if date_dir is None:
                print(f'\n*** Skipping {date_str}: no date directory found.')
                continue

        nc_files = sorted(date_dir.glob(file_glob))
        if len(nc_files) == 0:
            print(f'\n*** Skipping {date_str}: no ocean_{file_type} files found.')
            continue

        # Determine which files to process
        if file_num is not None:
            if file_num >= len(nc_files):
                print(f'\n*** Skipping {date_str}: file_num={file_num} '
                      f'but only {len(nc_files)} files.')
                continue
            file_indices = [file_num]
        else:
            file_indices = list(range(len(nc_files)))

        for fi in file_indices:
            nc_fn = nc_files[fi]
            print(f'\n===== {date_str} {file_label}#{fi:04d} =====')
            print(f'  File: {nc_fn}')

            # --- Load ROMS file ---
            ds = xr.open_dataset(nc_fn)

            # --- Extract 2D velocity field ---
            vx, vy, vel_title = get_velocity_2d(
                ds, dsg, args.vel_type, args.s_level)

            # --- Spatial subset (if requested) ---
            if bbox is not None:
                vx, vy, dsg_plot, _, _ = subset_to_bbox(
                    vx, vy, dsg, *bbox)
                dx_m, dy_m = get_grid_spacing(dsg_plot)
                vel_title += f' [{bbox_label}]'
                dsg_plot_last = dsg_plot
            else:
                dsg_plot = dsg

            max_speed = float(np.sqrt(vx**2 + vy**2).max())
            print(f'  Velocity field shape: {vx.shape}')
            print(f'  Max speed: {max_speed:.4f} m/s')

            # --- Detect features ---
            label_str = f'{date_str}, {file_label}#{fi:04d}'

            if args.method == 'swirl':
                # --- SWIRL EVC-based detection ---
                swirl_kwargs = dict(
                    v=[vx, vy],
                    grid_dx=[dx_m, dy_m],
                    verbose=args.verbose,
                )
                if args.param_file is not None:
                    swirl_kwargs['param_file'] = args.param_file

                vortices = swirl.Identification(**swirl_kwargs)
                vortices.run()

                # Diagnostic output to understand detection
                try:
                    rortex_max = max(
                        float(np.max(np.abs(r))) for r in vortices.rortex)
                    n_gevc = vortices.gevc_map.shape[1] \
                        if vortices.gevc_map.ndim == 2 else 0
                    print(f'  SWIRL diagnostics:')
                    print(f'    Max |rortex|: {rortex_max:.2e}')
                    print(f'    G-EVC points: {n_gevc}')
                except Exception as e:
                    print(f'  SWIRL diagnostics unavailable: {e}')

                n_vortices = len(vortices)
                print(f'  Identified {n_vortices} vortices (SWIRL)')

                if n_vortices > 0:
                    print(f'    Radii (grid units): {vortices.radii}')
                    print(f'    Centers (grid idx): {vortices.centers}')
                    print(f'    Orientations:       {vortices.orientations}')

                # Collect summary
                summary_records.append(dict(
                    date=date_str,
                    file_type=file_type,
                    file_num=fi,
                    n_vortices=n_vortices,
                    max_speed=max_speed,
                    method='swirl',
                ))

                # Extract per-vortex records
                vortex_records.extend(extract_vortex_records(
                    vortices, dsg_plot, vx.shape, date_str,
                    file_type, fi, args.vel_type, args.s_level,
                    dx_m, dy_m))

                # Plot
                if not args.no_plot:
                    if len(date_list) == 1 and len(file_indices) == 1:
                        print('  SWIRL diagnostic plots...')
                        plot_swirl_diagnostics(vortices, vx, vy, out_dir=None)

                    plot_name = (f'swirl_map_{date_str}_{args.vel_type}'
                                 f'_{file_label}{fi:04d}.png')
                    plot_path = out_dir / plot_name
                    plot_vortices_on_map(vortices, vx, vy, dsg_plot,
                                         vel_title, label_str,
                                         out_path=plot_path,
                                         ssh_times=ssh_times,
                                         ssh_values=ssh_values,
                                         ssh_idx=_ssh_counter)

            elif args.method == 'ow':
                # --- Okubo-Weiss detection ---
                features, OW, zeta = detect_ow_features(
                    vx, vy, dsg_plot, dx_m, dy_m,
                    ow_thresh=args.ow_thresh,
                    min_cells=args.min_cells,
                    smooth_sigma=args.smooth)

                n_vortices = len(features)
                print(f'  Identified {n_vortices} features (Okubo-Weiss)')
                for i, f in enumerate(features):
                    print(f'    V{i}: {f["rotation"]} '
                          f'r={f["radius_m"]:.0f}m '
                          f'({f["n_cells"]} cells) '
                          f'OW={f.get("mean_ow", 0):.2e} '
                          f'@ ({f["center_lon"]:.3f}, '
                          f'{f["center_lat"]:.3f})')

                summary_records.append(dict(
                    date=date_str,
                    file_type=file_type,
                    file_num=fi,
                    n_vortices=n_vortices,
                    max_speed=max_speed,
                    method='ow',
                ))

                vortex_records.extend(extract_vorticity_records(
                    features, date_str, file_type, fi,
                    args.vel_type, args.s_level))

                if not args.no_plot:
                    plot_name = (f'ow_map_{date_str}_{args.vel_type}'
                                 f'_{file_label}{fi:04d}.png')
                    plot_path = out_dir / plot_name
                    plot_ow_features(
                        features, OW, zeta, vx, vy, dsg_plot,
                        vel_title, label_str, dx_m, dy_m,
                        out_path=plot_path,
                        ssh_times=ssh_times,
                        ssh_values=ssh_values,
                        ssh_idx=_ssh_counter)

            else:
                # --- Vorticity-based detection ---
                features, zeta = detect_vorticity_features(
                    vx, vy, dsg_plot, dx_m, dy_m,
                    vort_thresh=args.vort_thresh,
                    min_cells=args.min_cells)

                n_vortices = len(features)
                print(f'  Identified {n_vortices} features (vorticity)')
                for i, f in enumerate(features):
                    print(f'    V{i}: {f["rotation"]} '
                          f'r={f["radius_m"]:.0f}m '
                          f'({f["n_cells"]} cells) '
                          f'ζ_mean={f["mean_vorticity"]:.2e} '
                          f'@ ({f["center_lon"]:.3f}, '
                          f'{f["center_lat"]:.3f})')

                summary_records.append(dict(
                    date=date_str,
                    file_type=file_type,
                    file_num=fi,
                    n_vortices=n_vortices,
                    max_speed=max_speed,
                    method='vorticity',
                ))

                vortex_records.extend(extract_vorticity_records(
                    features, date_str, file_type, fi,
                    args.vel_type, args.s_level))

                if not args.no_plot:
                    plot_name = (f'vort_map_{date_str}_{args.vel_type}'
                                 f'_{file_label}{fi:04d}.png')
                    plot_path = out_dir / plot_name
                    plot_vorticity_features(
                        features, zeta, vx, vy, dsg_plot,
                        vel_title, label_str, dx_m, dy_m,
                        out_path=plot_path,
                        ssh_times=ssh_times,
                        ssh_values=ssh_values,
                        ssh_idx=_ssh_counter)

            _ssh_counter += 1
            ds.close()

    # --- Build vortex DataFrame ---
    df_vortices = pd.DataFrame(vortex_records)
    print(f'\nTotal vortices identified: {len(df_vortices)} '
          f'across {len(summary_records)} snapshots')

    if len(df_vortices) > 0:
        print(df_vortices.to_string(index=False))

    # --- Print per-snapshot summary ---
    if len(summary_records) > 1:
        print('\n' + '=' * 60)
        print('PER-SNAPSHOT SUMMARY')
        print('=' * 60)
        df_summary = pd.DataFrame(summary_records)
        print(df_summary.to_string(index=False))

    # --- Save outputs ---
    if args.save:
        base = f'{args.method}_vortices_{ds0}_{ds1}_{file_type}_{args.vel_type}'

        # Save vortex DataFrame as CSV
        csv_path = out_dir / f'{base}.csv'
        df_vortices.to_csv(csv_path, index=False)
        print(f'Saved vortex CSV to {csv_path}')

        # Save vortex DataFrame as NetCDF
        if len(df_vortices) > 0:
            ds_out = xr.Dataset.from_dataframe(
                df_vortices.reset_index(drop=True))
            # Encode string columns as attributes or object → str
            nc_path = out_dir / f'{base}.nc'
            ds_out.to_netcdf(nc_path)
            print(f'Saved vortex NetCDF to {nc_path}')
        else:
            print('No vortices found; NetCDF not written.')

        # Save per-snapshot summary CSV
        summary_csv = out_dir / f'{args.method}_summary_{ds0}_{ds1}_{file_type}_{args.vel_type}.csv'
        pd.DataFrame(summary_records).to_csv(summary_csv, index=False)
        print(f'Saved snapshot summary to {summary_csv}')

    # --- Summary plot (all vortices across snapshots) ---
    if not args.no_plot and len(df_vortices) > 0:
        summary_plot_path = None
        if args.save:
            summary_plot_path = out_dir / (
                f'{args.method}_vortex_summary_{ds0}_{ds1}'
                f'_{file_type}_{args.vel_type}.png')
        plot_vortex_summary(df_vortices, dsg_plot_last, out_path=summary_plot_path)

    dsg.close()
    print('\nDone.')
