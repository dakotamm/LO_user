"""
Visualize tidal phase detection results and phase-averaged fields.

Reads output from compute_tide_phases.py and phase_avg_fields.py and
produces:
  1. zeta time series colored by flood/ebb with spring/neap shading
  2. 2x2 phase-averaged scalar maps (depth-avg, surface, bottom) for
     salt, temp, oxygen (mg/L), u, v
  3. 2x2 phase-averaged velocity quivers at depth-avg, surface, bottom

Designed to run locally (after copying results from apogee).

Usage
-----
    python tide_phase_analysis.py -gtx wb1_r0_xn11ab -label penn_cove \
        -0 2017.01.01 -1 2017.01.31 -file_type his
"""

import argparse
import sys
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun
from cmocean import cm as cmo
from cmcrameri import cm as cmc

import tide_phase_fun as tpf


# -----------------------------------------------------------------------
# Per-variable plot styling
# -----------------------------------------------------------------------
# (cmap, diverging?)  diverging=True -> symmetric vlims around 0
VAR_STYLE = {
    'u':      (cmo.balance, True),
    'v':      (cmo.balance, True),
    'w':      (cmo.balance, True),
    'zeta':   (cmo.balance, True),
    'salt':   (cmo.haline,  False),
    'temp':   (cmo.thermal, False),
    'oxygen': (cmo.oxy,     False),
    'NO3':    (cmo.matter,  False),
    'NH4':    (cmo.matter,  False),
    'phytoplankton': (cmo.algae,  False),
    'zooplankton':   (cmo.algae,  False),
    'TIC':    (cmo.dense,   False),
    'alkalinity': (cmo.dense, False),
}

# Per-variable display unit conversion (factor, label).
# Default: factor=1.0, label=variable name.
VAR_UNITS = {
    'oxygen':  (32.0 / 1000.0, 'DO [mg L$^{-1}$]'),  # mmol/m^3 -> mg/L
    'salt':    (1.0,           'salinity [g kg$^{-1}$]'),
    'temp':    (1.0,           'temperature [°C]'),
    'u':       (1.0,           'u [m s$^{-1}$]'),
    'v':       (1.0,           'v [m s$^{-1}$]'),
}


def _resolve_units(vn):
    """Strip _top/_bot, look up (factor, label)."""
    base = vn.replace('_top', '').replace('_bot', '')
    factor, label = VAR_UNITS.get(base, (1.0, base))
    return factor, label, base


# Map zoom: Penn Cove proper plus the entrance
ZOOM_BOUNDS = {
    'penn_cove': (-122.755, -122.60, 48.205, 48.255),
}

# Sub-region(s) to EXCLUDE when computing color limits (still plotted, just
# not used for vlim percentiles). Each entry is a list of
# (lon0, lon1, lat0, lat1) boxes.
COLOR_EXCLUDE = {
    'penn_cove': [(-122.77, -122.66, 48.20, 48.27)],
}

# Sub-region(s) to MASK OUT (set to NaN) entirely so they don't appear
# on the plot. Each entry is a list of (lon0, lon1, lat0, lat1) boxes.
PLOT_EXCLUDE = {
    'penn_cove': [(-122.755, -122.73, 48.205, 48.215)],
}


def _color_mask(lon, lat, label, bounds):
    """Boolean mask of cells used for vlim stats: inside zoom AND not in
    any COLOR_EXCLUDE box.
    """
    in_box = np.ones_like(lon, dtype=bool)
    if bounds is not None:
        in_box = ((lon >= bounds[0]) & (lon <= bounds[1])
                  & (lat >= bounds[2]) & (lat <= bounds[3]))
    for ex in COLOR_EXCLUDE.get(label, []):
        in_ex = ((lon >= ex[0]) & (lon <= ex[1])
                 & (lat >= ex[2]) & (lat <= ex[3]))
        in_box &= ~in_ex
    return in_box


# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description='Plot tidal phase analysis results.')
    parser.add_argument('-gtx', '--gtagex', type=str, required=True)
    parser.add_argument('-0', '--ds0', type=str, required=True)
    parser.add_argument('-1', '--ds1', type=str, required=True)
    parser.add_argument('-label', type=str, required=True)
    parser.add_argument('-file_type', type=str, default='his',
                        choices=['avg', 'his'],
                        help='Which phase_avg_fields output to plot')
    parser.add_argument('-out_dir', type=str, default=None,
                        help='Output directory for plots (default: LOo/tide_phase/plots)')
    parser.add_argument('-by_season', type=Lfun.boolean_string,
                        default=False,
                        help='If True, look for per-season phase_avg files '
                             '(<label>_<season>_<phase>.nc) and produce '
                             'phases x seasons comparison plots.')

    args = parser.parse_args()

    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
    for k, v in vars(args).items():
        if k not in Ldir:
            Ldir[k] = v

    if Ldir['out_dir'] is None:
        Ldir['out_dir'] = (Ldir['LOo'] / 'tide_phase'
                           / Ldir['gtagex'] / 'plots')
    else:
        Ldir['out_dir'] = Path(Ldir['out_dir'])
    Lfun.make_dir(Ldir['out_dir'])

    return Ldir


# -----------------------------------------------------------------------
# Plot 1: Zeta time series with phase shading
# -----------------------------------------------------------------------
def plot_zeta_phases(ds_phase, Ldir):
    """Time series of zeta colored by flood/ebb with spring/neap shading."""
    time = pd.DatetimeIndex(ds_phase['time'].values)
    zeta = ds_phase['zeta'].values
    is_flood = ds_phase['is_flood'].values.astype(bool)
    is_spring = ds_phase['is_spring'].values.astype(bool)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Spring/neap background shading
    for i in range(len(time) - 1):
        color = '#ffe0e0' if is_spring[i] else '#e0e0ff'
        ax.axvspan(time[i], time[i + 1], alpha=0.3, color=color,
                   linewidth=0)

    # Zeta colored by flood/ebb
    for i in range(len(time) - 1):
        c = 'tab:red' if is_flood[i] else 'tab:blue'
        ax.plot(time[i:i + 2], zeta[i:i + 2], color=c, linewidth=0.8)

    # UTide prediction if available
    if 'zeta_pred' in ds_phase:
        ax.plot(time, ds_phase['zeta_pred'].values, 'k--', linewidth=0.5,
                alpha=0.5, label='UTide prediction')
        ax.legend(loc='upper right', fontsize=8)

    ax.set_ylabel('ζ [m]')
    ax.set_title(f'Sea Surface Height — {Ldir["label"]} '
                 f'({Ldir["ds0"]} to {Ldir["ds1"]})')
    ax.legend(handles=[
        Patch(facecolor='tab:red', label='Flood'),
        Patch(facecolor='tab:blue', label='Ebb'),
        Patch(facecolor='#ffe0e0', alpha=0.5, label='Spring'),
        Patch(facecolor='#e0e0ff', alpha=0.5, label='Neap'),
    ], loc='upper left', fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_fn = Ldir['out_dir'] / (f'zeta_phases_{Ldir["label"]}_'
                                 f'{Ldir["ds0"]}_{Ldir["ds1"]}.png')
    fig.savefig(out_fn, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_fn}')
    plt.close(fig)


# -----------------------------------------------------------------------
# Plot 2: 2x2 phase-averaged fields
# -----------------------------------------------------------------------
def plot_phase_avg_fields(Ldir, vn='u', cmap=None, vlims=None):
    """2x2 panel of depth-averaged field for each tidal phase.

    Uses VAR_STYLE for default colormap and ZOOM_BOUNDS[label] for zoom.
    Vlims are shared across all four panels (per-variable, percentile-based).
    """
    phase_names = ['spring_flood', 'spring_ebb', 'neap_flood', 'neap_ebb']
    titles = ['Spring Flood', 'Spring Ebb', 'Neap Flood', 'Neap Ebb']

    style_cmap, diverging = VAR_STYLE.get(vn.replace('_top', '').replace('_bot', ''),
                                          (cmo.haline, False))
    if cmap is None:
        cmap = style_cmap
    factor, unit_label, base_vn = _resolve_units(vn)

    avg_dir = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
               / ('phase_avg_' + Ldir['ds0'] + '_' + Ldir['ds1']
                  + '_' + Ldir['file_type']))

    # First pass: read all four panels so we can pick shared vlims
    panels = []
    for pn in phase_names:
        fn = avg_dir / (Ldir['label'] + '_' + pn + '.nc')
        if not fn.is_file():
            panels.append(None)
            continue
        ds = xr.open_dataset(fn)
        if vn not in ds:
            ds.close()
            panels.append(None)
            continue
        fld = ds[vn].values * factor
        n_ts = ds.attrs.get('n_timesteps', '?')
        if base_vn == 'u':
            lon_key, lat_key = 'lon_u', 'lat_u'
        elif base_vn == 'v':
            lon_key, lat_key = 'lon_v', 'lat_v'
        else:
            lon_key, lat_key = 'lon_rho', 'lat_rho'
        if lon_key in ds and lat_key in ds:
            plon, plat = pfun.get_plon_plat(ds[lon_key].values,
                                            ds[lat_key].values)
        else:
            plon = plat = None
        ds.close()
        panels.append({'fld': fld, 'plon': plon, 'plat': plat,
                       'n': n_ts, 'pn': pn})

    # Determine zoom bounds first so we can compute vlims only on visible cells
    bounds = ZOOM_BOUNDS.get(Ldir['label'])

    if vlims is None:
        # Restrict vlim stats to cells inside the zoom window AND outside
        # any COLOR_EXCLUDE box for this label.
        sample_vals = []
        for p in panels:
            if p is None:
                continue
            v = p['fld']
            if p['plon'] is not None:
                lon_c = 0.25 * (p['plon'][:-1, :-1] + p['plon'][1:, :-1] +
                                p['plon'][:-1, 1:] + p['plon'][1:, 1:])
                lat_c = 0.25 * (p['plat'][:-1, :-1] + p['plat'][1:, :-1] +
                                p['plat'][:-1, 1:] + p['plat'][1:, 1:])
                mask = _color_mask(lon_c, lat_c, Ldir['label'], bounds)
                v = v[mask]
            sample_vals.append(v[np.isfinite(v)].ravel())
        if sample_vals and any(s.size for s in sample_vals):
            allv = np.concatenate(sample_vals)
            if diverging:
                vmax = float(np.nanpercentile(np.abs(allv), 98))
                vmin = -vmax
            else:
                vmin = float(np.nanpercentile(allv, 2))
                vmax = float(np.nanpercentile(allv, 98))
        else:
            vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = vlims

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    last_cs = None
    for idx, (panel, title) in enumerate(zip(panels, titles)):
        ax = axes.flat[idx]
        if panel is None:
            ax.set_title(f'{title}\n(no data)')
            ax.set_axis_off()
            continue
        if panel['plon'] is not None:
            fld_plot = panel['fld'].astype(float).copy()
            if bounds is not None:
                lon_c = 0.25 * (panel['plon'][:-1, :-1]
                                + panel['plon'][1:, :-1]
                                + panel['plon'][:-1, 1:]
                                + panel['plon'][1:, 1:])
                lat_c = 0.25 * (panel['plat'][:-1, :-1]
                                + panel['plat'][1:, :-1]
                                + panel['plat'][:-1, 1:]
                                + panel['plat'][1:, 1:])
                outside = ~((lon_c >= bounds[0]) & (lon_c <= bounds[1])
                            & (lat_c >= bounds[2]) & (lat_c <= bounds[3]))
                for ex in PLOT_EXCLUDE.get(Ldir['label'], []):
                    outside |= ((lon_c >= ex[0]) & (lon_c <= ex[1])
                                & (lat_c >= ex[2]) & (lat_c <= ex[3]))
                fld_plot[outside] = np.nan
            cs = ax.pcolormesh(panel['plon'], panel['plat'], fld_plot,
                               cmap=cmap, vmin=vmin, vmax=vmax)
            pfun.add_coast(ax)
            pfun.dar(ax)
            if bounds is not None:
                ax.set_xlim(bounds[0], bounds[1])
                ax.set_ylim(bounds[2], bounds[3])
            else:
                ax.set_xlim(np.nanmin(panel['plon']),
                            np.nanmax(panel['plon']))
                ax.set_ylim(np.nanmin(panel['plat']),
                            np.nanmax(panel['plat']))
        else:
            cs = ax.pcolormesh(panel['fld'], cmap=cmap,
                               vmin=vmin, vmax=vmax)
        ax.set_title(f'{title}  (n={panel["n"]})')
        last_cs = cs

    if last_cs is not None:
        cbar = fig.colorbar(last_cs, ax=axes.ravel().tolist(),
                            shrink=0.85, fraction=0.04, pad=0.02)
        cbar.set_label(unit_label)

    # Pretty title: indicate vertical layer if applicable
    if vn.endswith('_top'):
        layer = 'Surface'
    elif vn.endswith('_bot'):
        layer = 'Bottom'
    else:
        layer = 'Depth-Averaged'
    fig.suptitle(f'{layer} {base_vn} — {Ldir["label"]} '
                 f'({Ldir["file_type"]}) '
                 f'({Ldir["ds0"]} to {Ldir["ds1"]})', fontsize=14)

    out_fn = Ldir['out_dir'] / (f'phase_avg_{vn}_{Ldir["label"]}_{Ldir["file_type"]}_'
                                 f'{Ldir["ds0"]}_{Ldir["ds1"]}.png')
    fig.savefig(out_fn, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_fn}')
    plt.close(fig)


# -----------------------------------------------------------------------
# Plot 3: 2x2 quiver of (u, v) for each tidal phase
# -----------------------------------------------------------------------
def plot_phase_avg_quiver(Ldir, layer='', skip=2, scale=None):
    """Quiver of velocity for each tidal phase.

    layer : '' (depth-avg), '_top', or '_bot'
    skip  : grid stride for arrow density
    """
    phase_names = ['spring_flood', 'spring_ebb', 'neap_flood', 'neap_ebb']
    titles = ['Spring Flood', 'Spring Ebb', 'Neap Flood', 'Neap Ebb']

    avg_dir = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
               / ('phase_avg_' + Ldir['ds0'] + '_' + Ldir['ds1']
                  + '_' + Ldir['file_type']))

    u_vn = 'u' + layer
    v_vn = 'v' + layer
    layer_name = {'': 'Depth-Averaged',
                  '_top': 'Surface',
                  '_bot': 'Bottom'}[layer]

    bounds = ZOOM_BOUNDS.get(Ldir['label'])

    panels = []
    speed_vals = []
    for pn in phase_names:
        fn = avg_dir / (Ldir['label'] + '_' + pn + '.nc')
        if not fn.is_file():
            panels.append(None)
            continue
        ds = xr.open_dataset(fn)
        if u_vn not in ds or v_vn not in ds:
            ds.close()
            panels.append(None)
            continue

        # Regrid u and v to rho points so they live on the same grid
        u = ds[u_vn].values
        v = ds[v_vn].values
        # u: (eta_rho, xi_rho-1) -> average to xi_rho centers (drop ends)
        u_rho = 0.5 * (u[:, :-1] + u[:, 1:])      # (eta_rho, xi_rho-2)
        v_rho = 0.5 * (v[:-1, :] + v[1:, :])      # (eta_rho-2, xi_rho)
        # Trim to common interior
        u_int = u_rho[1:-1, :]                    # (eta_rho-2, xi_rho-2)
        v_int = v_rho[:, 1:-1]                    # (eta_rho-2, xi_rho-2)
        lon_rho = ds['lon_rho'].values[1:-1, 1:-1]
        lat_rho = ds['lat_rho'].values[1:-1, 1:-1]
        n_ts = ds.attrs.get('n_timesteps', '?')
        ds.close()

        spd = np.sqrt(u_int**2 + v_int**2)
        # Restrict speed stats to the zoom box minus exclusions
        mask = _color_mask(lon_rho, lat_rho, Ldir['label'], bounds)
        speed_vals.append(spd[mask & np.isfinite(spd)].ravel())

        panels.append({'u': u_int, 'v': v_int, 'spd': spd,
                       'lon': lon_rho, 'lat': lat_rho,
                       'n': n_ts})

    if not any(p is not None for p in panels):
        print(f'  no u/v data for layer="{layer}"; skipping quiver.')
        return

    if speed_vals:
        all_spd = np.concatenate(speed_vals)
        smax = float(np.nanpercentile(all_spd, 98)) if all_spd.size else 0.5
    else:
        smax = 0.5
    if scale is None:
        # Quiver `scale`: data units per axes width. Smaller -> longer arrows.
        scale = max(smax * 12.0, 0.5)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    last_cs = None
    for idx, (p, title) in enumerate(zip(panels, titles)):
        ax = axes.flat[idx]
        if p is None:
            ax.set_title(f'{title}\n(no data)')
            ax.set_axis_off()
            continue
        spd_plot = p['spd'].astype(float).copy()
        u_plot = p['u'].astype(float).copy()
        v_plot = p['v'].astype(float).copy()
        if bounds is not None:
            outside = ~((p['lon'] >= bounds[0]) & (p['lon'] <= bounds[1])
                        & (p['lat'] >= bounds[2]) & (p['lat'] <= bounds[3]))
            for ex in PLOT_EXCLUDE.get(Ldir['label'], []):
                outside |= ((p['lon'] >= ex[0]) & (p['lon'] <= ex[1])
                            & (p['lat'] >= ex[2]) & (p['lat'] <= ex[3]))
            spd_plot[outside] = np.nan
            u_plot[outside] = np.nan
            v_plot[outside] = np.nan
        cs = ax.pcolormesh(p['lon'], p['lat'], spd_plot,
                           cmap=cmo.speed, vmin=0, vmax=smax,
                           shading='auto')
        ax.quiver(p['lon'][::skip, ::skip], p['lat'][::skip, ::skip],
                  u_plot[::skip, ::skip], v_plot[::skip, ::skip],
                  scale=scale, color='k', width=0.003)
        pfun.add_coast(ax)
        pfun.dar(ax)
        if bounds is not None:
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
        ax.set_title(f'{title}  (n={p["n"]})')
        last_cs = cs

    if last_cs is not None:
        cbar = fig.colorbar(last_cs, ax=axes.ravel().tolist(),
                            shrink=0.85, fraction=0.04, pad=0.02)
        cbar.set_label('|U| [m s$^{-1}$]')

    fig.suptitle(f'{layer_name} velocity — {Ldir["label"]} '
                 f'({Ldir["file_type"]}) '
                 f'({Ldir["ds0"]} to {Ldir["ds1"]})', fontsize=14)

    tag = layer if layer else '_davg'
    out_fn = Ldir['out_dir'] / (f'phase_quiver{tag}_{Ldir["label"]}_'
                                 f'{Ldir["file_type"]}_'
                                 f'{Ldir["ds0"]}_{Ldir["ds1"]}.png')
    fig.savefig(out_fn, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_fn}')
    plt.close(fig)


# -----------------------------------------------------------------------
# Season-comparison plots (phases x seasons grid)
# -----------------------------------------------------------------------
SEASONS = ['JF', 'MA', 'MJ', 'JA', 'SO', 'ND']
SEASON_NAMES = {
    'JF': 'Jan-Feb', 'MA': 'Mar-Apr', 'MJ': 'May-Jun',
    'JA': 'Jul-Aug', 'SO': 'Sep-Oct', 'ND': 'Nov-Dec',
}
PHASES4 = ['spring_flood', 'spring_ebb', 'neap_flood', 'neap_ebb']
PHASE_NAMES4 = ['Spring Flood', 'Spring Ebb', 'Neap Flood', 'Neap Ebb']


def _load_phase_panel(fn, vn, base_vn, factor):
    """Read one phase_avg file; return panel dict or None."""
    if not fn.is_file():
        return None
    ds = xr.open_dataset(fn)
    if vn not in ds:
        ds.close()
        return None
    fld = ds[vn].values * factor
    n_ts = ds.attrs.get('n_timesteps', '?')
    if base_vn == 'u':
        lon_key, lat_key = 'lon_u', 'lat_u'
    elif base_vn == 'v':
        lon_key, lat_key = 'lon_v', 'lat_v'
    else:
        lon_key, lat_key = 'lon_rho', 'lat_rho'
    if lon_key in ds and lat_key in ds:
        plon, plat = pfun.get_plon_plat(ds[lon_key].values,
                                        ds[lat_key].values)
    else:
        plon = plat = None
    ds.close()
    return {'fld': fld, 'plon': plon, 'plat': plat, 'n': n_ts}


def plot_phase_avg_fields_by_season(Ldir, vn='salt'):
    """Phases (rows) x seasons (cols) grid for one scalar variable/layer."""
    style_cmap, diverging = VAR_STYLE.get(
        vn.replace('_top', '').replace('_bot', ''), (cmo.haline, False))
    factor, unit_label, base_vn = _resolve_units(vn)
    bounds = ZOOM_BOUNDS.get(Ldir['label'])

    avg_dir = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
               / ('phase_avg_' + Ldir['ds0'] + '_' + Ldir['ds1']
                  + '_' + Ldir['file_type']))

    # grid[i_phase][i_season] = panel or None
    grid = [[None] * len(SEASONS) for _ in PHASES4]
    for i, pn in enumerate(PHASES4):
        for j, sn in enumerate(SEASONS):
            fn = avg_dir / (Ldir['label'] + '_' + sn + '_' + pn + '.nc')
            grid[i][j] = _load_phase_panel(fn, vn, base_vn, factor)

    # Shared vlims across all panels
    sample_vals = []
    for row in grid:
        for p in row:
            if p is None:
                continue
            v = p['fld']
            if p['plon'] is not None:
                lon_c = 0.25 * (p['plon'][:-1, :-1] + p['plon'][1:, :-1] +
                                p['plon'][:-1, 1:] + p['plon'][1:, 1:])
                lat_c = 0.25 * (p['plat'][:-1, :-1] + p['plat'][1:, :-1] +
                                p['plat'][:-1, 1:] + p['plat'][1:, 1:])
                mask = _color_mask(lon_c, lat_c, Ldir['label'], bounds)
                v = v[mask]
            sample_vals.append(v[np.isfinite(v)].ravel())
    if sample_vals and any(s.size for s in sample_vals):
        allv = np.concatenate(sample_vals)
        if diverging:
            vmax = float(np.nanpercentile(np.abs(allv), 98))
            vmin = -vmax
        else:
            vmin = float(np.nanpercentile(allv, 2))
            vmax = float(np.nanpercentile(allv, 98))
    else:
        vmin, vmax = -1.0, 1.0

    nrow, ncol = len(PHASES4), len(SEASONS)
    # Cell aspect ~ width/height of zoom box (Penn Cove is wide & short)
    if bounds is not None:
        cw = bounds[1] - bounds[0]
        ch = (bounds[3] - bounds[2]) / np.cos(np.deg2rad(
            0.5 * (bounds[2] + bounds[3])))
        cell_aspect = cw / ch if ch > 0 else 2.0
    else:
        cell_aspect = 2.0
    cell_h = 1.6
    cell_w = cell_h * cell_aspect
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(cell_w * ncol + 1.2, cell_h * nrow + 1.0),
        sharex=True, sharey=True,
        gridspec_kw={'wspace': 0.04, 'hspace': 0.04})
    last_cs = None
    for i in range(nrow):
        for j in range(ncol):
            ax = axes[i, j]
            p = grid[i][j]
            if i == 0:
                ax.set_title(SEASON_NAMES[SEASONS[j]], fontsize=12)
            if j == 0:
                ax.set_ylabel(PHASE_NAMES4[i], fontsize=12,
                              labelpad=8)
            if p is None:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
                ax.set_xticks([]); ax.set_yticks([])
                continue
            fld_plot = p['fld'].astype(float).copy()
            if p['plon'] is not None and bounds is not None:
                lon_c = 0.25 * (p['plon'][:-1, :-1] + p['plon'][1:, :-1] +
                                p['plon'][:-1, 1:] + p['plon'][1:, 1:])
                lat_c = 0.25 * (p['plat'][:-1, :-1] + p['plat'][1:, :-1] +
                                p['plat'][:-1, 1:] + p['plat'][1:, 1:])
                outside = ~((lon_c >= bounds[0]) & (lon_c <= bounds[1])
                            & (lat_c >= bounds[2]) & (lat_c <= bounds[3]))
                for ex in PLOT_EXCLUDE.get(Ldir['label'], []):
                    outside |= ((lon_c >= ex[0]) & (lon_c <= ex[1])
                                & (lat_c >= ex[2]) & (lat_c <= ex[3]))
                fld_plot[outside] = np.nan
            cs = ax.pcolormesh(p['plon'], p['plat'], fld_plot,
                               cmap=style_cmap, vmin=vmin, vmax=vmax)
            pfun.add_coast(ax)
            pfun.dar(ax)
            if bounds is not None:
                ax.set_xlim(bounds[0], bounds[1])
                ax.set_ylim(bounds[2], bounds[3])
            ax.tick_params(labelsize=8)
            # Only show tick labels on outer edges
            if i != nrow - 1:
                ax.tick_params(labelbottom=False)
            if j != 0:
                ax.tick_params(labelleft=False)
            ax.text(0.97, 0.05, f'n={p["n"]}', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.75, lw=0))
            last_cs = cs

    if last_cs is not None:
        cbar = fig.colorbar(last_cs, ax=axes, location='right',
                            shrink=0.9, fraction=0.02, pad=0.01)
        cbar.set_label(unit_label, fontsize=11)

    if vn.endswith('_top'):
        layer = 'Surface'
    elif vn.endswith('_bot'):
        layer = 'Bottom'
    else:
        layer = 'Depth-Averaged'
    fig.suptitle(f'{layer} {base_vn} by season — {Ldir["label"]} '
                 f'({Ldir["file_type"]}) '
                 f'({Ldir["ds0"]} to {Ldir["ds1"]})', fontsize=13)

    out_fn = Ldir['out_dir'] / (
        f'phase_avg_season_{vn}_{Ldir["label"]}_{Ldir["file_type"]}_'
        f'{Ldir["ds0"]}_{Ldir["ds1"]}.png')
    fig.savefig(out_fn, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_fn}')
    plt.close(fig)


def plot_phase_avg_quiver_by_season(Ldir, layer='', skip=2, scale=None):
    """Phases x seasons grid of velocity quivers."""
    bounds = ZOOM_BOUNDS.get(Ldir['label'])
    avg_dir = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
               / ('phase_avg_' + Ldir['ds0'] + '_' + Ldir['ds1']
                  + '_' + Ldir['file_type']))
    u_vn = 'u' + layer
    v_vn = 'v' + layer
    layer_name = {'': 'Depth-Averaged',
                  '_top': 'Surface',
                  '_bot': 'Bottom'}[layer]

    grid = [[None] * len(SEASONS) for _ in PHASES4]
    speed_vals = []
    for i, pn in enumerate(PHASES4):
        for j, sn in enumerate(SEASONS):
            fn = avg_dir / (Ldir['label'] + '_' + sn + '_' + pn + '.nc')
            if not fn.is_file():
                continue
            ds = xr.open_dataset(fn)
            if u_vn not in ds or v_vn not in ds:
                ds.close()
                continue
            u = ds[u_vn].values
            v = ds[v_vn].values
            u_rho = 0.5 * (u[:, :-1] + u[:, 1:])
            v_rho = 0.5 * (v[:-1, :] + v[1:, :])
            u_int = u_rho[1:-1, :]
            v_int = v_rho[:, 1:-1]
            lon_rho = ds['lon_rho'].values[1:-1, 1:-1]
            lat_rho = ds['lat_rho'].values[1:-1, 1:-1]
            n_ts = ds.attrs.get('n_timesteps', '?')
            ds.close()
            spd = np.sqrt(u_int**2 + v_int**2)
            mask = _color_mask(lon_rho, lat_rho, Ldir['label'], bounds)
            speed_vals.append(spd[mask & np.isfinite(spd)].ravel())
            grid[i][j] = {'u': u_int, 'v': v_int, 'spd': spd,
                          'lon': lon_rho, 'lat': lat_rho, 'n': n_ts}

    if not any(p is not None for row in grid for p in row):
        print(f'  no u/v data for layer="{layer}"; skipping season quiver.')
        return

    if speed_vals:
        all_spd = np.concatenate(speed_vals)
        smax = float(np.nanpercentile(all_spd, 98)) if all_spd.size else 0.5
    else:
        smax = 0.5
    if scale is None:
        scale = max(smax * 12.0, 0.5)

    nrow, ncol = len(PHASES4), len(SEASONS)
    if bounds is not None:
        cw = bounds[1] - bounds[0]
        ch = (bounds[3] - bounds[2]) / np.cos(np.deg2rad(
            0.5 * (bounds[2] + bounds[3])))
        cell_aspect = cw / ch if ch > 0 else 2.0
    else:
        cell_aspect = 2.0
    cell_h = 1.6
    cell_w = cell_h * cell_aspect
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(cell_w * ncol + 1.2, cell_h * nrow + 1.0),
        sharex=True, sharey=True,
        gridspec_kw={'wspace': 0.04, 'hspace': 0.04})
    last_cs = None
    for i in range(nrow):
        for j in range(ncol):
            ax = axes[i, j]
            p = grid[i][j]
            if i == 0:
                ax.set_title(SEASON_NAMES[SEASONS[j]], fontsize=12)
            if j == 0:
                ax.set_ylabel(PHASE_NAMES4[i], fontsize=12,
                              labelpad=8)
            if p is None:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
                ax.set_xticks([]); ax.set_yticks([])
                continue
            spd_plot = p['spd'].astype(float).copy()
            u_plot = p['u'].astype(float).copy()
            v_plot = p['v'].astype(float).copy()
            if bounds is not None:
                outside = ~((p['lon'] >= bounds[0])
                            & (p['lon'] <= bounds[1])
                            & (p['lat'] >= bounds[2])
                            & (p['lat'] <= bounds[3]))
                for ex in PLOT_EXCLUDE.get(Ldir['label'], []):
                    outside |= ((p['lon'] >= ex[0]) & (p['lon'] <= ex[1])
                                & (p['lat'] >= ex[2]) & (p['lat'] <= ex[3]))
                spd_plot[outside] = np.nan
                u_plot[outside] = np.nan
                v_plot[outside] = np.nan
            cs = ax.pcolormesh(p['lon'], p['lat'], spd_plot,
                               cmap=cmo.speed, vmin=0, vmax=smax,
                               shading='auto')
            ax.quiver(p['lon'][::skip, ::skip], p['lat'][::skip, ::skip],
                      u_plot[::skip, ::skip], v_plot[::skip, ::skip],
                      scale=scale, color='k', width=0.003)
            pfun.add_coast(ax)
            pfun.dar(ax)
            if bounds is not None:
                ax.set_xlim(bounds[0], bounds[1])
                ax.set_ylim(bounds[2], bounds[3])
            ax.tick_params(labelsize=8)
            if i != nrow - 1:
                ax.tick_params(labelbottom=False)
            if j != 0:
                ax.tick_params(labelleft=False)
            ax.text(0.97, 0.05, f'n={p["n"]}', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.75, lw=0))
            last_cs = cs

    if last_cs is not None:
        cbar = fig.colorbar(last_cs, ax=axes, location='right',
                            shrink=0.9, fraction=0.02, pad=0.01)
        cbar.set_label('|U| [m s$^{-1}$]', fontsize=11)

    fig.suptitle(f'{layer_name} velocity by season — {Ldir["label"]} '
                 f'({Ldir["file_type"]}) '
                 f'({Ldir["ds0"]} to {Ldir["ds1"]})', fontsize=13)

    tag = layer if layer else '_davg'
    out_fn = Ldir['out_dir'] / (
        f'phase_quiver_season{tag}_{Ldir["label"]}_'
        f'{Ldir["file_type"]}_{Ldir["ds0"]}_{Ldir["ds1"]}.png')
    fig.savefig(out_fn, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_fn}')
    plt.close(fig)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == '__main__':
    Ldir = get_args()

    label = Ldir['label']
    ds0 = Ldir['ds0']
    ds1 = Ldir['ds1']

    # Load phase labels
    phase_fn = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
                / ('tide_phases_' + ds0 + '_' + ds1)
                / (label + '.nc'))

    if phase_fn.is_file():
        ds_phase = xr.open_dataset(phase_fn)
        print('--- Plot 1: Zeta with phase shading ---')
        plot_zeta_phases(ds_phase, Ldir)
        ds_phase.close()
    else:
        print(f'Phase file not found: {phase_fn}')

    # Phase-averaged fields — try common variables
    avg_dir = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
               / ('phase_avg_' + ds0 + '_' + ds1 + '_' + Ldir['file_type']))
    if avg_dir.is_dir():
        sample_files = list(avg_dir.glob(label + '_*.nc'))
        if sample_files:
            ds_sample = xr.open_dataset(sample_files[0])
            available = set(ds_sample.data_vars)
            ds_sample.close()

            # Variables to plot at depth-avg / surface / bottom (when present)
            scalar_vns = ['salt', 'temp', 'oxygen']
            layered_vns = []
            for base in scalar_vns + ['u', 'v']:
                for suffix in ('', '_top', '_bot'):
                    vn = base + suffix
                    if vn in available:
                        layered_vns.append(vn)

            # Anything else (non-grid, non-velocity) the file has
            grid_keys = {'lon_rho', 'lat_rho', 'lon_u', 'lat_u',
                         'lon_v', 'lat_v', 'h',
                         'mask_rho', 'mask_u', 'mask_v'}
            extras = sorted(v for v in available
                            if v not in grid_keys and v not in layered_vns)

            print('\n--- Plot 2: Phase-averaged scalar fields ---')
            for vn in layered_vns + extras:
                if Ldir['by_season']:
                    plot_phase_avg_fields_by_season(Ldir, vn=vn)
                else:
                    plot_phase_avg_fields(Ldir, vn=vn)

            # Quivers — depth-avg, surface, bottom
            print('\n--- Plot 3: Phase-averaged velocity quivers ---')
            for layer in ('', '_top', '_bot'):
                if ('u' + layer) in available and ('v' + layer) in available:
                    if Ldir['by_season']:
                        plot_phase_avg_quiver_by_season(Ldir, layer=layer)
                    else:
                        plot_phase_avg_quiver(Ldir, layer=layer)
    else:
        print(f'No phase_avg directory found: {avg_dir}')

    print('\nDone.')
