"""
Visualize tidal phase detection results and phase-averaged fields.

Reads output from compute_tide_phases.py, phase_avg_fields.py, and
phase_avg_budgets.py to create summary plots.

Designed to run locally (after copying results from apogee).

Usage
-----
    python tide_phase_analysis.py -gtx wb1_t0_xn11ab -label penn_cove \
        -0 2024.01.01 -1 2024.06.30

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

# Map zoom: Penn Cove + northern Saratoga Passage
ZOOM_BOUNDS = {
    'penn_cove': (-122.78, -122.40, 48.05, 48.30),
}


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

    style_cmap, diverging = VAR_STYLE.get(vn, (cmo.haline, False))
    if cmap is None:
        cmap = style_cmap

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
        fld = ds[vn].values
        n_ts = ds.attrs.get('n_timesteps', '?')
        if vn == 'u':
            lon_key, lat_key = 'lon_u', 'lat_u'
        elif vn == 'v':
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
        # Restrict vlim stats to cells inside the zoom window when possible
        sample_vals = []
        for p in panels:
            if p is None:
                continue
            v = p['fld']
            if bounds is not None and p['plon'] is not None:
                # plon/plat are corner arrays (one larger than fld in each dim);
                # average to cell centers for masking
                lon_c = 0.25 * (p['plon'][:-1, :-1] + p['plon'][1:, :-1] +
                                p['plon'][:-1, 1:] + p['plon'][1:, 1:])
                lat_c = 0.25 * (p['plat'][:-1, :-1] + p['plat'][1:, :-1] +
                                p['plat'][:-1, 1:] + p['plat'][1:, 1:])
                in_box = ((lon_c >= bounds[0]) & (lon_c <= bounds[1])
                          & (lat_c >= bounds[2]) & (lat_c <= bounds[3]))
                v = v[in_box]
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
            cs = ax.pcolormesh(panel['plon'], panel['plat'], panel['fld'],
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
        cbar.set_label(vn)

    fig.suptitle(f'Depth-Averaged {vn} — {Ldir["label"]} '
                 f'({Ldir["file_type"]}) '
                 f'({Ldir["ds0"]} to {Ldir["ds1"]})', fontsize=14)

    out_fn = Ldir['out_dir'] / (f'phase_avg_{vn}_{Ldir["label"]}_{Ldir["file_type"]}_'
                                 f'{Ldir["ds0"]}_{Ldir["ds1"]}.png')
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
            available_vns = [vn for vn in ds_sample.data_vars
                             if vn not in ('lon_rho', 'lat_rho', 'lon_u', 'lat_u',
                                           'lon_v', 'lat_v', 'h',
                                           'mask_rho', 'mask_u', 'mask_v')]
            ds_sample.close()

            print('\n--- Plot 2: Phase-averaged fields ---')
            for vn in available_vns:
                plot_phase_avg_fields(Ldir, vn=vn)
    else:
        print(f'No phase_avg directory found: {avg_dir}')

    print('\nDone.')
