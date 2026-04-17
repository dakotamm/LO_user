"""
Visualize tidal phase detection results and phase-averaged fields.

Reads output from compute_tide_phases.py, phase_avg_fields.py, and
phase_avg_budgets.py to create summary plots.

Designed to run locally (after copying results from apogee).

Usage
-----
    python tide_phase_analysis.py -gtx wb1_r0_xn11b -sect_name pc0 \
        -0 2017.09.01 -1 2017.09.30

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
# Argument parsing
# -----------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description='Plot tidal phase analysis results.')
    parser.add_argument('-gtx', '--gtagex', type=str, required=True)
    parser.add_argument('-0', '--ds0', type=str, required=True)
    parser.add_argument('-1', '--ds1', type=str, required=True)
    parser.add_argument('-sect_name', type=str, required=True)
    parser.add_argument('-out_dir', type=str, default=None,
                        help='Output directory for plots (default: Desktop/pltz)')

    args = parser.parse_args()

    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
    for k, v in vars(args).items():
        if k not in Ldir:
            Ldir[k] = v

    if Ldir['out_dir'] is None:
        Ldir['out_dir'] = Path.home() / 'Desktop' / 'pltz'
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
    ax.set_title(f'Sea Surface Height — {Ldir["sect_name"]} '
                 f'({Ldir["ds0"]} to {Ldir["ds1"]})')
    ax.legend(handles=[
        Patch(facecolor='tab:red', label='Flood'),
        Patch(facecolor='tab:blue', label='Ebb'),
        Patch(facecolor='#ffe0e0', alpha=0.5, label='Spring'),
        Patch(facecolor='#e0e0ff', alpha=0.5, label='Neap'),
    ], loc='upper left', fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_fn = Ldir['out_dir'] / (f'zeta_phases_{Ldir["sect_name"]}_'
                                 f'{Ldir["ds0"]}_{Ldir["ds1"]}.png')
    fig.savefig(out_fn, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_fn}')
    plt.close(fig)


# -----------------------------------------------------------------------
# Plot 2: 2x2 phase-averaged fields
# -----------------------------------------------------------------------
def plot_phase_avg_fields(Ldir, vn='u', cmap=cmo.balance, vlims=None):
    """2x2 panel of depth-averaged field for each tidal phase."""
    phase_names = ['spring_flood', 'spring_ebb', 'neap_flood', 'neap_ebb']
    titles = ['Spring Flood', 'Spring Ebb', 'Neap Flood', 'Neap Ebb']

    avg_dir = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
               / ('phase_avg_' + Ldir['ds0'] + '_' + Ldir['ds1']))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (pn, title) in enumerate(zip(phase_names, titles)):
        ax = axes.flat[idx]
        fn = avg_dir / (Ldir['sect_name'] + '_' + pn + '.nc')
        if not fn.is_file():
            ax.set_title(f'{title}\n(no data)')
            continue

        ds = xr.open_dataset(fn)

        if vn not in ds:
            ax.set_title(f'{title}\n({vn} not available)')
            ds.close()
            continue

        fld = ds[vn].values
        n_ts = ds.attrs.get('n_timesteps', '?')

        # Determine appropriate grid coordinates
        if vn in ('u',):
            lon_key, lat_key = 'lon_u', 'lat_u'
        elif vn in ('v',):
            lon_key, lat_key = 'lon_v', 'lat_v'
        else:
            lon_key, lat_key = 'lon_rho', 'lat_rho'

        if lon_key in ds and lat_key in ds:
            lon = ds[lon_key].values
            lat = ds[lat_key].values
            plon, plat = pfun.get_plon_plat(lon, lat)
        else:
            plon = plat = None

        ds.close()

        if vlims is None:
            vmax = np.nanpercentile(np.abs(fld), 95)
            vmin = -vmax
        else:
            vmin, vmax = vlims

        if plon is not None:
            cs = ax.pcolormesh(plon, plat, fld, cmap=cmap,
                               vmin=vmin, vmax=vmax)
            pfun.add_coast(ax)
            pfun.dar(ax)
        else:
            cs = ax.pcolormesh(fld, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_title(f'{title}\n(n={n_ts})')
        fig.colorbar(cs, ax=ax, fraction=0.046)

    fig.suptitle(f'Depth-Averaged {vn} — {Ldir["sect_name"]} '
                 f'({Ldir["ds0"]} to {Ldir["ds1"]})', fontsize=14)
    fig.tight_layout()

    out_fn = Ldir['out_dir'] / (f'phase_avg_{vn}_{Ldir["sect_name"]}_'
                                 f'{Ldir["ds0"]}_{Ldir["ds1"]}.png')
    fig.savefig(out_fn, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_fn}')
    plt.close(fig)


# -----------------------------------------------------------------------
# Plot 3: Phase-resolved transport bar chart
# -----------------------------------------------------------------------
def plot_phase_budgets(Ldir):
    """Bar chart of qnet, Qin, Qout by phase."""
    budget_dir = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
                  / ('phase_budget_' + Ldir['ds0'] + '_' + Ldir['ds1']))
    budget_fn = budget_dir / (Ldir['sect_name'] + '.nc')

    if not budget_fn.is_file():
        print(f'Budget file not found: {budget_fn} — skipping budget plot.')
        return

    ds = xr.open_dataset(budget_fn)
    phases = ds['phase'].values
    qnet = ds['qnet_mean'].values
    Qin = ds['Qin_mean'].values
    Qout = ds['Qout_mean'].values
    n_ts = ds['n_timesteps'].values
    ds.close()

    x = np.arange(len(phases))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, qnet, width, label='Qnet', color='gray')
    ax.bar(x, Qin, width, label='Qin', color='tab:red', alpha=0.7)
    ax.bar(x + width, Qout, width, label='Qout', color='tab:blue', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{p}\n(n={n})' for p, n in zip(phases, n_ts)])
    ax.set_ylabel('Transport [m³/s]')
    ax.set_title(f'Phase-Resolved Transport — {Ldir["sect_name"]} '
                 f'({Ldir["ds0"]} to {Ldir["ds1"]})')
    ax.legend()
    ax.axhline(0, color='k', linewidth=0.5)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    out_fn = Ldir['out_dir'] / (f'phase_budget_{Ldir["sect_name"]}_'
                                 f'{Ldir["ds0"]}_{Ldir["ds1"]}.png')
    fig.savefig(out_fn, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_fn}')
    plt.close(fig)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == '__main__':
    Ldir = get_args()

    sect_name = Ldir['sect_name']
    ds0 = Ldir['ds0']
    ds1 = Ldir['ds1']

    # Load phase labels
    phase_fn = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
                / ('tide_phases_' + ds0 + '_' + ds1)
                / (sect_name + '.nc'))

    if phase_fn.is_file():
        ds_phase = xr.open_dataset(phase_fn)
        print('--- Plot 1: Zeta with phase shading ---')
        plot_zeta_phases(ds_phase, Ldir)
        ds_phase.close()
    else:
        print(f'Phase file not found: {phase_fn}')

    # Phase-averaged fields — try common variables
    avg_dir = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
               / ('phase_avg_' + ds0 + '_' + ds1))
    if avg_dir.is_dir():
        # Check which variables are available in any phase file
        sample_files = list(avg_dir.glob(sect_name + '_*.nc'))
        if sample_files:
            ds_sample = xr.open_dataset(sample_files[0])
            available_vns = [vn for vn in ds_sample.data_vars
                             if vn not in ('lon_rho', 'lat_rho', 'lon_u', 'lat_u',
                                           'lon_v', 'lat_v', 'h',
                                           'mask_rho', 'mask_u', 'mask_v')]
            ds_sample.close()

            print('\n--- Plot 2: Phase-averaged fields ---')
            for vn in available_vns:
                cmap = cmo.balance if vn in ('u', 'v') else cmo.haline
                plot_phase_avg_fields(Ldir, vn=vn, cmap=cmap)
    else:
        print(f'No phase_avg directory found: {avg_dir}')

    # Budget bar chart
    print('\n--- Plot 3: Phase-resolved transport ---')
    plot_phase_budgets(Ldir)

    print('\nDone.')
