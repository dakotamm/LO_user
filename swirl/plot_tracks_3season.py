#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepared: 2026/06/11

Author: Dakota Mascarenas

Three-season eddy track maps for the swirl (Okubo-Weiss) eddy-tracking output.

Reads a per-detection ``*_tracked.csv`` (one row per eddy detection, carrying
center_lon/center_lat, rotation, snapshot_idx, track_id, time) and draws eddy
trajectories grouped into three 4-month seasons:

    Winter (Dec-Mar)   months 12, 1, 2, 3   (Dec rolled into the next year)
    Spring (Apr-Jul)   months 4-7
    Low-DO (Aug-Nov)   months 8-11

Season names / binning match Mascarenas et al. (2026, R1). Each track is
assigned to a single season (the most common season across its detections).

Outputs (to <ROOT>/plots/):
    <stem>_tracks_3season.png             3-panel map, one panel per season
    <stem>_tracks_<season>.png            one standalone map per season

Styling follows Desktop/Mascarenas_etal_2026_R1_working/figure_*.py:
    - transparent background, dpi=500, bbox_inches='tight'
    - lightgray dashed grid (alpha=0.5)
    - rotation colors  CCW=blue #4565e8 (cyclonic), CW=red #e04256 (anticyclonic)
    - bold lower-left panel letters (a, b, c)

Usage
-----
    python plot_tracks_3season.py                       # defaults below
    python plot_tracks_3season.py -gtx wb1_r0_xn11b \
        -0 2017.01.01 -1 2017.12.31 -vel depth_avg \
        [-method ow] [-ftype his] [-min_persistence 3]
    python plot_tracks_3season.py -csv /path/to/..._tracked.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

# --- paper styling (Mascarenas et al. 2026 R1) ----------------------------
RED = '#e04256'    # CW / anticyclonic
BLUE = '#4565e8'   # CCW / cyclonic
ROT_COLOR = {'CCW': BLUE, 'CW': RED}
GRID_KW = dict(color='lightgray', linestyle='--', alpha=0.5)

# month -> season label; December folds forward into Winter of the next year
SEASONS = {
    'Winter (Dec-Mar)': (12, 1, 2, 3),
    'Spring (Apr-Jul)': (4, 5, 6, 7),
    'Low-DO (Aug-Nov)': (8, 9, 10, 11),
}
SEASON_ORDER = list(SEASONS)
MONTH_TO_SEASON = {m: s for s, months in SEASONS.items() for m in months}
PANEL_LETTERS = ['a', 'b', 'c']


# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description='Three-season eddy track maps.')
    p.add_argument('-gtx', '--gtagex', type=str, default='wb1_r0_xn11b')
    p.add_argument('-0', '--ds0', type=str, default='2017.01.01')
    p.add_argument('-1', '--ds1', type=str, default='2017.12.31')
    p.add_argument('-method', type=str, default='ow',
                   choices=['ow', 'vorticity', 'swirl'])
    p.add_argument('-ftype', '--file_type', type=str, default='his',
                   choices=['his', 'avg'])
    p.add_argument('-vel', '--vel_type', type=str, default='depth_avg',
                   choices=['surface', 'depth_avg', 'depth_level'])
    p.add_argument('-csv', type=str, default=None,
                   help='Explicit per-detection *_tracked.csv path.')
    p.add_argument('-out_dir', type=str, default=None)
    p.add_argument('-min_persistence', type=int, default=3,
                   help='Drop tracks with fewer than this many detections.')
    p.add_argument('-grid_file', type=str, default=None)
    args = p.parse_args()

    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
    default_dir = Ldir['LOo'] / 'swirl' / args.gtagex

    if args.csv is None:
        fname = (f'{args.method}_vortices_{args.ds0}_{args.ds1}'
                 f'_{args.file_type}_{args.vel_type}_tracked.csv')
        args.csv = str(default_dir / fname)
    if args.out_dir is None:
        args.out_dir = str(default_dir / 'plots')
    if args.grid_file is None:
        args.grid_file = str(Ldir['grid'] / 'grid.nc')
    return args


# ---------------------------------------------------------------------------
def load_grid(grid_file):
    try:
        import xarray as xr
        return xr.open_dataset(grid_file)
    except Exception as e:
        print(f'  (no model-grid backdrop: {e})')
        return None


def draw_backdrop(ax, dsg, bbox):
    """Greyscale bathymetry + coastline, fixed to the data extent."""
    if dsg is not None:
        lon = dsg.lon_rho.values
        lat = dsg.lat_rho.values
        h = dsg.h.values.copy()
        h[dsg.mask_rho.values == 0] = np.nan
        plon, plat = pfun.get_plon_plat(lon, lat)
        ax.pcolormesh(plon, plat, h, cmap='Greys', alpha=0.3,
                      shading='flat', zorder=-5)
        pfun.add_coast(ax)
        pfun.dar(ax)
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.grid(zorder=-7, **GRID_KW)


def plot_panel(ax, df_season, dsg, bbox):
    """Draw all trajectories for one season, colored by rotation sense."""
    draw_backdrop(ax, dsg, bbox)
    for tid, g in df_season.groupby('track_id'):
        g = g.sort_values('snapshot_idx')
        lon = g['center_lon'].values
        lat = g['center_lat'].values
        color = ROT_COLOR.get(g['rotation'].iloc[0], '0.4')
        ax.plot(lon, lat, '-', color=color, linewidth=1.0, alpha=0.7,
                zorder=2)
        ax.plot(lon[0], lat[0], 'o', color=color, markersize=4,
                markerfacecolor='white', markeredgewidth=1.0, zorder=3)
        ax.plot(lon[-1], lat[-1], 's', color=color, markersize=4, zorder=3)
    ax.tick_params(labelsize=8)


def rotation_legend_handles():
    return [
        plt.Line2D([], [], color=BLUE, linewidth=1.4, label='CCW (cyclonic)'),
        plt.Line2D([], [], color=RED, linewidth=1.4, label='CW (anticyclonic)'),
        plt.Line2D([], [], marker='o', linestyle='', markerfacecolor='white',
                   markeredgecolor='0.3', markersize=6, label='start'),
        plt.Line2D([], [], marker='s', linestyle='', color='0.3',
                   markersize=6, label='end'),
    ]


def panel_letter(ax, letter):
    ax.text(0.025, 0.95, letter, transform=ax.transAxes, fontsize=14,
            fontweight='bold', color='k', va='top', ha='left')


# ---------------------------------------------------------------------------
def assign_track_season(df):
    """Most common season across each track's detections."""
    season = (df.dropna(subset=['season'])
                .groupby('track_id')['season']
                .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None))
    return df['track_id'].map(season)


def data_bbox(df, pad=0.004):
    lon0, lon1 = df['center_lon'].min(), df['center_lon'].max()
    lat0, lat1 = df['center_lat'].min(), df['center_lat'].max()
    return (lon0 - pad, lon1 + pad, lat0 - pad, lat1 + pad)


def main():
    args = get_args()
    csv_in = Path(args.csv)
    if not csv_in.is_file():
        sys.exit(f'ERROR: not found: {csv_in}\n'
                 'Run track_eddies.py first to produce a *_tracked.csv.')

    df = pd.read_csv(csv_in, parse_dates=['time'])
    if df.empty:
        sys.exit('Empty CSV.')

    # min-persistence filter
    counts = df.groupby('track_id').size()
    keep = counts[counts >= args.min_persistence].index
    df = df[df['track_id'].isin(keep)].copy()
    if df.empty:
        sys.exit('No tracks meet -min_persistence.')

    # season per detection -> dominant season per track
    df['season'] = df['time'].dt.month.map(MONTH_TO_SEASON)
    df['season'] = assign_track_season(df)
    df = df.dropna(subset=['season'])

    n_tracks = df['track_id'].nunique()
    print(f'{len(df)} detections across {n_tracks} tracks '
          f'(min_persistence={args.min_persistence})')
    for s in SEASON_ORDER:
        print(f'  {s:18s}: {df[df.season == s].track_id.nunique()} tracks')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_in.stem.replace('_tracked', '')

    bbox = data_bbox(df)
    dsg = load_grid(args.grid_file)

    # --- combined 3-panel figure ------------------------------------------
    fig, axd = plt.subplot_mosaic([SEASON_ORDER], figsize=(13, 5.2),
                                  layout='constrained')
    for letter, season in zip(PANEL_LETTERS, SEASON_ORDER):
        ax = axd[season]
        sub = df[df['season'] == season]
        plot_panel(ax, sub, dsg, bbox)
        ax.set_title(f'{season}  (n={sub.track_id.nunique()})', fontsize=11)
        panel_letter(ax, letter)
    fig.legend(handles=rotation_legend_handles(), loc='lower center',
               ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.04))
    out = out_dir / f'{stem}_tracks_3season.png'
    fig.savefig(out, bbox_inches='tight', dpi=500, transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)

    # --- standalone per-season figures ------------------------------------
    for season in SEASON_ORDER:
        sub = df[df['season'] == season]
        fig, ax = plt.subplots(figsize=(6, 5.5), layout='constrained')
        plot_panel(ax, sub, dsg, bbox)
        ax.set_title(f'{season}  (n={sub.track_id.nunique()})', fontsize=12)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(handles=rotation_legend_handles(), loc='upper left',
                  fontsize=9, frameon=False)
        tag = season.split(' ')[0].lower()  # winter / spring / low-do
        out = out_dir / f'{stem}_tracks_{tag}.png'
        fig.savefig(out, bbox_inches='tight', dpi=500, transparent=True)
        print(f'Saved: {out}')
        plt.close(fig)

    if dsg is not None:
        dsg.close()


if __name__ == '__main__':
    main()
