"""
Plot eddy trajectories from track_eddies.py output.

Reads a *_tracked.csv (per-detection with track_id) and produces:
  1. A map of all surviving track trajectories, colored by track_id,
     with start/end markers and rotation indicated.
  2. (Optional) A separate small-multiples figure showing the top N
     longest-lived tracks individually.

Usage
-----
    # Use the same conventions as run_swirl_roms.py / track_eddies.py
    python plot_tracks.py -gtx wb1_t0_xn11ab \
        -0 2024.01.01 -1 2024.01.31 \
        [-method ow] [-ftype his] [-vel surface] \
        [-top_n 12] [-min_persistence 3] \
        [-penn_cove True]

    # Or pass an explicit tracked CSV
    python plot_tracks.py -gtx wb1_t0_xn11ab \
        -tracked_csv /path/to/ow_vortices_..._tracked.csv
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mcm

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun


PENN_COVE_BBOX = (-122.74, -122.56, 48.21, 48.26)


# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description='Plot eddy trajectories.')
    p.add_argument('-gtx', '--gtagex', type=str, required=True)
    p.add_argument('-0', '--ds0', type=str, default=None)
    p.add_argument('-1', '--ds1', type=str, default=None)
    p.add_argument('-method', type=str, default='ow',
                   choices=['ow', 'vorticity', 'swirl'])
    p.add_argument('-ftype', '--file_type', type=str, default='his',
                   choices=['his', 'avg'])
    p.add_argument('-vel', '--vel_type', type=str, default='surface',
                   choices=['surface', 'depth_avg', 'depth_level'])
    p.add_argument('-tracked_csv', type=str, default=None,
                   help='Explicit *_tracked.csv path (overrides date-based)')
    p.add_argument('-in_dir', type=str, default=None,
                   help='Dir containing the tracked CSV '
                        '(default: LOo/swirl/<gtagex>/)')
    p.add_argument('-out_dir', type=str, default=None,
                   help='Plot output dir (default: same as in_dir)')
    p.add_argument('-top_n', type=int, default=12,
                   help='Number of longest tracks to show in small multiples')
    p.add_argument('-min_persistence', type=int, default=3,
                   help='Hide tracks with fewer than this many detections')
    p.add_argument('-penn_cove', type=lambda s: s.lower() == 'true',
                   default=False,
                   help='Restrict map view to Penn Cove bbox.')
    p.add_argument('-grid_file', type=str, default=None,
                   help='Optional ROMS grid.nc for coastline backdrop '
                        '(default: Ldir["grid"]/grid.nc).')

    args = p.parse_args()

    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
    default_dir = Ldir['LOo'] / 'swirl' / args.gtagex

    if args.tracked_csv is None:
        if args.ds0 is None or args.ds1 is None:
            sys.exit('ERROR: provide either -tracked_csv or -0 and -1.')
        in_dir = Path(args.in_dir) if args.in_dir else default_dir
        fname = (f'{args.method}_vortices_{args.ds0}_{args.ds1}'
                 f'_{args.file_type}_{args.vel_type}_tracked.csv')
        args.tracked_csv = str(in_dir / fname)

    if args.out_dir is None:
        args.out_dir = str(Path(args.tracked_csv).parent)

    if args.grid_file is None:
        args.grid_file = str(Ldir['grid'] / 'grid.nc')

    return args


# ---------------------------------------------------------------------------
def _load_grid_for_backdrop(grid_file):
    """Open grid.nc; return None if unavailable so plotting still works."""
    try:
        import xarray as xr
        return xr.open_dataset(grid_file)
    except Exception as e:
        print(f'  (no coastline backdrop: {e})')
        return None


def _draw_backdrop(ax, dsg, bbox=None):
    """Light bathymetry + coastline backdrop."""
    if dsg is None:
        return
    lon = dsg.lon_rho.values
    lat = dsg.lat_rho.values
    h = dsg.h.values.copy()
    h[dsg.mask_rho.values == 0] = np.nan
    plon, plat = pfun.get_plon_plat(lon, lat)
    ax.pcolormesh(plon, plat, h, cmap='Greys', alpha=0.3,
                  shading='flat', zorder=0)
    pfun.add_coast(ax)
    pfun.dar(ax)
    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])


# ---------------------------------------------------------------------------
def plot_all_tracks(df, dsg, bbox, out_path, title):
    """One map with every surviving track as a polyline."""
    track_ids = sorted(df['track_id'].unique())
    n_tracks = len(track_ids)

    cmap = mcm.get_cmap('tab20', max(n_tracks, 1))

    fig, ax = plt.subplots(figsize=(11, 9))
    _draw_backdrop(ax, dsg, bbox)

    for k, tid in enumerate(track_ids):
        g = df[df['track_id'] == tid].sort_values('snapshot_idx')
        lon = g['center_lon'].values
        lat = g['center_lat'].values
        rot = g['rotation'].iloc[0]

        color = cmap(k % cmap.N)
        ls = '-' if rot == 1 else '--'
        ax.plot(lon, lat, ls, color=color, linewidth=1.2, alpha=0.85,
                zorder=2)
        # Start: open circle; End: filled square
        ax.plot(lon[0], lat[0], 'o', color=color, markersize=5,
                markerfacecolor='white', markeredgewidth=1.2, zorder=3)
        ax.plot(lon[-1], lat[-1], 's', color=color, markersize=5,
                zorder=3)
        # Label at midpoint
        mid = len(lon) // 2
        ax.text(lon[mid], lat[mid], str(int(tid)), fontsize=7,
                color='black', alpha=0.9, zorder=4,
                ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.6, pad=0.5))

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'{title}\n{n_tracks} tracks  '
                 '(solid=CCW, dashed=CW; ○=start, ■=end)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


def plot_top_n_tracks(df, dsg, bbox, out_path, top_n, title):
    """Small multiples of the top N longest-lived tracks."""
    counts = df.groupby('track_id').size().sort_values(ascending=False)
    top_ids = counts.head(top_n).index.tolist()
    n = len(top_ids)
    if n == 0:
        return
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow),
                             squeeze=False)

    for k, tid in enumerate(top_ids):
        ax = axes[k // ncol, k % ncol]
        _draw_backdrop(ax, dsg, bbox)
        g = df[df['track_id'] == tid].sort_values('snapshot_idx')
        lon = g['center_lon'].values
        lat = g['center_lat'].values
        sidx = g['snapshot_idx'].values
        rot_str = 'CCW' if g['rotation'].iloc[0] == 1 else 'CW'

        # Color by snapshot order
        sc = ax.scatter(lon, lat, c=sidx, cmap='viridis', s=18,
                        zorder=3, edgecolor='k', linewidth=0.3)
        ax.plot(lon, lat, 'k-', linewidth=0.6, alpha=0.5, zorder=2)
        ax.plot(lon[0], lat[0], 'wo', markersize=7,
                markeredgecolor='k', markeredgewidth=1.2, zorder=4)
        ax.plot(lon[-1], lat[-1], 'ks', markersize=7, zorder=4)

        ax.set_title(f'track {int(tid)}: {len(g)} dets, {rot_str}',
                     fontsize=9)
        ax.tick_params(labelsize=7)
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02,
                     label='snapshot_idx')

    # Hide unused axes
    for k in range(n, nrow * ncol):
        axes[k // ncol, k % ncol].axis('off')

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    args = get_args()
    csv_in = Path(args.tracked_csv)
    if not csv_in.is_file():
        sys.exit(f'ERROR: tracked CSV not found: {csv_in}')

    df = pd.read_csv(csv_in)
    if df.empty:
        sys.exit('Tracked CSV is empty.')

    # Optional re-filter by persistence (in case tracks csv was looser)
    counts = df.groupby('track_id').size()
    keep = counts[counts >= args.min_persistence].index
    df = df[df['track_id'].isin(keep)].copy()
    print(f'{len(df)} detections across {df["track_id"].nunique()} tracks '
          f'(min_persistence={args.min_persistence})')

    if df.empty:
        sys.exit('No tracks meet -min_persistence.')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_in.stem.replace('_tracked', '')

    bbox = PENN_COVE_BBOX if args.penn_cove else None
    dsg = _load_grid_for_backdrop(args.grid_file)

    title = f'{stem}'
    plot_all_tracks(df, dsg, bbox,
                    out_dir / f'{stem}_tracks_map.png', title)
    plot_top_n_tracks(df, dsg, bbox,
                      out_dir / f'{stem}_top{args.top_n}.png',
                      args.top_n, title)

    if dsg is not None:
        dsg.close()


if __name__ == '__main__':
    main()
