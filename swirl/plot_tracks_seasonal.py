"""
Season-aware track plotting.

Consumes a *_tracked_with_phase.csv produced by add_tide_phase.py
(which carries 'season' and 'phase' columns from the tide_phase pipeline)
and produces:

  1. <stem>_tracks_by_season.png  — small-multiples map, one panel per
     season (JF/MA/MJ/JA/SO/ND), tracks colored by track_id.
  2. <stem>_tracks_by_phase.png   — small-multiples map, one panel per
     dominant tide phase (spring_flood, spring_ebb, neap_flood,
     neap_ebb, slack_high, slack_low).
  3. <stem>_season_summary.png    — bar chart: # tracks per season,
     split by rotation (CCW vs CW).

Each track is assigned a single 'dominant season' / 'dominant phase'
(the most common label across its detections).

Usage
-----
    python plot_tracks_seasonal.py -gtx wb1_t0_xn11ab \
        -0 2024.01.01 -1 2024.01.31 \
        [-method ow] [-ftype his] [-vel surface] \
        [-min_persistence 3] [-penn_cove True]

    # Or pass an explicit annotated CSV:
    python plot_tracks_seasonal.py -gtx wb1_t0_xn11ab \
        -csv /path/to/..._tracked_with_phase.csv
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib.ticker import MaxNLocator

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

# Reuse season ordering from tide_phase
sys.path.insert(0, str(Path(__file__).parent.parent / 'tide_phase'))
from phase_avg_fields import SEASONS  # noqa: E402

SEASON_ORDER = list(SEASONS.keys())  # JF, MA, MJ, JA, SO, ND
SEASON_NAMES = {
    'JF': 'Jan-Feb', 'MA': 'Mar-Apr', 'MJ': 'May-Jun',
    'JA': 'Jul-Aug', 'SO': 'Sep-Oct', 'ND': 'Nov-Dec',
}
PHASES4 = ['spring_flood', 'spring_ebb', 'neap_flood', 'neap_ebb']
PHASE_NAMES4 = ['Spring Flood', 'Spring Ebb', 'Neap Flood', 'Neap Ebb']
PHASE_ORDER = PHASES4 + ['slack_high', 'slack_low']

# Discrete colors for the 6 tide-phase categories (+ unclassified -> grey)
PHASE_COLORS = {
    'spring_flood': '#08519c',  # deep blue
    'spring_ebb':   '#a50f15',  # deep red
    'neap_flood':   '#6baed6',  # light blue
    'neap_ebb':     '#fb6a4a',  # light red
    'slack_high':   '#ffd92f',  # yellow
    'slack_low':    '#7570b3',  # purple
    'unclassified': '#bdbdbd',  # grey
}

PENN_COVE_BBOX = (-122.74, -122.625, 48.215, 48.245)


# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description='Season-aware track plots.')
    p.add_argument('-gtx', '--gtagex', type=str, required=True)
    p.add_argument('-0', '--ds0', type=str, default=None)
    p.add_argument('-1', '--ds1', type=str, default=None)
    p.add_argument('-method', type=str, default='ow',
                   choices=['ow', 'vorticity', 'swirl'])
    p.add_argument('-ftype', '--file_type', type=str, default='his',
                   choices=['his', 'avg'])
    p.add_argument('-vel', '--vel_type', type=str, default='surface',
                   choices=['surface', 'depth_avg', 'depth_level'])
    p.add_argument('-csv', type=str, default=None,
                   help='Explicit *_tracked_with_phase.csv path.')
    p.add_argument('-out_dir', type=str, default=None)
    p.add_argument('-min_persistence', type=int, default=10)
    p.add_argument('-penn_cove', type=lambda s: s.lower() == 'true',
                   default=False)
    p.add_argument('-grid_file', type=str, default=None)

    args = p.parse_args()

    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
    default_dir = Ldir['LOo'] / 'swirl' / args.gtagex

    if args.csv is None:
        if args.ds0 is None or args.ds1 is None:
            sys.exit('ERROR: provide either -csv or -0 and -1.')
        fname = (f'{args.method}_vortices_{args.ds0}_{args.ds1}'
                 f'_{args.file_type}_{args.vel_type}'
                 f'_tracked_with_phase.csv')
        args.csv = str(default_dir / fname)
    if args.out_dir is None:
        args.out_dir = str(default_dir / 'plots')
    if args.grid_file is None:
        args.grid_file = str(Ldir['grid'] / 'grid.nc')
    return args


# ---------------------------------------------------------------------------
def _load_grid(grid_file):
    try:
        import xarray as xr
        return xr.open_dataset(grid_file)
    except Exception as e:
        print(f'  (no coastline backdrop: {e})')
        return None


def _draw_backdrop(ax, dsg, bbox=None):
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


def _plot_one_panel(ax, df_panel, dsg, bbox, label, cmap):
    _draw_backdrop(ax, dsg, bbox)
    track_ids = sorted(df_panel['track_id'].unique())
    for k, tid in enumerate(track_ids):
        g = df_panel[df_panel['track_id'] == tid].sort_values('snapshot_idx')
        lon = g['center_lon'].values
        lat = g['center_lat'].values
        rot = g['rotation'].iloc[0]
        color = cmap(k % cmap.N)
        ls = '-' if rot == 'CCW' else '--'
        ax.plot(lon, lat, ls, color=color, linewidth=1.0, alpha=0.85,
                zorder=2)
        ax.plot(lon[0], lat[0], 'o', color=color, markersize=4,
                markerfacecolor='white', markeredgewidth=1.0, zorder=3)
        ax.plot(lon[-1], lat[-1], 's', color=color, markersize=4,
                zorder=3)
    ax.set_title(f'{label}  (n={len(track_ids)})', fontsize=10)
    ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
def assign_dominant(df, col):
    """Most common value of `col` per track_id."""
    return (df.dropna(subset=[col])
              .groupby('track_id')[col]
              .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None))


def plot_by_category(df, dsg, bbox, out_path, category_col, order, title):
    """Small-multiples: one panel per category value, in given order."""
    dom = assign_dominant(df, category_col)
    df = df.copy()
    df['_cat'] = df['track_id'].map(dom)

    # Keep only categories that appear (preserve order)
    present = [c for c in order if c in dom.values]
    if not present:
        print(f'  (no tracks have a "{category_col}"; skipping)')
        return
    n = len(present)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))

    cmap = mcm.get_cmap('tab20', max(df['track_id'].nunique(), 1))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.4 * nrow),
                             squeeze=False)
    for k, cat in enumerate(present):
        ax = axes[k // ncol, k % ncol]
        sub = df[df['_cat'] == cat]
        _plot_one_panel(ax, sub, dsg, bbox, cat, cmap)
    for k in range(n, nrow * ncol):
        axes[k // ncol, k % ncol].axis('off')

    # Rotation legend
    handles = [
        plt.Line2D([], [], color='0.3', linestyle='-', linewidth=1.2,
                   label='CCW'),
        plt.Line2D([], [], color='0.3', linestyle='--', linewidth=1.2,
                   label='CW'),
        plt.Line2D([], [], marker='o', linestyle='',
                   markerfacecolor='white', markeredgecolor='0.3',
                   markersize=6, label='start'),
        plt.Line2D([], [], marker='s', linestyle='', color='0.3',
                   markersize=6, label='end'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'{title}  —  by dominant {category_col}', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


def plot_tracks_phase_x_season(df, dsg, bbox, out_path, title,
                               color_by='flood_fraction'):
    """
    4 phases (rows) x 6 seasons (cols) grid of track maps.
    Styling matches tide_phase_analysis.plot_phase_avg_fields_by_season:
      - Aspect-ratio sizing from bbox
      - sharex/sharey, tight gridspec
      - SEASON_NAMES headers, PHASE_NAMES4 row labels
      - MaxNLocator(3) ticks, outer-edge labels only
      - n=... annotation in lower-right
      - Single shared colorbar (when color_by='flood_fraction')
    """
    if 'phase' not in df.columns or 'season' not in df.columns:
        print('  (need both "phase" and "season" columns)')
        return

    dom_phase = assign_dominant(df, 'phase')
    dom_season = assign_dominant(df, 'season')
    rot = df.groupby('track_id')['rotation'].first()
    flood_frac = (df.groupby('track_id')['is_flood'].mean()
                  if 'is_flood' in df.columns else None)

    track_meta = pd.DataFrame({
        'phase': dom_phase, 'season': dom_season, 'rotation': rot,
    })
    if flood_frac is not None:
        track_meta['flood_frac'] = flood_frac

    # Color setup
    norm = cmap = None
    if color_by == 'flood_fraction' and flood_frac is not None:
        cmap = plt.get_cmap('RdBu')
        norm = plt.Normalize(0, 1)
        def color_fn(tid):
            f = track_meta.loc[tid, 'flood_frac']
            return cmap(norm(f)) if not np.isnan(f) else '0.5'
    elif color_by == 'rotation':
        def color_fn(tid):
            return '#1f77b4' if track_meta.loc[tid, 'rotation'] == 'CCW' \
                else '#d62728'
    else:
        tab = mcm.get_cmap('tab20', max(len(track_meta), 1))
        ids_sorted = sorted(track_meta.index)
        idx_map = {tid: i for i, tid in enumerate(ids_sorted)}
        def color_fn(tid):
            return tab(idx_map[tid] % tab.N)

    nrow, ncol = len(PHASES4), len(SEASON_ORDER)

    # Cell aspect from bbox (lon/lat-corrected)
    if bbox is not None:
        cw = bbox[1] - bbox[0]
        ch = (bbox[3] - bbox[2]) / np.cos(np.deg2rad(
            0.5 * (bbox[2] + bbox[3])))
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

    for i, ph in enumerate(PHASES4):
        for j, sn in enumerate(SEASON_ORDER):
            ax = axes[i, j]
            _draw_backdrop(ax, dsg, bbox)

            sel_ids = track_meta[(track_meta['phase'] == ph) &
                                 (track_meta['season'] == sn)].index
            for tid in sel_ids:
                g = df[df['track_id'] == tid].sort_values('snapshot_idx')
                rot_str = g['rotation'].iloc[0]
                ls = '-' if rot_str == 'CCW' else '--'
                ax.plot(g['center_lon'].values, g['center_lat'].values,
                        ls, color=color_fn(tid), linewidth=1.0,
                        alpha=0.85, zorder=2)

            if i == 0:
                ax.set_title(SEASON_NAMES[sn], fontsize=12)
            if j == 0:
                ax.set_ylabel(PHASE_NAMES4[i], fontsize=12, labelpad=8)

            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
            if i != nrow - 1:
                ax.tick_params(labelbottom=False)
            if j != 0:
                ax.tick_params(labelleft=False)

            ax.text(0.97, 0.05, f'n={len(sel_ids)}',
                    transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.75, lw=0))

    if color_by == 'flood_fraction' and flood_frac is not None:
        sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, location='right',
                            shrink=0.9, fraction=0.02, pad=0.01)
        cbar.set_label('Flood fraction along track  (0=ebb, 1=flood)',
                       fontsize=11)
    elif color_by == 'rotation':
        handles = [
            plt.Line2D([], [], color='#1f77b4', linewidth=1.4, label='CCW'),
            plt.Line2D([], [], color='#d62728', linewidth=1.4, label='CW'),
        ]
        fig.legend(handles=handles, loc='lower center', ncol=2,
                   fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(f'Eddy tracks by phase x season — {title}  '
                 f'(color={color_by})', fontsize=13)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


def plot_tracks_phase_colored(df, dsg, bbox, out_path, title,
                              by_season=False):
    """
    Single map (or per-season facets) where each detection along a track
    is colored by its instantaneous tide phase. Track lines are drawn in
    light grey underneath so the trajectory remains visible.
    """
    if 'phase' not in df.columns:
        print('  (no "phase" column; skipping phase-colored plot)')
        return

    if by_season:
        seasons_present = [s for s in SEASON_ORDER
                           if (df['season'] == s).any()]
        if not seasons_present:
            return
        n = len(seasons_present)
        ncol = min(3, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(4.6 * ncol, 4.4 * nrow),
                                 squeeze=False)
        panels = [(seasons_present[k], df[df['season'] == seasons_present[k]],
                   axes[k // ncol, k % ncol]) for k in range(n)]
        for k in range(n, nrow * ncol):
            axes[k // ncol, k % ncol].axis('off')
    else:
        fig, ax = plt.subplots(figsize=(11, 9))
        panels = [(None, df, ax)]

    for label, sub, ax in panels:
        _draw_backdrop(ax, dsg, bbox)
        # Light grey trajectory lines
        for tid, g in sub.groupby('track_id'):
            g = g.sort_values('snapshot_idx')
            ax.plot(g['center_lon'].values, g['center_lat'].values,
                    '-', color='0.6', linewidth=0.8, alpha=0.6, zorder=2)
        # Detection points colored by phase
        for ph, gp in sub.groupby('phase'):
            ax.scatter(gp['center_lon'].values, gp['center_lat'].values,
                       c=PHASE_COLORS.get(ph, '#bdbdbd'),
                       s=14, edgecolor='k', linewidth=0.2,
                       label=ph, zorder=3)
        if label is not None:
            ax.set_title(f'{label}  (n_tracks={sub["track_id"].nunique()})',
                         fontsize=10)
        ax.tick_params(labelsize=7)

    # One shared legend
    handles = [plt.Line2D([], [], marker='o', linestyle='',
                          markerfacecolor=PHASE_COLORS[p],
                          markeredgecolor='k', markeredgewidth=0.3,
                          markersize=7, label=p)
               for p in PHASE_ORDER]
    fig.legend(handles=handles, loc='lower center', ncol=6,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    suptitle = (f'{title}  —  detections colored by tide phase'
                + ('  (faceted by season)' if by_season else ''))
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


def plot_tracks_flood_fraction(df, dsg, bbox, out_path, title):
    """
    Single map: each track drawn as one line, colored by its
    flood-fraction (0 = always ebbing, 1 = always flooding).
    """
    if 'is_flood' not in df.columns:
        print('  (no "is_flood" column; skipping flood-fraction plot)')
        return

    frac = df.groupby('track_id')['is_flood'].mean()

    fig, ax = plt.subplots(figsize=(11, 9))
    _draw_backdrop(ax, dsg, bbox)
    cmap = plt.get_cmap('RdBu')  # red=ebb, blue=flood
    norm = plt.Normalize(vmin=0, vmax=1)

    for tid, g in df.groupby('track_id'):
        g = g.sort_values('snapshot_idx')
        f = float(frac.loc[tid]) if tid in frac.index else np.nan
        if np.isnan(f):
            continue
        color = cmap(norm(f))
        rot = g['rotation'].iloc[0]
        ls = '-' if rot == 'CCW' else '--'
        ax.plot(g['center_lon'].values, g['center_lat'].values,
                ls, color=color, linewidth=1.4, alpha=0.9, zorder=2)
        ax.plot(g['center_lon'].iloc[0], g['center_lat'].iloc[0], 'o',
                color=color, markersize=4, markerfacecolor='white',
                markeredgewidth=1.0, zorder=3)
        ax.plot(g['center_lon'].iloc[-1], g['center_lat'].iloc[-1], 's',
                color=color, markersize=4, zorder=3)

    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('Flood fraction along track  (0=ebb, 1=flood)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'{title}  —  tracks colored by flood fraction\n'
                 '(solid=CCW, dashed=CW; ○=start, ■=end)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


def plot_season_summary(df, out_path, title):
    """Bar chart: # tracks per season split by rotation."""
    dom_season = assign_dominant(df, 'season')
    rot = df.groupby('track_id')['rotation'].first()
    sum_df = pd.DataFrame({'season': dom_season, 'rotation': rot}).dropna()

    counts = (sum_df.groupby(['season', 'rotation'])
                    .size().unstack(fill_value=0)
                    .reindex(SEASON_ORDER, fill_value=0))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(counts))
    w = 0.4
    if 'CCW' in counts.columns:
        ax.bar(x - w / 2, counts['CCW'].values, w,
               label='CCW', color='#1f77b4')
    if 'CW' in counts.columns:
        ax.bar(x + w / 2, counts['CW'].values, w,
               label='CW', color='#d62728')
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index)
    ax.set_xlabel('Season (2-month bin)')
    ax.set_ylabel('# tracks (dominant season)')
    ax.set_title(f'{title}  —  tracks per season')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    args = get_args()
    csv_in = Path(args.csv)
    if not csv_in.is_file():
        sys.exit(f'ERROR: not found: {csv_in}\n'
                 'Run add_tide_phase.py first.')

    df = pd.read_csv(csv_in)
    if df.empty:
        sys.exit('Empty CSV.')

    if 'season' not in df.columns:
        sys.exit('ERROR: CSV has no "season" column. '
                 'Run add_tide_phase.py to annotate.')

    counts = df.groupby('track_id').size()
    keep = counts[counts >= args.min_persistence].index
    df = df[df['track_id'].isin(keep)].copy()
    print(f'{len(df)} detections across {df["track_id"].nunique()} tracks '
          f'(min_persistence={args.min_persistence})')
    if df.empty:
        sys.exit('No tracks meet -min_persistence.')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_in.stem.replace('_tracked_with_phase', '')

    bbox = PENN_COVE_BBOX if args.penn_cove else None
    dsg = _load_grid(args.grid_file)
    title = stem

    plot_by_category(df, dsg, bbox,
                     out_dir / f'{stem}_tracks_by_season.png',
                     'season', SEASON_ORDER, title)

    if 'phase' in df.columns:
        plot_by_category(df, dsg, bbox,
                         out_dir / f'{stem}_tracks_by_phase.png',
                         'phase', PHASE_ORDER, title)

    plot_season_summary(df, out_dir / f'{stem}_season_summary.png', title)

    plot_tracks_phase_colored(df, dsg, bbox,
                              out_dir / f'{stem}_tracks_phase_colored.png',
                              title, by_season=False)
    plot_tracks_phase_colored(
        df, dsg, bbox,
        out_dir / f'{stem}_tracks_phase_colored_by_season.png',
        title, by_season=True)
    plot_tracks_flood_fraction(df, dsg, bbox,
                               out_dir / f'{stem}_tracks_flood_fraction.png',
                               title)

    plot_tracks_phase_x_season(
        df, dsg, bbox,
        out_dir / f'{stem}_tracks_phase_x_season_flood.png',
        title, color_by='flood_fraction')
    plot_tracks_phase_x_season(
        df, dsg, bbox,
        out_dir / f'{stem}_tracks_phase_x_season_rotation.png',
        title, color_by='rotation')

    if dsg is not None:
        dsg.close()


if __name__ == '__main__':
    main()
