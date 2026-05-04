"""
Track eddies across snapshots from run_swirl_roms.py output.

Reads the per-snapshot vortex CSV produced by run_swirl_roms.py
(<method>_vortices_*.csv) and links detections that are spatially
close on consecutive snapshots into tracks. Outputs:

    1. <input>_tracked.csv     — original CSV + 'track_id' column
    2. <input>_tracks.csv      — one row per track with persistence stats
    3. (optional) tide-phase fractions per track if -phase_file given

Tracking algorithm (greedy nearest-neighbor on centroids):
  - Snapshots are processed in chronological order (date, file_num).
  - For each detection at snapshot t, find the nearest detection in
    snapshot t-1 with the same rotation sign and within `max_dist_km`.
  - If a match exists, inherit its track_id; otherwise start a new track.
  - Each prior detection can match at most one current detection
    (smaller distance wins).
  - Tracks may have gaps up to `max_gap` snapshots (still counted as
    same track if a detection reappears within the gap and distance).

Usage
-----
    # Mirror the run_swirl_roms.py CLI to find the input CSV
    python track_eddies.py -gtx wb1_t0_xn11ab \
        -0 2024.01.01 -1 2024.01.31 \
        -method ow -ftype his -vel surface \
        [-max_dist_km 2.0] [-max_gap 1] [-min_persistence 3] \
        [-phase_file path/to/phases_penn_cove.nc]

    # Or pass an explicit CSV path
    python track_eddies.py -gtx wb1_t0_xn11ab \
        -csv /path/to/ow_vortices_..._.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from lo_tools import Lfun


# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description='Track eddies across snapshots.')

    # --- Run identification (matches run_swirl_roms.py) ---
    p.add_argument('-gtx', '--gtagex', type=str, required=True,
                   help='gtagex, e.g. wb1_t0_xn11ab')
    p.add_argument('-0', '--ds0', type=str, default=None,
                   help='Start date YYYY.MM.DD (matches run_swirl_roms output)')
    p.add_argument('-1', '--ds1', type=str, default=None,
                   help='End date YYYY.MM.DD')
    p.add_argument('-method', type=str, default='ow',
                   choices=['ow', 'vorticity', 'swirl'],
                   help='Detection method used by run_swirl_roms.py')
    p.add_argument('-ftype', '--file_type', type=str, default='his',
                   choices=['his', 'avg'])
    p.add_argument('-vel', '--vel_type', type=str, default='surface',
                   choices=['surface', 'depth_avg', 'depth_level'])

    # --- Override / explicit path ---
    p.add_argument('-csv', type=str, default=None,
                   help='Explicit CSV path (overrides -gtx/-0/-1/etc).')
    p.add_argument('-in_dir', type=str, default=None,
                   help='Directory containing the CSV '
                        '(default: LOo/swirl/<gtagex>/).')
    p.add_argument('-out_dir', type=str, default=None,
                   help='Output directory (default: same as input dir).')

    # --- Tracking parameters ---
    p.add_argument('-max_dist_km', type=float, default=2.0,
                   help='Max centroid displacement to link detections [km]')
    p.add_argument('-max_gap', type=int, default=1,
                   help='Max consecutive missing snapshots within a track')
    p.add_argument('-min_persistence', type=int, default=2,
                   help='Drop tracks with fewer than this many detections')
    p.add_argument('-snapshot_dt_hours', type=float, default=1.0,
                   help='Hours between snapshots (for duration column)')
    p.add_argument('-phase_file', type=str, default=None,
                   help='Optional NetCDF with is_flood/is_spring labels '
                        'from compute_tide_phases.py')

    args = p.parse_args()

    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
    default_dir = Ldir['LOo'] / 'swirl' / args.gtagex

    # Resolve input CSV
    if args.csv is None:
        if args.ds0 is None or args.ds1 is None:
            sys.exit('ERROR: provide either -csv or both -0 and -1.')
        in_dir = Path(args.in_dir) if args.in_dir else default_dir
        fname = (f'{args.method}_vortices_{args.ds0}_{args.ds1}'
                 f'_{args.file_type}_{args.vel_type}.csv')
        args.csv = str(in_dir / fname)

    if args.out_dir is None:
        args.out_dir = str(Path(args.csv).parent)

    return args


# ---------------------------------------------------------------------------
def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km between scalar/array points."""
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
def assign_tracks(df, max_dist_km, max_gap):
    """
    Assign a track_id to every row of df. df must already have a
    snapshot_idx column (0-based, monotonic) and rotation column.
    """
    df = df.sort_values(['snapshot_idx', 'vortex_id']).reset_index(drop=True)
    n = len(df)
    track_ids = np.full(n, -1, dtype=int)
    next_track_id = 0

    # Group rows by snapshot
    snap_groups = {s: g.index.to_numpy() for s, g in df.groupby('snapshot_idx')}
    snap_indices = sorted(snap_groups.keys())

    # active_tracks: track_id -> dict(last_snap, last_lon, last_lat, rotation)
    active = {}

    for si in snap_indices:
        rows = snap_groups[si]
        # Candidates from active tracks within max_gap missing snapshots.
        # max_gap=0 means must appear in immediately-previous snapshot.
        # max_gap=1 means may skip 1 snapshot, etc.
        candidates = [
            (tid, info) for tid, info in active.items()
            if (si - info['last_snap']) <= (max_gap + 1)
        ]

        # Build distance matrix between current detections and candidates
        # Greedy: sort all (detection, track) pairs by distance, assign in order
        pairs = []
        for r in rows:
            r_lon = df.at[r, 'center_lon']
            r_lat = df.at[r, 'center_lat']
            r_rot = df.at[r, 'rotation']
            for tid, info in candidates:
                if info['rotation'] != r_rot:
                    continue
                d = haversine_km(r_lon, r_lat,
                                 info['last_lon'], info['last_lat'])
                if d <= max_dist_km:
                    pairs.append((d, r, tid))
        pairs.sort(key=lambda x: x[0])

        used_rows = set()
        used_tracks = set()
        for d, r, tid in pairs:
            if r in used_rows or tid in used_tracks:
                continue
            track_ids[r] = tid
            used_rows.add(r)
            used_tracks.add(tid)

        # Unmatched detections start new tracks
        for r in rows:
            if track_ids[r] == -1:
                track_ids[r] = next_track_id
                next_track_id += 1

        # Update active dict with current snapshot's tracks
        for r in rows:
            tid = track_ids[r]
            active[tid] = dict(
                last_snap=si,
                last_lon=df.at[r, 'center_lon'],
                last_lat=df.at[r, 'center_lat'],
                rotation=df.at[r, 'rotation'],
            )

        # Drop tracks that have aged out
        active = {tid: info for tid, info in active.items()
                  if (si - info['last_snap']) <= (max_gap + 1)}

    df['track_id'] = track_ids
    return df


# ---------------------------------------------------------------------------
def summarize_tracks(df, snapshot_dt_hours):
    """One row per track with persistence + mean stats."""
    rows = []
    for tid, g in df.groupby('track_id'):
        g = g.sort_values('snapshot_idx')
        n = len(g)
        s0 = int(g['snapshot_idx'].iloc[0])
        s1 = int(g['snapshot_idx'].iloc[-1])
        span = s1 - s0 + 1  # snapshots from first to last appearance
        rec = dict(
            track_id=int(tid),
            n_detections=n,
            snapshot_first=s0,
            snapshot_last=s1,
            span_snapshots=span,
            duration_hours=span * snapshot_dt_hours,
            rotation=g['rotation'].iloc[0],
            mean_lon=float(g['center_lon'].mean()),
            mean_lat=float(g['center_lat'].mean()),
            mean_radius_m=float(g['radius_m'].mean()),
            mean_n_cells=float(g['n_cells'].mean()),
        )
        if 'mean_vorticity' in g:
            rec['mean_vorticity'] = float(g['mean_vorticity'].mean())
        if 'time' in g:
            rec['time_first'] = g['time'].iloc[0]
            rec['time_last'] = g['time'].iloc[-1]
        # phase fractions if available
        for col in ('is_flood', 'is_spring'):
            if col in g:
                rec[f'frac_{col}'] = float(g[col].mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values('n_detections', ascending=False)


# ---------------------------------------------------------------------------
def attach_time_and_phase(df, phase_file):
    """
    Build a 'time' column from (date, file_num) ordering and optionally
    join with tide-phase labels.
    """
    # Snapshot ordering: sort unique (date, file_num) and assign idx.
    # We don't reconstruct exact timestamps from filenames here; the
    # snapshot_idx is what tracking needs. If a phase file is provided
    # we resample by snapshot count (assumes regular cadence).
    keys = (df[['date', 'file_num']].drop_duplicates()
            .sort_values(['date', 'file_num']).reset_index(drop=True))
    keys['snapshot_idx'] = np.arange(len(keys))
    df = df.merge(keys, on=['date', 'file_num'], how='left')

    if phase_file is None:
        return df

    ds = xr.open_dataset(phase_file)
    t = pd.DatetimeIndex(ds['time'].values)
    is_flood = ds['is_flood'].values.astype(bool)
    is_spring = (ds['is_spring'].values.astype(bool)
                 if 'is_spring' in ds else None)
    ds.close()

    n_snap = len(keys)
    if len(t) < n_snap:
        print(f'WARNING: phase file has {len(t)} timesteps but CSV implies '
              f'{n_snap} snapshots. Truncating.')
        n_snap = len(t)

    # Assume snapshot_idx maps directly into the phase array
    keys['time'] = t[:n_snap].repeat(1)[:n_snap]
    keys['is_flood'] = is_flood[:n_snap]
    if is_spring is not None:
        keys['is_spring'] = is_spring[:n_snap]
    df = df.drop(columns=[c for c in ('time', 'is_flood', 'is_spring')
                          if c in df.columns])
    df = df.merge(keys[['date', 'file_num', 'time', 'is_flood']
                       + (['is_spring'] if is_spring is not None else [])],
                  on=['date', 'file_num'], how='left')
    return df


# ---------------------------------------------------------------------------
def main():
    args = get_args()
    csv_in = Path(args.csv)
    if not csv_in.is_file():
        print(f'ERROR: CSV not found: {csv_in}')
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else csv_in.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_in)
    if df.empty:
        print('Input CSV is empty. Nothing to track.')
        sys.exit(0)

    print(f'Loaded {len(df)} detections from {csv_in.name}')

    df = attach_time_and_phase(df, args.phase_file)
    print(f'Spans {df["snapshot_idx"].nunique()} snapshots')

    df = assign_tracks(df, args.max_dist_km, args.max_gap)
    n_tracks = df['track_id'].nunique()
    print(f'Linked into {n_tracks} raw tracks')

    summary = summarize_tracks(df, args.snapshot_dt_hours)
    summary = summary[summary['n_detections'] >= args.min_persistence]
    print(f'{len(summary)} tracks meet -min_persistence={args.min_persistence}')

    # Filter detections to surviving tracks for the tracked CSV
    keep_ids = set(summary['track_id'].tolist())
    df_out = df[df['track_id'].isin(keep_ids)].copy()

    stem = csv_in.stem
    tracked_csv = out_dir / f'{stem}_tracked.csv'
    tracks_csv = out_dir / f'{stem}_tracks.csv'
    df_out.to_csv(tracked_csv, index=False)
    summary.to_csv(tracks_csv, index=False)
    print(f'Saved: {tracked_csv}')
    print(f'Saved: {tracks_csv}')

    # Quick stdout summary
    if not summary.empty:
        print('\nTop tracks by persistence:')
        cols = ['track_id', 'n_detections', 'duration_hours',
                'rotation', 'mean_lon', 'mean_lat', 'mean_radius_m']
        cols += [c for c in ('frac_is_flood', 'frac_is_spring')
                 if c in summary.columns]
        print(summary[cols].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
