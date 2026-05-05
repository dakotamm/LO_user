"""
Annotate eddy CSVs with tide phase + 2-month season.

Reuses the tide_phase pipeline output:
  - Phase NetCDF from compute_tide_phases.py (LOo/tide_phase/<gtagex>/
    tide_phases_<ds0>_<ds1>/<label>.nc) provides per-timestep
    is_flood / is_ebb / is_spring / is_neap / slack_hi / slack_lo
    plus a categorical 'phase' variable.
  - SEASONS dict from tide_phase.phase_avg_fields (JF / MA / MJ / JA / SO / ND).

Inputs handled:
  1. Per-detection CSV (raw or *_tracked):
       Adds columns: phase, is_flood, is_spring, season
  2. Per-track summary CSV (*_tracks):
       Adds columns: frac_is_flood, frac_is_spring, dominant_phase,
                     dominant_season

Joining uses pd.merge_asof on the actual ROMS timestamp ('time' column).
This requires run_swirl_roms.py to have written a 'time' column (added
recently); older CSVs without 'time' will fall back to the ordinal
join used inside track_eddies.py.

Usage
-----
    # Mirror run_swirl_roms.py / track_eddies.py CLI
    python add_tide_phase.py -gtx wb1_t0_xn11ab \
        -0 2024.01.01 -1 2024.01.31 \
        -label penn_cove

    # Specify both files explicitly
    python add_tide_phase.py -gtx wb1_t0_xn11ab \
        -csv /path/to/ow_vortices_..._tracked.csv \
        -phase_file /path/to/tide_phases_.../penn_cove.nc
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

# Reuse the existing season definition
sys.path.insert(0, str(Path(__file__).parent.parent / 'tide_phase'))
from phase_avg_fields import SEASONS, _season_of  # noqa: E402


# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description='Annotate eddy CSV with tide phase + season.')

    # Mirror run_swirl_roms.py CLI for path resolution
    p.add_argument('-gtx', '--gtagex', type=str, required=True)
    p.add_argument('-0', '--ds0', type=str, default=None)
    p.add_argument('-1', '--ds1', type=str, default=None)
    p.add_argument('-method', type=str, default='ow',
                   choices=['ow', 'vorticity', 'swirl'])
    p.add_argument('-ftype', '--file_type', type=str, default='his')
    p.add_argument('-vel', '--vel_type', type=str, default='surface')
    p.add_argument('-label', type=str, default='penn_cove',
                   help='Label used by tide_phase pipeline.')

    # Which CSV(s) to annotate
    p.add_argument('-which', type=str, default='both',
                   choices=['detections', 'tracks', 'both'],
                   help='Which CSV to annotate: per-detection (_tracked), '
                        'per-track (_tracks), or both.')

    # Explicit overrides
    p.add_argument('-csv', type=str, default=None,
                   help='Explicit CSV path (forces -which to match its type).')
    p.add_argument('-phase_file', type=str, default=None,
                   help='Explicit phase NetCDF (overrides label-based).')
    p.add_argument('-out_dir', type=str, default=None,
                   help='Output dir (default: same as input CSV dir).')

    args = p.parse_args()

    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
    swirl_dir = Ldir['LOo'] / 'swirl' / args.gtagex

    # Resolve phase file
    if args.phase_file is None:
        if args.ds0 is None or args.ds1 is None:
            sys.exit('ERROR: provide -phase_file or both -0 and -1.')
        args.phase_file = str(
            Ldir['LOo'] / 'tide_phase' / args.gtagex
            / f'tide_phases_{args.ds0}_{args.ds1}' / f'{args.label}.nc')

    # Resolve detection/track CSVs
    if args.csv is not None:
        args._explicit_csvs = [Path(args.csv)]
    else:
        if args.ds0 is None or args.ds1 is None:
            sys.exit('ERROR: provide -csv or both -0 and -1.')
        base = (f'{args.method}_vortices_{args.ds0}_{args.ds1}'
                f'_{args.file_type}_{args.vel_type}')
        candidates = []
        if args.which in ('detections', 'both'):
            candidates.append(swirl_dir / f'{base}_tracked.csv')
        if args.which in ('tracks', 'both'):
            candidates.append(swirl_dir / f'{base}_tracks.csv')
        args._explicit_csvs = candidates

    if args.out_dir is None:
        args.out_dir = str(args._explicit_csvs[0].parent)

    return args


# ---------------------------------------------------------------------------
def load_phase_dataframe(phase_file):
    """Load tide_phase NetCDF as a DataFrame indexed by time."""
    ds = xr.open_dataset(phase_file)
    t = pd.DatetimeIndex(ds['time'].values)
    df = pd.DataFrame({'time': t})
    for vn in ('is_flood', 'is_ebb', 'is_spring', 'is_neap',
               'slack_hi', 'slack_lo'):
        if vn in ds:
            df[vn] = ds[vn].values.astype(bool)
    if 'phase' in ds:
        # Decode integer flag -> string label
        flags = ds['phase'].attrs.get('flag_values', [])
        meanings = ds['phase'].attrs.get('flag_meanings', '')
        if isinstance(meanings, str):
            meanings = meanings.split()
        flag_map = dict(zip(flags, meanings))
        df['phase'] = [flag_map.get(int(v), 'unclassified')
                       for v in ds['phase'].values]
    ds.close()
    return df.sort_values('time').reset_index(drop=True)


def annotate_detections(det_csv, phase_df, out_path):
    """Add phase / is_flood / is_spring / season columns per detection."""
    df = pd.read_csv(det_csv)
    if 'time' not in df.columns:
        sys.exit(f'ERROR: {det_csv.name} has no "time" column. '
                 'Re-run run_swirl_roms.py with the updated version that '
                 'writes ROMS timestamps to the CSV.')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # Nearest-time merge (asof) — robust to skipped snapshots
    merged = pd.merge_asof(df, phase_df, on='time', direction='nearest',
                           tolerance=pd.Timedelta('30min'))

    n_unmatched = merged['is_flood'].isna().sum() if 'is_flood' in merged \
        else 0
    if n_unmatched:
        print(f'  WARN: {n_unmatched} detections had no phase match '
              'within 30 min')

    # Add season tag
    merged['season'] = [_season_of(np.datetime64(t)) for t in merged['time']]

    merged.to_csv(out_path, index=False)
    print(f'Saved: {out_path}')
    return merged


def summarize_tracks(annotated_det_df):
    """
    Per-track aggregates from a fully-annotated detection DataFrame.
    Used to refresh _tracks.csv with phase/season fractions.
    """
    rows = []
    for tid, g in annotated_det_df.groupby('track_id'):
        rec = dict(track_id=int(tid), n_detections=len(g))
        for col in ('is_flood', 'is_spring'):
            if col in g:
                rec[f'frac_{col}'] = float(g[col].mean())
        if 'phase' in g:
            rec['dominant_phase'] = g['phase'].mode().iloc[0]
        if 'season' in g:
            valid = g['season'].dropna()
            rec['dominant_season'] = (valid.mode().iloc[0]
                                      if len(valid) else None)
        rows.append(rec)
    return pd.DataFrame(rows)


def annotate_tracks(tracks_csv, det_annotated, out_path):
    """
    Merge per-track phase/season aggregates into the tracks CSV.
    """
    tracks = pd.read_csv(tracks_csv)
    agg = summarize_tracks(det_annotated)
    merged = tracks.merge(agg, on='track_id', how='left',
                          suffixes=('', '_new'))
    # Prefer the freshly-computed columns where they overlap
    for col in agg.columns:
        if col == 'track_id':
            continue
        new_col = f'{col}_new'
        if new_col in merged.columns:
            merged[col] = merged[new_col]
            merged.drop(columns=[new_col], inplace=True)
    merged.to_csv(out_path, index=False)
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
def main():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_path = Path(args.phase_file)
    if not phase_path.is_file():
        sys.exit(f'ERROR: phase file not found: {phase_path}\n'
                 f'Run tide_phase/extract_zeta_ts.py and '
                 f'tide_phase/compute_tide_phases.py first.')
    print(f'Loading phase file: {phase_path}')
    phase_df = load_phase_dataframe(phase_path)
    print(f'  {len(phase_df)} timesteps, '
          f'{phase_df["time"].iloc[0]} -> {phase_df["time"].iloc[-1]}')

    det_annotated = None
    for csv_path in args._explicit_csvs:
        if not csv_path.is_file():
            print(f'  SKIP (not found): {csv_path}')
            continue
        is_tracks = csv_path.name.endswith('_tracks.csv')
        if is_tracks:
            if det_annotated is None:
                # Need detections to compute aggregates; try sibling _tracked
                sib = csv_path.parent / csv_path.name.replace(
                    '_tracks.csv', '_tracked.csv')
                if not sib.is_file():
                    print(f'  WARN: cannot annotate {csv_path.name} without '
                          f'{sib.name} (skipping).')
                    continue
                # Annotate detections in-memory only
                det_df = pd.read_csv(sib)
                if 'time' in det_df.columns:
                    det_df['time'] = pd.to_datetime(det_df['time'])
                    det_df = det_df.sort_values('time').reset_index(drop=True)
                    det_annotated = pd.merge_asof(
                        det_df, phase_df, on='time', direction='nearest',
                        tolerance=pd.Timedelta('30min'))
                    det_annotated['season'] = [
                        _season_of(np.datetime64(t))
                        for t in det_annotated['time']]
                else:
                    print(f'  WARN: {sib.name} lacks "time" column; '
                          'cannot aggregate.')
                    continue
            out_path = out_dir / csv_path.name.replace(
                '_tracks.csv', '_tracks_with_phase.csv')
            annotate_tracks(csv_path, det_annotated, out_path)
        else:
            out_path = out_dir / csv_path.name.replace(
                '.csv', '_with_phase.csv')
            det_annotated = annotate_detections(csv_path, phase_df, out_path)

    print('Done.')


if __name__ == '__main__':
    main()
