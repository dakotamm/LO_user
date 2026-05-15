"""
Run Okubo-Weiss detection (run_swirl_roms.py) over each hypoxia event
window from find_hypoxia_events.py, at three vertical layers:
surface, depth-averaged, and bottom (s_lev=0).

For each event row, three subprocess calls are issued (one per layer),
all using lowpass (tide-averaged daily) ROMS output and the Penn Cove
bbox. Outputs land under per-event/per-layer subdirectories so the
runs don't overwrite each other.

Usage (on apogee):
    conda activate loenv
    python run_swirl_hypoxia_events.py -gtx wb1_r0_xn11b -ro 2 -nproc 8

    # Dry-run (print commands only):
    python run_swirl_hypoxia_events.py -gtx wb1_r0_xn11b -ro 2 -dry True

Output layout:
    LOo/swirl/<gtx>/hypoxia_events/event_<id>_<lead>_<end>/<layer>/
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from lo_tools import Lfun


# (cli vel arg, optional s_lev, subdir name)
LAYERS = [
    ('surface',     None, 'surface'),
    ('depth_avg',   None, 'depth_avg'),
    ('depth_level',  0,   'bottom'),
]


def _bool(s):
    return str(s).lower() in ('true', '1', 'yes')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('-gtx', '--gtagex', type=str, default='wb1_r0_xn11b')
    p.add_argument('-mooring', type=str, default='M1')
    p.add_argument('-year', type=int, default=2017)
    p.add_argument('-ro', '--roms_out_num', type=int, default=2,
                   help='Which roms_out path to use (passed to run_swirl_roms).')
    p.add_argument('-nproc', type=int, default=8)
    p.add_argument('-events_csv', type=str, default=None,
                   help='Override events CSV path.')
    p.add_argument('-script', type=str,
                   default=str(Path(__file__).parent / 'run_swirl_roms.py'),
                   help='Path to run_swirl_roms.py')
    p.add_argument('-min_cells', type=int, default=9)
    p.add_argument('-dry', type=_bool, default=False,
                   help='Print commands without executing.')
    p.add_argument('-event_id', type=int, default=None,
                   help='Run only this single event_id.')
    p.add_argument('-layers', type=str, default='all',
                   help="Comma list of layers from {surface,depth_avg,bottom}, "
                        "or 'all'.")
    return p.parse_args()


def select_layers(spec):
    if spec.strip().lower() == 'all':
        return LAYERS
    wanted = {s.strip() for s in spec.split(',')}
    chosen = [L for L in LAYERS if L[2] in wanted]
    if not chosen:
        raise ValueError(f'No valid layers in {spec!r}; '
                         f'choose from surface,depth_avg,bottom.')
    return chosen


def build_command(script, gtx, ro, ds0, ds1, vel, s_lev,
                  out_dir, nproc, min_cells):
    cmd = [
        sys.executable, str(script),
        '-gtx', gtx,
        '-ro', str(ro),
        '-0', ds0,
        '-1', ds1,
        '-method', 'ow',
        '-ftype', 'lowpass',
        '-vel', vel,
        '-penn_cove', 'True',
        '-save', 'True',
        '-no_plot', 'False',
        '-nproc', str(nproc),
        '-min_cells', str(min_cells),
        '-out_dir', str(out_dir),
    ]
    if s_lev is not None:
        cmd += ['-s_lev', str(s_lev)]
    return cmd


def fmt_date(ts):
    return pd.Timestamp(ts).strftime('%Y.%m.%d')


def main():
    args = parse_args()
    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)

    if args.events_csv is not None:
        events_csv = Path(args.events_csv)
    else:
        events_csv = (Ldir['LOo'] / 'swirl' / args.gtagex
                      / f'hypoxia_events_{args.mooring}_{args.year}.csv')
    if not events_csv.exists():
        raise FileNotFoundError(
            f'Events CSV not found: {events_csv}\n'
            f'Run find_hypoxia_events.py first.')

    events = pd.read_csv(events_csv, parse_dates=[
        'event_start', 'event_end', 'lead_start'])
    if args.event_id is not None:
        events = events[events.event_id == args.event_id]
        if events.empty:
            raise ValueError(f'event_id {args.event_id} not in {events_csv}')

    layers = select_layers(args.layers)

    base_out = (Ldir['LOo'] / 'swirl' / args.gtagex / 'hypoxia_events')
    base_out.mkdir(parents=True, exist_ok=True)

    print(f'Driver: {len(events)} event(s) x {len(layers)} layer(s) '
          f'= {len(events) * len(layers)} runs.')
    print(f'Base output: {base_out}')
    print()

    n_run = 0
    n_fail = 0
    for _, row in events.iterrows():
        eid = int(row['event_id'])
        ds0 = fmt_date(row['lead_start'])
        ds1 = fmt_date(row['event_end'])
        event_dir = base_out / f'event_{eid:02d}_{ds0}_{ds1}'

        for vel, s_lev, layer_name in layers:
            out_dir = event_dir / layer_name
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_command(args.script, args.gtagex, args.roms_out_num,
                                ds0, ds1, vel, s_lev,
                                out_dir, args.nproc, args.min_cells)

            print(f'--- event {eid} / {layer_name} ({ds0} -> {ds1}) ---')
            print('  ' + ' '.join(cmd))
            if args.dry:
                continue

            try:
                subprocess.run(cmd, check=True)
                n_run += 1
            except subprocess.CalledProcessError as e:
                n_fail += 1
                print(f'  *** FAILED (exit {e.returncode}) ***')

    if not args.dry:
        print()
        print(f'Done: {n_run} succeeded, {n_fail} failed.')


if __name__ == '__main__':
    main()
