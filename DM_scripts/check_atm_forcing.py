"""
Inspect an atm forcing folder for out-of-range or NaN values that could
cause a ROMS blowup. Reports per-variable, per-time-step min/max and
flags any time step that falls outside the physically reasonable bounds
that atm02 enforces (Tair in [-20, 45] C, Qair in [0, 100] %).

Run on klone, e.g.:

    python check_atm_forcing.py \
        -d /mmfs1/gscratch/macc/dakotamm/LO_output/forcing/wb1/f2025.05.01/atm00

Or scan a date range under a grid:

    python check_atm_forcing.py \
        -base /mmfs1/gscratch/macc/dakotamm/LO_output/forcing/wb1 \
        -frc atm00 -d0 2025.04.28 -d1 2025.05.05
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr


VAR_LIST = ['Pair', 'rain', 'swrad', 'lwrad_down',
            'Tair', 'Qair', 'Uwind', 'Vwind']

# (lo, hi) physical sanity bounds. None means no check on that side.
BOUNDS = {
    'Pair':       (800.0, 1100.0),     # mbar
    'rain':       (0.0, None),         # kg m-2 s-1
    'swrad':      (-50.0, 1400.0),     # W m-2 (small negative tolerated)
    'lwrad_down': (50.0, 700.0),       # W m-2
    'Tair':       (-20.0, 45.0),       # deg C   <-- atm02 clamps to these
    'Qair':       (0.0, 100.0),        # %       <-- atm02 clamps to these
    'Uwind':      (-75.0, 75.0),       # m s-1
    'Vwind':      (-75.0, 75.0),       # m s-1
}


def check_folder(folder: Path, verbose: bool = True) -> int:
    """Return number of problems found in this folder."""
    nproblems = 0
    print('\n========== %s ==========' % folder)
    if not folder.is_dir():
        print('  (folder does not exist)')
        return 1

    for vn in VAR_LIST:
        fn = folder / (vn + '.nc')
        if not fn.is_file():
            print('  %-12s MISSING' % vn)
            nproblems += 1
            continue

        try:
            ds = xr.open_dataset(fn)
        except Exception as e:
            print('  %-12s OPEN FAILED: %s' % (vn, e))
            nproblems += 1
            continue

        if vn not in ds.data_vars:
            print('  %-12s variable not in dataset' % vn)
            ds.close()
            nproblems += 1
            continue

        arr = ds[vn].values  # (time, eta, xi)
        nan_count = int(np.sum(~np.isfinite(arr)))
        amin = float(np.nanmin(arr)) if arr.size else float('nan')
        amax = float(np.nanmax(arr)) if arr.size else float('nan')

        lo, hi = BOUNDS.get(vn, (None, None))
        bad = False
        msg_extra = ''
        if nan_count > 0:
            bad = True
            msg_extra += ' NaN/inf=%d' % nan_count
        if lo is not None and amin < lo:
            bad = True
        if hi is not None and amax > hi:
            bad = True

        flag = '!!' if bad else '  '
        print('%s %-12s shape=%-22s min=%12.4f  max=%12.4f%s'
              % (flag, vn, str(arr.shape), amin, amax, msg_extra))

        # If bad, point at which time step(s) are guilty
        if bad and arr.ndim >= 1:
            with np.errstate(invalid='ignore'):
                per_t_min = np.nanmin(arr.reshape(arr.shape[0], -1), axis=1)
                per_t_max = np.nanmax(arr.reshape(arr.shape[0], -1), axis=1)
                per_t_nan = np.sum(
                    ~np.isfinite(arr.reshape(arr.shape[0], -1)), axis=1)
            for tt in range(arr.shape[0]):
                t_bad = False
                if per_t_nan[tt] > 0:
                    t_bad = True
                if lo is not None and per_t_min[tt] < lo:
                    t_bad = True
                if hi is not None and per_t_max[tt] > hi:
                    t_bad = True
                if t_bad:
                    print('       t=%2d  min=%12.4f  max=%12.4f  nan=%d'
                          % (tt, per_t_min[tt], per_t_max[tt],
                             int(per_t_nan[tt])))
                    nproblems += 1

        ds.close()

    return nproblems


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dir', type=str, default=None,
                   help='single forcing folder to check '
                        '(e.g. .../wb1/f2025.05.01/atm00)')
    p.add_argument('-base', type=str, default=None,
                   help='base forcing folder (e.g. .../forcing/wb1) '
                        'used together with -frc, -d0, -d1')
    p.add_argument('-frc', type=str, default='atm00',
                   help='forcing subfolder name (default atm00)')
    p.add_argument('-d0', type=str, default=None,
                   help='start date YYYY.MM.DD (inclusive)')
    p.add_argument('-d1', type=str, default=None,
                   help='end date YYYY.MM.DD (inclusive)')
    args = p.parse_args()

    total = 0
    if args.dir is not None:
        total += check_folder(Path(args.dir))
    elif args.base is not None and args.d0 is not None and args.d1 is not None:
        base = Path(args.base)
        dt0 = datetime.strptime(args.d0, '%Y.%m.%d')
        dt1 = datetime.strptime(args.d1, '%Y.%m.%d')
        dt = dt0
        while dt <= dt1:
            ds = dt.strftime('%Y.%m.%d')
            total += check_folder(base / ('f' + ds) / args.frc)
            dt += timedelta(days=1)
    else:
        p.error('Provide either -d, or all of -base/-d0/-d1.')

    print('\n--- done; problems flagged: %d ---' % total)


if __name__ == '__main__':
    main()
