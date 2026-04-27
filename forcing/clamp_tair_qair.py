"""
Clamp Tair and Qair in existing ROMS atm forcing NetCDF files in-place,
matching the limits enforced by the atm02 forcing generator:

    Tair: clipped to [-20, 45]  (deg C)
    Qair: clipped to [  0, 100] (%)

This is intended as a one-off fix for atm00 days where WRF produced
unphysical Tair/Qair values that cause ROMS bulk-flux blowups.

Usage:

    # one folder
    python clamp_tair_qair.py \
        -d /dat1/dakotamm/LO_output/forcing/wb1/f2025.04.29/atm00

    # date range
    python clamp_tair_qair.py \
        -base /dat1/dakotamm/LO_output/forcing/wb1 \
        -frc atm00 -d0 2025.04.28 -d1 2025.05.05

Add --dry-run to only report what would change without writing.

The script writes to a sibling ".clamped.tmp" file then atomically
replaces the original, so an interrupted run won't leave a half-written
NetCDF in place.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr


LIMITS = {
    'Tair': (-20.0, 45.0),
    'Qair': (0.0, 100.0),
}


def clamp_file(fn: Path, dry_run: bool = False) -> bool:
    """Clamp Tair or Qair in `fn` if applicable. Returns True if file
    was (or would be) modified."""
    vn = fn.stem  # 'Tair' or 'Qair'
    if vn not in LIMITS:
        return False
    if not fn.is_file():
        print('  MISSING: %s' % fn)
        return False

    lo, hi = LIMITS[vn]
    with xr.open_dataset(fn) as ds:
        ds.load()  # bring into memory so we can close & overwrite

    if vn not in ds.data_vars:
        print('  %s not in %s' % (vn, fn))
        return False

    arr = ds[vn].values
    n_lo = int(np.sum(arr < lo))
    n_hi = int(np.sum(arr > hi))
    n_nan = int(np.sum(~np.isfinite(arr)))

    if n_lo == 0 and n_hi == 0 and n_nan == 0:
        print('  %s OK  (min=%.3f max=%.3f)' %
              (fn.name, float(np.nanmin(arr)), float(np.nanmax(arr))))
        return False

    print('  %s clamp: below=%d above=%d nan=%d  '
          '(orig min=%.3f max=%.3f) -> [%.1f, %.1f]'
          % (fn.name, n_lo, n_hi, n_nan,
             float(np.nanmin(arr)), float(np.nanmax(arr)), lo, hi))

    if dry_run:
        return True

    new = arr.copy()
    # Replace any non-finite values with the nearer bound midpoint
    # (shouldn't normally happen, but be safe).
    if n_nan > 0:
        new = np.where(np.isfinite(new), new, 0.5 * (lo + hi))
    new = np.clip(new, lo, hi)

    # Preserve attrs/dims/coords; only replace the values.
    ds[vn].values[...] = new

    # Preserve original encoding (dtype, _FillValue, zlib, etc.) where present.
    enc = {}
    for v in ds.data_vars:
        e = {k: ds[v].encoding[k]
             for k in ('dtype', '_FillValue', 'zlib', 'complevel',
                       'chunksizes', 'shuffle')
             if k in ds[v].encoding}
        if e:
            enc[v] = e

    tmp = fn.with_suffix('.clamped.tmp')
    if tmp.exists():
        tmp.unlink()
    ds.to_netcdf(tmp, encoding=enc)
    ds.close()
    tmp.replace(fn)  # atomic on POSIX
    return True


def process_folder(folder: Path, dry_run: bool = False) -> int:
    print('\n========== %s ==========' % folder)
    if not folder.is_dir():
        print('  (folder does not exist)')
        return 0
    n_changed = 0
    for vn in LIMITS.keys():
        if clamp_file(folder / (vn + '.nc'), dry_run=dry_run):
            n_changed += 1
    return n_changed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dir', type=str, default=None,
                   help='single forcing folder to clamp '
                        '(e.g. .../wb1/f2025.04.29/atm00)')
    p.add_argument('-base', type=str, default=None,
                   help='base forcing folder (e.g. .../forcing/wb1) '
                        'used together with -frc, -d0, -d1')
    p.add_argument('-frc', type=str, default='atm00',
                   help='forcing subfolder name (default atm00)')
    p.add_argument('-d0', type=str, default=None,
                   help='start date YYYY.MM.DD (inclusive)')
    p.add_argument('-d1', type=str, default=None,
                   help='end date YYYY.MM.DD (inclusive)')
    p.add_argument('--dry-run', action='store_true',
                   help='report only; do not modify files')
    args = p.parse_args()

    total = 0
    if args.dir is not None:
        total += process_folder(Path(args.dir), dry_run=args.dry_run)
    elif args.base is not None and args.d0 is not None and args.d1 is not None:
        base = Path(args.base)
        dt0 = datetime.strptime(args.d0, '%Y.%m.%d')
        dt1 = datetime.strptime(args.d1, '%Y.%m.%d')
        dt = dt0
        while dt <= dt1:
            ds = dt.strftime('%Y.%m.%d')
            total += process_folder(base / ('f' + ds) / args.frc,
                                    dry_run=args.dry_run)
            dt += timedelta(days=1)
    else:
        p.error('Provide either -d, or all of -base/-d0/-d1.')

    tag = ' (dry run)' if args.dry_run else ''
    print('\n--- done%s; files modified: %d ---' % (tag, total))


if __name__ == '__main__':
    main()
