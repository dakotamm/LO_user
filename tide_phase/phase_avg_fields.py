"""
Average ROMS fields grouped by tidal phase.

Reads tide phase labels from compute_tide_phases.py output, then loops
through ROMS his or avg files and accumulates phase-grouped averages
of user-selected variables (depth-averaged for 3D fields).

Usage
-----
    # From his files (instantaneous, 25/day):
    python phase_avg_fields.py -gtx wb1_t0_xn11ab -label penn_cove \
        -0 2024.01.01 -1 2024.06.30 -file_type his -vn_list u,v,salt

    # From avg files (hourly means, 24/day):
    python phase_avg_fields.py -gtx wb1_t0_xn11ab -label penn_cove \
        -0 2024.01.01 -1 2024.06.30 -file_type avg -vn_list u,v,salt
"""

import argparse
import sys
import warnings
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from time import time as timer
from concurrent.futures import ProcessPoolExecutor, as_completed

from lo_tools import Lfun

import tide_phase_fun as tpf


# -----------------------------------------------------------------------
# Worker (top-level so it pickles for ProcessPoolExecutor)
# -----------------------------------------------------------------------
def _process_one_file(args):
    """Open one ROMS file, return (fn, file_time, {vn: 2d depth-avg}, grid).

    grid is a dict of horizontal coord/mask arrays, returned only when
    `want_grid` is True (caller asks the first worker to grab it).
    """
    fn, vn_list, want_grid = args
    out_vns = {}
    grid = {}
    try:
        ds = xr.open_dataset(fn)
    except Exception as e:
        return (str(fn), None, out_vns, grid, f'open failed: {e}')

    ot = ds.ocean_time.values
    t = ot.item() if ot.ndim == 0 else ot[0].item()
    file_time = np.datetime64(t, 'ns')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for vn in vn_list:
            if vn not in ds:
                continue
            fld = ds[vn].values.squeeze()
            if fld.ndim == 3:
                out_vns[vn] = np.nanmean(fld, axis=0).astype(float)
            elif fld.ndim == 2:
                out_vns[vn] = fld.astype(float)

    if want_grid:
        for gvn in ['lon_rho', 'lat_rho', 'lon_u', 'lat_u',
                    'lon_v', 'lat_v', 'h', 'mask_rho',
                    'mask_u', 'mask_v']:
            if gvn in ds:
                grid[gvn] = ds[gvn].values

    ds.close()
    return (str(fn), file_time, out_vns, grid, None)


# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description='Average ROMS fields grouped by tidal phase.')
    parser.add_argument('-gtx', '--gtagex', type=str, required=True)
    parser.add_argument('-ro', '--roms_out_num', type=int, default=0)
    parser.add_argument('-0', '--ds0', type=str, required=True)
    parser.add_argument('-1', '--ds1', type=str, required=True)
    # Phase label source (must match extract_zeta_ts/compute_tide_phases label)
    parser.add_argument('-label', type=str, required=True,
                        help='Label matching the tide_phases output filename')
    parser.add_argument('-file_type', type=str, default='his',
                        choices=['avg', 'his'],
                        help='ROMS file type: avg (24/day means) or his (25/day instantaneous)')
    # Variables to average
    parser.add_argument('-vn_list', type=str, default='u,v,salt',
                        help='Comma-separated variable names, e.g. u,v,salt,temp,oxygen')
    # Phases to compute (default: all four combos)
    parser.add_argument('-phases', type=str,
                        default='spring_flood,spring_ebb,neap_flood,neap_ebb',
                        help='Comma-separated phase names to average over')
    parser.add_argument('-Nproc', type=int, default=10,
                        help='Number of parallel worker processes')
    parser.add_argument('-test', '--testing', type=Lfun.boolean_string,
                        default=False)

    args = parser.parse_args()

    # Build Ldir
    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
    for k, v in vars(args).items():
        if k not in Ldir:
            Ldir[k] = v
    if Ldir['roms_out_num'] > 0:
        Ldir['roms_out'] = Ldir['roms_out' + str(Ldir['roms_out_num'])]

    return Ldir


# -----------------------------------------------------------------------
# File list builders
# -----------------------------------------------------------------------
def get_avg_fn_list(Ldir, ds0, ds1):
    fmt = '%Y.%m.%d'
    dt0 = datetime.strptime(ds0, fmt)
    dt1 = datetime.strptime(ds1, fmt)
    dir0 = Ldir['roms_out'] / Ldir['gtagex']
    fn_list = []
    dt = dt0
    while dt <= dt1:
        f_string = 'f' + dt.strftime(fmt)
        for nhis in range(1, 25):
            nhiss = ('0000' + str(nhis))[-4:]
            fn_list.append(dir0 / f_string / ('ocean_avg_' + nhiss + '.nc'))
        dt += timedelta(days=1)
    return fn_list


def get_his_fn_list(Ldir, ds0, ds1):
    fmt = '%Y.%m.%d'
    dt0 = datetime.strptime(ds0, fmt)
    dt1 = datetime.strptime(ds1, fmt)
    dir0 = Ldir['roms_out'] / Ldir['gtagex']
    fn_list = []
    dt = dt0
    while dt <= dt1:
        f_string = 'f' + dt.strftime(fmt)
        for nhis in range(1, 26):
            nhiss = ('0000' + str(nhis))[-4:]
            fn = dir0 / f_string / ('ocean_his_' + nhiss + '.nc')
            fn_list.append(fn)
        dt += timedelta(days=1)
    return fn_list


def get_fn_time(fn):
    """Extract the timestamp from a ROMS file without loading all data."""
    ds = xr.open_dataset(fn)
    ot = ds.ocean_time.values
    ds.close()
    t = ot.item() if ot.ndim == 0 else ot[0].item()
    return np.datetime64(t, 'ns')


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == '__main__':
    tt0 = timer()
    Ldir = get_args()

    label = Ldir['label']
    ds0 = Ldir['ds0']
    ds1 = Ldir['ds1']
    vn_list = [s.strip() for s in Ldir['vn_list'].split(',')]
    phase_names = [s.strip() for s in Ldir['phases'].split(',')]

    print(f'Variables: {vn_list}')
    print(f'Phases: {phase_names}')
    print(f'File type: {Ldir["file_type"]}')

    # ----- Load phase labels -----
    phase_dir = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
                 / ('tide_phases_' + ds0 + '_' + ds1))
    phase_fn = phase_dir / (label + '.nc')
    if not phase_fn.is_file():
        print(f'ERROR: phase file not found: {phase_fn}')
        print('Run compute_tide_phases.py first.')
        sys.exit(1)

    ds_phase = xr.open_dataset(phase_fn)
    phase_time = ds_phase['time'].values
    phase_int = ds_phase['phase'].values  # int8 encoded
    # Decode phase integers back to names
    flag_meanings = ds_phase['phase'].attrs.get('flag_meanings', '').split()
    flag_values = ds_phase['phase'].attrs.get('flag_values', [])
    val_to_name = dict(zip(flag_values, flag_meanings))
    phase_str = np.array([val_to_name.get(int(v), 'unclassified')
                          for v in phase_int])
    ds_phase.close()

    print(f'Phase labels: {len(phase_time)} timesteps')

    # ----- Build ROMS file list -----
    if Ldir['file_type'] == 'avg':
        fn_list = get_avg_fn_list(Ldir, ds0, ds1)
    else:
        fn_list = get_his_fn_list(Ldir, ds0, ds1)

    fn_list = [fn for fn in fn_list if fn.is_file()]
    if len(fn_list) == 0:
        print('ERROR: no ROMS files found.')
        sys.exit(1)
    print(f'Found {len(fn_list)} {Ldir["file_type"]} files.')

    if Ldir['testing']:
        fn_list = fn_list[:48]

    # ----- Map each ROMS file to its nearest phase label -----
    # Build a time-to-phase lookup
    import pandas as pd
    phase_series = pd.Series(phase_str, index=pd.DatetimeIndex(phase_time))

    # ----- Accumulate phase-averaged fields (parallel I/O) -----
    # Structure: accumulators[phase_name][vn] = sum array
    #            counts[phase_name] = int
    accumulators = {pn: {} for pn in phase_names}
    counts = {pn: 0 for pn in phase_names}
    grid_info = {}

    Nproc = max(1, int(Ldir['Nproc']))
    print(f'Processing {len(fn_list)} files with Nproc={Nproc} ...')

    # Build worker arg list: only the first task pulls grid_info
    work = [(fn, vn_list, ii == 0) for ii, fn in enumerate(fn_list)]

    n_done = 0
    n_skipped = 0
    n_errors = 0
    with ProcessPoolExecutor(max_workers=Nproc) as ex:
        for result in ex.map(_process_one_file, work, chunksize=4):
            fn_str, file_time, out_vns, grid, err = result
            n_done += 1
            if err is not None:
                n_errors += 1
                print(f'  [warn] {fn_str}: {err}')
                continue
            if grid and not grid_info:
                grid_info = grid

            # Match this file's time to a phase label
            idx = np.argmin(np.abs(phase_time - file_time))
            dt_diff = abs((phase_time[idx] - file_time)
                          / np.timedelta64(1, 'h'))
            if dt_diff > 1.5:
                n_skipped += 1
                continue
            file_phase = phase_str[idx]
            if file_phase not in phase_names:
                n_skipped += 1
                continue

            for vn, arr in out_vns.items():
                if vn not in accumulators[file_phase]:
                    accumulators[file_phase][vn] = arr.copy()
                else:
                    accumulators[file_phase][vn] += arr
            counts[file_phase] += 1

            if n_done % 100 == 0:
                print(f'  processed {n_done}/{len(fn_list)} files '
                      f'({timer() - tt0:.1f} s)')

    print(f'  done: {n_done} files, {n_skipped} skipped, '
          f'{n_errors} errors ({timer() - tt0:.1f} s)')

    # ----- Compute means and save -----
    out_dir = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
               / ('phase_avg_' + ds0 + '_' + ds1 + '_' + Ldir['file_type']))
    Lfun.make_dir(out_dir)

    for pn in phase_names:
        if counts[pn] == 0:
            print(f'  {pn}: no timesteps — skipping')
            continue

        print(f'  {pn}: {counts[pn]} timesteps averaged')
        out_fn = out_dir / (label + '_' + pn + '.nc')

        ds_out = xr.Dataset(attrs={
            'gridname': Ldir['gridname'],
            'gtagex': Ldir['gtagex'],
            'ds0': ds0,
            'ds1': ds1,
            'phase': pn,
            'label': label,
            'file_type': Ldir['file_type'],
            'n_timesteps': counts[pn],
        })

        # Grid coordinates — use proper ROMS dim names since rho/u/v grids
        # have different sizes.
        gvn_dims = {
            'lon_rho':  ('eta_rho', 'xi_rho'),
            'lat_rho':  ('eta_rho', 'xi_rho'),
            'mask_rho': ('eta_rho', 'xi_rho'),
            'h':        ('eta_rho', 'xi_rho'),
            'lon_u':    ('eta_u', 'xi_u'),
            'lat_u':    ('eta_u', 'xi_u'),
            'mask_u':   ('eta_u', 'xi_u'),
            'lon_v':    ('eta_v', 'xi_v'),
            'lat_v':    ('eta_v', 'xi_v'),
            'mask_v':   ('eta_v', 'xi_v'),
        }
        for gvn, gdata in grid_info.items():
            if gvn in gvn_dims:
                ds_out[gvn] = (gvn_dims[gvn], gdata)

        # Map each variable to the right horizontal grid
        vn_dims = {
            'u': ('eta_u', 'xi_u'),
            'v': ('eta_v', 'xi_v'),
        }
        # Phase-averaged fields
        for vn in vn_list:
            if vn in accumulators[pn]:
                mean_fld = accumulators[pn][vn] / counts[pn]
                hdims = vn_dims.get(vn, ('eta_rho', 'xi_rho'))
                if mean_fld.ndim == 2:
                    ds_out[vn] = (hdims, mean_fld)
                elif mean_fld.ndim == 1:
                    ds_out[vn] = ((hdims[0],), mean_fld)
                ds_out[vn].attrs['long_name'] = f'depth-averaged {vn}, phase={pn}'

        ds_out.to_netcdf(out_fn)
        ds_out.close()
        print(f'    saved: {out_fn}')

    print(f'\nTotal time: {timer() - tt0:.1f} s')
