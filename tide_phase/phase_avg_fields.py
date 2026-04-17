"""
Average ROMS fields grouped by tidal phase.

Reads tide phase labels from compute_tide_phases.py output, then loops
through ROMS his or avg files and accumulates phase-grouped averages
of user-selected variables.

Replaces the manual approach in 20260130b.py with automated phase grouping.

Usage examples
--------------
    # Depth-averaged u,v,salt from avg files for Penn Cove phases:
    python phase_avg_fields.py -gtx wb1_r0_xn11b -ctag pc0 \
        -sect_name pc0 -0 2017.09.01 -1 2017.09.30 -file_type avg \
        -vn_list u,v,salt

    # From his files with a spatial subset (box job):
    python phase_avg_fields.py -gtx wb1_r0_xn11b -ctag pc0 \
        -sect_name pc0 -0 2017.09.01 -1 2017.09.30 -file_type his \
        -vn_list u,v,salt,oxygen

    # Using cas7 grid with c0 collection:
    python phase_avg_fields.py -gtx cas7_trapsV00_meV00 -ctag c0 \
        -sect_name ai1 -0 2017.07.04 -1 2017.07.06 -file_type avg \
        -vn_list salt,temp
"""

import argparse
import sys
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from time import time as timer

from lo_tools import Lfun

import tide_phase_fun as tpf


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
    # Phase label source
    parser.add_argument('-ctag', '--collection_tag', type=str, default=None)
    parser.add_argument('-sect_name', type=str, default=None,
                        help='Section name used for phase detection')
    parser.add_argument('-label', type=str, default=None,
                        help='Custom label (for point-based phase files)')
    # File type
    parser.add_argument('-file_type', type=str, default='avg',
                        choices=['avg', 'his'],
                        help='ROMS file type: avg or his')
    # Variables to average
    parser.add_argument('-vn_list', type=str, default='u,v,salt',
                        help='Comma-separated variable names, e.g. u,v,salt,temp,oxygen')
    # Phases to compute (default: all four combos)
    parser.add_argument('-phases', type=str,
                        default='spring_flood,spring_ebb,neap_flood,neap_ebb',
                        help='Comma-separated phase names to average over')
    parser.add_argument('-test', '--testing', type=Lfun.boolean_string,
                        default=False)

    args = parser.parse_args()

    if args.sect_name is not None:
        args.label = args.sect_name
    elif args.label is None:
        print('ERROR: provide -sect_name or -label')
        sys.exit(1)

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
# File list builders (same as extract_zeta_ts.py)
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
            fn = dir0 / f_string / ('ocean_avg_' + nhiss + '.nc')
            fn_list.append(fn)
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

    # ----- Accumulate phase-averaged fields -----
    # Structure: accumulators[phase_name][vn] = sum array
    #            counts[phase_name] = int
    accumulators = {pn: {} for pn in phase_names}
    counts = {pn: 0 for pn in phase_names}
    grid_info = {}  # store grid coords from first file

    for ii, fn in enumerate(fn_list):
        # Get file time and find matching phase
        file_time = get_fn_time(fn)
        # Find nearest phase label
        idx = np.argmin(np.abs(phase_time - file_time))
        dt_diff = abs((phase_time[idx] - file_time) / np.timedelta64(1, 'h'))
        if dt_diff > 1.5:
            # Skip files that don't have a close phase match
            continue
        file_phase = phase_str[idx]

        if file_phase not in phase_names:
            continue

        # Open and accumulate
        ds = xr.open_dataset(fn)

        for vn in vn_list:
            if vn not in ds:
                continue
            fld = ds[vn].values.squeeze()  # remove time dim

            # Depth-average 3D fields (z is first axis after squeeze)
            if fld.ndim == 3:
                fld_depth_avg = np.nanmean(fld, axis=0)
            elif fld.ndim == 2:
                fld_depth_avg = fld
            else:
                continue

            if vn not in accumulators[file_phase]:
                accumulators[file_phase][vn] = fld_depth_avg.copy().astype(float)
            else:
                accumulators[file_phase][vn] += fld_depth_avg

        # Save grid info from first file
        if not grid_info:
            for gvn in ['lon_rho', 'lat_rho', 'lon_u', 'lat_u',
                         'lon_v', 'lat_v', 'h', 'mask_rho', 'mask_u', 'mask_v']:
                if gvn in ds:
                    grid_info[gvn] = ds[gvn].values

        ds.close()
        counts[file_phase] += 1

        if (ii + 1) % 100 == 0:
            print(f'  processed {ii + 1}/{len(fn_list)} files')

    # ----- Compute means and save -----
    out_dir = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
               / ('phase_avg_' + ds0 + '_' + ds1))
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

        # Grid coordinates
        for gvn, gdata in grid_info.items():
            dims = {1: ('d0',), 2: ('d0', 'd1')}
            ds_out[gvn] = (dims[gdata.ndim], gdata)

        # Phase-averaged fields
        for vn in vn_list:
            if vn in accumulators[pn]:
                mean_fld = accumulators[pn][vn] / counts[pn]
                if mean_fld.ndim == 2:
                    ds_out[vn] = (('d0', 'd1'), mean_fld)
                elif mean_fld.ndim == 1:
                    ds_out[vn] = (('d0',), mean_fld)
                ds_out[vn].attrs['long_name'] = f'depth-averaged {vn}, phase={pn}'

        ds_out.to_netcdf(out_fn)
        ds_out.close()
        print(f'    saved: {out_fn}')

    print(f'\nTotal time: {timer() - tt0:.1f} s')
