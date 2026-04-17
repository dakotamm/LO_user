"""
Extract a zeta (SSH) time series from ROMS output files.

Extracts mean zeta along a TEF section, or at a user-specified (lon, lat) point.
Can also pull qnet from an existing TEF bulk file.

Designed to run on apogee (or wherever ROMS output lives).

Usage examples
--------------
    # Section-mean zeta from avg files:
    python extract_zeta_ts.py -gtx wb1_r0_xn11b -ctag pc0 -sect_name pc0 \
        -0 2017.09.01 -1 2017.09.30 -file_type avg

    # Single-point zeta from his files:
    python extract_zeta_ts.py -gtx wb1_r0_xn11b -0 2017.09.01 -1 2017.09.30 \
        -file_type his -lon -122.7 -lat 48.23

    # Also grab qnet from existing bulk output:
    python extract_zeta_ts.py -gtx wb1_r0_xn11b -ctag pc0 -sect_name pc0 \
        -0 2017.09.01 -1 2017.09.30 -file_type avg -get_bulk True
"""

import argparse
import sys
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from time import time as timer

from lo_tools import Lfun, zfun


# -----------------------------------------------------------------------
# Argument parsing  (mirrors extract_argfun pattern but self-contained)
# -----------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description='Extract zeta time series from ROMS output.')
    parser.add_argument('-gtx', '--gtagex', type=str, required=True,
                        help='e.g. wb1_r0_xn11b or cas7_trapsV00_meV00')
    parser.add_argument('-ro', '--roms_out_num', type=int, default=0)
    parser.add_argument('-0', '--ds0', type=str, required=True,
                        help='Start date, e.g. 2017.09.01')
    parser.add_argument('-1', '--ds1', type=str, required=True,
                        help='End date, e.g. 2017.09.30')
    # Section-based extraction
    parser.add_argument('-ctag', '--collection_tag', type=str, default=None,
                        help='TEF collection tag, e.g. pc0, c0')
    parser.add_argument('-sect_name', type=str, default=None,
                        help='Section name within the collection, e.g. pc0')
    # Point-based extraction
    parser.add_argument('-lon', type=float, default=None,
                        help='Longitude for single-point extraction')
    parser.add_argument('-lat', type=float, default=None,
                        help='Latitude for single-point extraction')
    # File type
    parser.add_argument('-file_type', type=str, default='avg',
                        choices=['avg', 'his'],
                        help='ROMS file type: avg or his')
    # Optional: also load qnet from existing bulk output
    parser.add_argument('-get_bulk', type=Lfun.boolean_string, default=False,
                        help='Also load qnet/qprism from existing bulk output')
    parser.add_argument('-test', '--testing', type=Lfun.boolean_string,
                        default=False)

    args = parser.parse_args()

    # Validate: need either section info or point info
    has_section = (args.collection_tag is not None) and (args.sect_name is not None)
    has_point = (args.lon is not None) and (args.lat is not None)
    if not has_section and not has_point:
        print('ERROR: provide either (-ctag + -sect_name) or (-lon + -lat)')
        sys.exit(1)

    # Build Ldir
    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
    for k, v in vars(args).items():
        if k not in Ldir:
            Ldir[k] = v
    if Ldir['roms_out_num'] > 0:
        Ldir['roms_out'] = Ldir['roms_out' + str(Ldir['roms_out_num'])]

    return Ldir, has_section, has_point


# -----------------------------------------------------------------------
# File list builders
# -----------------------------------------------------------------------
def get_avg_fn_list(Ldir, ds0, ds1):
    """All 24 hourly avg files per day."""
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
    """History files: ocean_his_0001 .. ocean_his_0025 per day."""
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


# -----------------------------------------------------------------------
# Grid helpers
# -----------------------------------------------------------------------
def find_nearest_ij(lon_rho, lat_rho, target_lon, target_lat):
    """Return (j, i) indices of the nearest rho-grid point."""
    dist = (lon_rho - target_lon)**2 + (lat_rho - target_lat)**2
    j, i = np.unravel_index(np.argmin(dist), dist.shape)
    return int(j), int(i)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == '__main__':
    tt0 = timer()
    Ldir, has_section, has_point = get_args()

    # Build file list
    if Ldir['file_type'] == 'avg':
        fn_list = get_avg_fn_list(Ldir, Ldir['ds0'], Ldir['ds1'])
    else:
        fn_list = get_his_fn_list(Ldir, Ldir['ds0'], Ldir['ds1'])

    # Filter to files that exist
    fn_list = [fn for fn in fn_list if fn.is_file()]
    if len(fn_list) == 0:
        print('ERROR: no ROMS files found.')
        sys.exit(1)
    print(f'Found {len(fn_list)} {Ldir["file_type"]} files.')
    print(f'  First: {fn_list[0]}')
    print(f'  Last:  {fn_list[-1]}')

    if Ldir['testing']:
        fn_list = fn_list[:48]  # two days

    # ----- Determine extraction indices -----
    if has_section:
        gctag = Ldir['gridname'] + '_' + Ldir['collection_tag']
        tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'
        sect_df_fn = tef2_dir / ('sect_df_' + gctag + '.p')
        sect_df = pd.read_pickle(sect_df_fn)
        # Select rows for the requested section
        sn_mask = sect_df.sn == Ldir['sect_name']
        if sn_mask.sum() == 0:
            print(f"ERROR: section '{Ldir['sect_name']}' not found in {sect_df_fn}")
            sys.exit(1)
        sect_rows = sect_df[sn_mask]
        # Indices: average zeta at rho points flanking the section
        jrp = sect_rows.jrp.values
        irp = sect_rows.irp.values
        jrm = sect_rows.jrm.values
        irm = sect_rows.irm.values
        extract_mode = 'section'
        label = Ldir['sect_name']
        print(f'Extracting section-mean zeta for "{label}" ({len(jrp)} points)')
    else:
        # Point-based: find nearest grid cell
        ds0 = xr.open_dataset(fn_list[0])
        lon_rho = ds0.lon_rho.values
        lat_rho = ds0.lat_rho.values
        ds0.close()
        j_pt, i_pt = find_nearest_ij(lon_rho, lat_rho,
                                       Ldir['lon'], Ldir['lat'])
        extract_mode = 'point'
        label = f'lon{Ldir["lon"]:.4f}_lat{Ldir["lat"]:.4f}'
        print(f'Extracting point zeta at ({Ldir["lon"]}, {Ldir["lat"]}) '
              f'→ grid (j={j_pt}, i={i_pt})')

    # ----- Loop through files and extract zeta -----
    zeta_list = []
    time_list = []
    for ii, fn in enumerate(fn_list):
        ds = xr.open_dataset(fn)
        ot = ds.ocean_time.values  # datetime64
        zeta_2d = ds.zeta.values.squeeze()  # (eta_rho, xi_rho)
        ds.close()

        if extract_mode == 'section':
            # Average zeta at rho points flanking the section (same as tef2)
            z_sect = (zeta_2d[jrp, irp] + zeta_2d[jrm, irm]) / 2.0
            zeta_list.append(np.nanmean(z_sect))
        else:
            zeta_list.append(float(zeta_2d[j_pt, i_pt]))
        time_list.append(ot.item() if ot.ndim == 0 else ot[0].item())

        if (ii + 1) % 100 == 0:
            print(f'  processed {ii + 1}/{len(fn_list)} files')

    zeta_arr = np.array(zeta_list)
    time_arr = np.array(time_list, dtype='datetime64[ns]')

    # ----- Optionally load qnet / qprism from bulk output -----
    qnet_arr = None
    qprism_arr = None
    qnet_time = None
    if Ldir['get_bulk'] and has_section:
        bulk_dir = (Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'tef2'
                    / ('bulk_avg_' + Ldir['ds0'] + '_' + Ldir['ds1']))
        bulk_fn = bulk_dir / (Ldir['sect_name'] + '.nc')
        if bulk_fn.is_file():
            dsb = xr.open_dataset(bulk_fn)
            if 'qnet' in dsb:
                qnet_arr = dsb.qnet.values
            if 'qprism' in dsb:
                qprism_arr = dsb.qprism.values
            qnet_time = dsb.time.values
            dsb.close()
            print(f'Loaded bulk data from {bulk_fn}')
        else:
            print(f'WARNING: bulk file not found: {bulk_fn}')

    # ----- Save output -----
    out_dir = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
               / ('zeta_ts_' + Ldir['ds0'] + '_' + Ldir['ds1']))
    Lfun.make_dir(out_dir)

    out_fn = out_dir / (label + '.nc')

    ds_out = xr.Dataset(
        coords={'time': time_arr},
        attrs={
            'gridname': Ldir['gridname'],
            'gtagex': Ldir['gtagex'],
            'ds0': Ldir['ds0'],
            'ds1': Ldir['ds1'],
            'extract_mode': extract_mode,
            'label': label,
            'file_type': Ldir['file_type'],
        },
    )
    ds_out['zeta'] = ('time', zeta_arr)
    ds_out['zeta'].attrs['units'] = 'm'
    ds_out['zeta'].attrs['long_name'] = 'sea surface height'

    # Add bulk variables on their own time axis if available
    if qnet_arr is not None:
        ds_out['qnet_time'] = ('bulk_time', qnet_time)
        ds_out['qnet'] = ('bulk_time', qnet_arr)
        ds_out['qnet'].attrs['units'] = 'm3/s'
    if qprism_arr is not None:
        ds_out['qprism'] = ('bulk_time', qprism_arr)
        ds_out['qprism'].attrs['units'] = 'm3/s'

    ds_out.to_netcdf(out_fn)
    ds_out.close()

    print(f'\nSaved: {out_fn}')
    print(f'  zeta shape: {zeta_arr.shape}')
    print(f'  time range: {time_arr[0]} to {time_arr[-1]}')
    print(f'Total time: {timer() - tt0:.1f} s')
