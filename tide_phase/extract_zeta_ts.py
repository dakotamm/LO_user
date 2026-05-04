"""
Extract a zeta (SSH) time series at a single (lon, lat) point from
ROMS output files.

Simplified: point-based only, no TEF/section dependencies.

Usage
-----
    # Using a known label (coords looked up from tide_phase_fun.LOCATIONS):
    python extract_zeta_ts.py -gtx wb1_t0_xn11ab \
        -0 2024.01.01 -1 2024.06.30 -label penn_cove

    # Or pass coords explicitly:
    python extract_zeta_ts.py -gtx wb1_t0_xn11ab \
        -0 2024.01.01 -1 2024.06.30 \
        -lon -122.7 -lat 48.23 -label my_point
"""

import argparse
import sys
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from time import time as timer

from lo_tools import Lfun

import tide_phase_fun as tpf


# -----------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description='Extract zeta time series at a (lon, lat) point.')
    parser.add_argument('-gtx', '--gtagex', type=str, required=True)
    parser.add_argument('-ro', '--roms_out_num', type=int, default=0)
    parser.add_argument('-0', '--ds0', type=str, required=True)
    parser.add_argument('-1', '--ds1', type=str, required=True)
    parser.add_argument('-lon', type=float, default=None,
                        help='Longitude (optional if -label is a known location)')
    parser.add_argument('-lat', type=float, default=None,
                        help='Latitude (optional if -label is a known location)')
    parser.add_argument('-label', type=str, default=None,
                        help='Output filename label. If matches tide_phase_fun.LOCATIONS, '
                             'coords are looked up from there.')
    parser.add_argument('-test', '--testing', type=Lfun.boolean_string,
                        default=False)

    args = parser.parse_args()

    # Resolve lon/lat from -label if not given explicitly
    if args.lon is None or args.lat is None:
        if args.label is None:
            print('ERROR: provide -lon and -lat, or -label matching a known location.')
            print(f'Known labels: {sorted(tpf.LOCATIONS.keys())}')
            sys.exit(1)
        try:
            lon_lookup, lat_lookup = tpf.get_location(args.label)
        except KeyError as e:
            print(f'ERROR: {e}')
            sys.exit(1)
        args.lon = lon_lookup if args.lon is None else args.lon
        args.lat = lat_lookup if args.lat is None else args.lat

    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
    for k, v in vars(args).items():
        if k not in Ldir:
            Ldir[k] = v
    if Ldir['roms_out_num'] > 0:
        Ldir['roms_out'] = Ldir['roms_out' + str(Ldir['roms_out_num'])]

    if Ldir['label'] is None:
        Ldir['label'] = f'lon{Ldir["lon"]:.4f}_lat{Ldir["lat"]:.4f}'

    return Ldir


# -----------------------------------------------------------------------
def get_fn_list(Ldir, ds0, ds1):
    """Hourly his files (25/day) between ds0 and ds1."""
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
            fn_list.append(dir0 / f_string / ('ocean_his_' + nhiss + '.nc'))
        dt += timedelta(days=1)
    return fn_list


def find_nearest_ij(lon_rho, lat_rho, target_lon, target_lat):
    dist = (lon_rho - target_lon)**2 + (lat_rho - target_lat)**2
    j, i = np.unravel_index(np.argmin(dist), dist.shape)
    return int(j), int(i)


# -----------------------------------------------------------------------
if __name__ == '__main__':
    tt0 = timer()
    Ldir = get_args()

    fn_list = [fn for fn in get_fn_list(Ldir, Ldir['ds0'], Ldir['ds1'])
               if fn.is_file()]
    if len(fn_list) == 0:
        print('ERROR: no ROMS files found.')
        sys.exit(1)
    print(f'Found {len(fn_list)} his files.')
    print(f'  First: {fn_list[0]}')
    print(f'  Last:  {fn_list[-1]}')

    if Ldir['testing']:
        fn_list = fn_list[:48]

    # Find nearest grid cell from first file
    ds0 = xr.open_dataset(fn_list[0])
    j_pt, i_pt = find_nearest_ij(ds0.lon_rho.values, ds0.lat_rho.values,
                                  Ldir['lon'], Ldir['lat'])
    actual_lon = float(ds0.lon_rho.values[j_pt, i_pt])
    actual_lat = float(ds0.lat_rho.values[j_pt, i_pt])
    actual_h = float(ds0.h.values[j_pt, i_pt]) if 'h' in ds0 else np.nan
    ds0.close()
    print(f'Target ({Ldir["lon"]}, {Ldir["lat"]}) -> '
          f'grid (j={j_pt}, i={i_pt}) at ({actual_lon:.4f}, {actual_lat:.4f}), '
          f'h={actual_h:.1f} m')

    # Loop and extract
    zeta_list = []
    time_list = []
    for ii, fn in enumerate(fn_list):
        ds = xr.open_dataset(fn)
        ot = ds.ocean_time.values
        zeta_2d = ds.zeta.values.squeeze()
        ds.close()
        zeta_list.append(float(zeta_2d[j_pt, i_pt]))
        time_list.append(ot.item() if ot.ndim == 0 else ot[0].item())
        if (ii + 1) % 200 == 0:
            print(f'  processed {ii + 1}/{len(fn_list)} files')

    zeta_arr = np.array(zeta_list)
    time_arr = np.array(time_list, dtype='datetime64[ns]')

    # Save
    out_dir = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
               / ('zeta_ts_' + Ldir['ds0'] + '_' + Ldir['ds1']))
    Lfun.make_dir(out_dir)
    out_fn = out_dir / (Ldir['label'] + '.nc')

    ds_out = xr.Dataset(
        coords={'time': time_arr},
        attrs={
            'gridname': Ldir['gridname'],
            'gtagex': Ldir['gtagex'],
            'ds0': Ldir['ds0'],
            'ds1': Ldir['ds1'],
            'label': Ldir['label'],
            'file_type': 'his',
            'lon_target': Ldir['lon'],
            'lat_target': Ldir['lat'],
            'lon_actual': actual_lon,
            'lat_actual': actual_lat,
            'h': actual_h,
            'j_index': j_pt,
            'i_index': i_pt,
        },
    )
    ds_out['zeta'] = ('time', zeta_arr)
    ds_out['zeta'].attrs['units'] = 'm'
    ds_out['zeta'].attrs['long_name'] = 'sea surface height'

    ds_out.to_netcdf(out_fn)
    ds_out.close()

    print(f'\nSaved: {out_fn}')
    print(f'  zeta shape: {zeta_arr.shape}')
    print(f'  time range: {time_arr[0]} to {time_arr[-1]}')
    print(f'Total time: {timer() - tt0:.1f} s')
