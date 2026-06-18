"""
Fast single-pass mooring extraction at the King County + Ecology station
locations, reading daily lowpassed.nc files directly.

Unlike LO/extract/moor/extract_moor.py (which re-reads every daily file once
PER STATION, i.e. ~Nfiles x Nstations file opens, and needs an ocean_his file
for grid info), this opens each lowpassed.nc EXACTLY ONCE and pulls all station
columns in that single pass. For a big grid (e.g. cas7) this is ~Nstations
times fewer reads. Grid + S-coordinate info and z_rho are taken/computed from
the lowpassed.nc itself, so no ocean_his files are required (handy when the
history and lowpass output live in different roms_out directories).

Output (one file per station, same layout the plotting scripts expect):
    LO_output/extract/<gtx>/moor/<job>/<station>_<ds0>_<ds1>.nc

Run on apogee:
    python 20260617_extract_moorings_fast.py -gtx cas7_t2_xn11b -ro 2 -Nproc 20 > moor.log &

Test (first ~30 files, serial):
    python 20260617_extract_moorings_fast.py -gtx cas7_t2_xn11b -ro 2 -test True
"""

import sys
import argparse
import numpy as np
import xarray as xr
from time import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from lo_tools import Lfun, zfun

# ---- embedded station list (same lon/lat as the wb1 extraction) --------------
STATIONS = {
    'ADM001':       (-122.616699, 48.029999),
    'ADM003':       (-122.481796, 47.879169),
    'PSS019':       (-122.300003, 48.011669),
    'PTH005':       (-122.763298, 48.083328),
    'SAR003':       (-122.489998, 48.108330),
    'SKG003':       (-122.488297, 48.296669),
    'PENNCOVEENT':  (-122.6550, 48.2370),
    'PENNCOVEWEST': (-122.7200, 48.2249),
    'PSUSANBUOY':   (-122.4200, 48.1750),
    'PSUSANENT':    (-122.3300, 48.0600),
    'PSUSANKP':     (-122.4000, 48.1300),
    'Poss DO-2':    (-122.3358, 47.9392),
    'SARATOGACH':   (-122.3690, 48.0440),
    'SARATOGAOP':   (-122.5500, 48.1840),
    'SARATOGARP':   (-122.5500, 48.2400),
}


def sanitize(name):
    return name.replace(' ', '_')


# ---- worker: read all station columns from one lowpassed.nc ------------------
def read_file(fn, eta_idx, xi_idx, var_list):
    """Return (time, {var: (nstn, nt, N)}, zeta (nstn, nt)) for one file."""
    ds = xr.open_dataset(fn)
    eta = xr.DataArray(eta_idx, dims='station')
    xi = xr.DataArray(xi_idx, dims='station')
    t = ds['ocean_time'].values
    cols = {}
    for v in var_list:
        # isel returns dims (ocean_time, s_rho, station) -> (station, nt, N)
        arr = ds[v].isel(eta_rho=eta, xi_rho=xi).values
        cols[v] = np.transpose(arr, (2, 0, 1))
    z = ds['zeta'].isel(eta_rho=eta, xi_rho=xi).values   # (ocean_time, station)
    zeta = np.transpose(z, (1, 0))                        # (station, nt)
    ds.close()
    return t, cols, zeta


def z_rho_column(h, zeta, s_rho, Cs_r, hc, Vtransform):
    """z_rho for one station column. h scalar, zeta (T,). Returns (T, N)."""
    zeta = zeta[:, None]
    s_rho = s_rho[None, :]
    Cs_r = Cs_r[None, :]
    if int(Vtransform) == 1:
        zr0 = (s_rho - Cs_r) * hc + Cs_r * h
        return zr0 + zeta * (1 + zr0 / h)
    else:  # Vtransform == 2 (LO default)
        zr0 = (s_rho * hc + Cs_r * h) / (hc + h)
        return zeta + (zeta + h) * zr0


def nearest_water(iy, ix, mask):
    """Nearest water cell (mask==1) to (iy,ix), searching outward."""
    if mask[iy, ix] == 1:
        return iy, ix
    ny, nx = mask.shape
    for r in range(1, 25):
        best = None
        for jy in range(max(0, iy-r), min(ny, iy+r+1)):
            for jx in range(max(0, ix-r), min(nx, ix+r+1)):
                if mask[jy, jx] == 1:
                    d = (jy-iy)**2 + (jx-ix)**2
                    if best is None or d < best[0]:
                        best = (d, jy, jx)
        if best is not None:
            return best[1], best[2]
    return iy, ix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gtx', '--gtagex', type=str, default='cas7_t2_xn11b')
    parser.add_argument('-ro', '--roms_out_num', type=int, default=2)
    parser.add_argument('-0', '--ds0', type=str, default='2024.01.02')
    parser.add_argument('-1', '--ds1', type=str, default='2025.12.30')
    parser.add_argument('-job', type=str, default='KCEcology_2024_2025')
    parser.add_argument('-vars', type=str,
                        default='salt,temp,oxygen,NO3,NH4,chlorophyll,'
                                'phytoplankton,alkalinity,TIC')
    parser.add_argument('-Nproc', type=int, default=10)
    parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)
    args = parser.parse_args()

    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
    if args.roms_out_num > 0:
        Ldir['roms_out'] = Ldir['roms_out' + str(args.roms_out_num)]
    Ldir['ds0'], Ldir['ds1'] = args.ds0, args.ds1

    var_list = [v.strip() for v in args.vars.split(',') if v.strip()]

    # build the lowpass file list using LO's own logic (paths match -lt lowpass)
    fn_list = Lfun.get_fn_list('lowpass', Ldir, args.ds0, args.ds1)
    fn_list = [fn for fn in fn_list if fn.is_file()]
    if args.testing:
        fn_list = fn_list[:30]

    print(' 20260617_extract_moorings_fast '.center(60, '='))
    print('gtagex   = %s   (roms_out -> %s)' % (args.gtagex, Ldir['roms_out']))
    print('period   = %s to %s' % (args.ds0, args.ds1))
    print('vars     = %s' % var_list)
    print('files    = %d found' % len(fn_list))
    if len(fn_list) == 0:
        print('*** No lowpassed.nc files found. Check -ro / dates / gtx.')
        sys.exit(1)

    # grid + S-coordinate info from the first file (no ocean_his needed)
    g = xr.open_dataset(fn_list[0])
    lon2 = g['lon_rho'].values
    lat2 = g['lat_rho'].values
    h2 = g['h'].values
    mask = g['mask_rho'].values
    s_rho = g['s_rho'].values
    Cs_r = g['Cs_r'].values
    hc = float(g['hc'].values)
    Vtransform = int(g['Vtransform'].values) if 'Vtransform' in g else 2
    Lon = lon2[0, :]
    Lat = lat2[:, 0]
    # keep only requested vars that actually exist in the files
    missing = [v for v in var_list if v not in g.data_vars]
    if missing:
        print('  (vars not in files, skipping: %s)' % missing)
        var_list = [v for v in var_list if v in g.data_vars]
    g.close()

    # station grid indices (nudged to nearest water cell if needed)
    names = list(STATIONS.keys())
    eta_idx, xi_idx, st_lon, st_lat = [], [], [], []
    for nm in names:
        lo, la = STATIONS[nm]
        iy = zfun.find_nearest_ind(Lat, la)
        ix = zfun.find_nearest_ind(Lon, lo)
        iy, ix = nearest_water(iy, ix, mask)
        eta_idx.append(iy); xi_idx.append(ix)
        st_lon.append(float(lon2[iy, ix])); st_lat.append(float(lat2[iy, ix]))
    eta_idx = np.array(eta_idx); xi_idx = np.array(xi_idx)

    # read every file once, in parallel, pulling all station columns
    tt0 = time()
    worker = partial(read_file, eta_idx=eta_idx, xi_idx=xi_idx, var_list=var_list)
    results = []
    if args.testing or args.Nproc <= 1:
        for i, fn in enumerate(fn_list):
            results.append(worker(fn))
            if i % 50 == 0:
                print('  read %d/%d' % (i+1, len(fn_list))); sys.stdout.flush()
    else:
        with ProcessPoolExecutor(max_workers=args.Nproc) as ex:
            for i, r in enumerate(ex.map(worker, fn_list)):
                results.append(r)
                if i % 50 == 0:
                    print('  read %d/%d' % (i+1, len(fn_list))); sys.stdout.flush()
    print('  read all files in %d sec' % (time() - tt0))

    # sort by time and concatenate
    results.sort(key=lambda r: r[0][0])
    ocean_time = np.concatenate([r[0] for r in results])
    data = {v: np.concatenate([r[1][v] for r in results], axis=1) for v in var_list}
    zeta = np.concatenate([r[2] for r in results], axis=1)   # (nstn, T)

    # write one file per station
    out_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'moor' / args.job
    Lfun.make_dir(out_dir)
    for si, nm in enumerate(names):
        h = float(h2[eta_idx[si], xi_idx[si]])
        z_rho = z_rho_column(h, zeta[si], s_rho, Cs_r, hc, Vtransform)  # (T, N)
        ds_out = xr.Dataset(
            data_vars={
                **{v: (('ocean_time', 's_rho'), data[v][si]) for v in var_list},
                'z_rho': (('ocean_time', 's_rho'), z_rho),
                'lon_rho': ((), st_lon[si]),
                'lat_rho': ((), st_lat[si]),
                'h': ((), h),
            },
            coords={'ocean_time': ocean_time, 's_rho': s_rho},
        )
        fn = out_dir / ('%s_%s_%s.nc' % (sanitize(nm), args.ds0, args.ds1))
        ds_out.to_netcdf(fn, unlimited_dims=['ocean_time'])
        print('  wrote %s' % fn.name)

    print('Done. %d stations, %d time records, in %d sec total.'
          % (len(names), len(ocean_time), time() - tt0))


if __name__ == '__main__':
    main()
