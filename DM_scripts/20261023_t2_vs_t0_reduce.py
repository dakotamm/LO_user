"""
Reduce two wb1 runs (default wb1_t0_xn11abbur00 and wb1_t2_xn11abbur00) over the
days BOTH have output, for the t2-vs-t0 comparison plots
(20261023_t2_vs_t0_plot.py).

wb1 ocean_avg files are HOURLY averages (NAVG = NHIS, 24 per day folder), so a
daily mean here is the mean of all ocean_avg_*.nc in the day folder. Each file is
opened exactly once. WET_DRY is on, and ROMS writes a dry cell-hour as 0 for
every variable, so means are taken over wet hours only (surface temp != 0);
including the zeros scaled every variable on the mudflats by the wet fraction. A day is used only if both runs have >= -navg avg files
in it (so a day still being copied to apogee is skipped).

Saved per run (same day axis for both):
  - daily-mean full-depth columns at the 15 KC + Ecology stations
    (salt, temp, oxygen, NO3, NH4, chlorophyll, phytoplankton, TIC, alkalinity)
    plus zeta and z_rho
  - daily-mean bottom-layer oxygen map
  - monthly means of the bottom and surface layer for salt, temp, oxygen, NO3,
    NH4, chlorophyll, and the number of days per month each cell was hypoxic
    (daily-mean DO < 2 mg/L anywhere in the column)
Grid fields (h, mask_rho, lon/lat) come from the run's own output.

Output:
    LO_output/DM_outs/20261023_t2_vs_t0/reduce_<gtx>_<day0>_<day1>.nc

Run on apogee:
    python 20261023_t2_vs_t0_reduce.py -gtxs wb1_t0_xn11abbur00,wb1_t2_xn11abbur00 -ro 2 -Nproc 20 > t2t0_reduce.log &

Test (first 2 common days, serial):
    python 20261023_t2_vs_t0_reduce.py -ro 2 -test True
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from time import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor

from lo_tools import Lfun, zfun, zrfun

OUT_FOLDER = '20261023_t2_vs_t0'

# same station positions as the 2026.06 KC + Ecology mooring extractions
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
STN_VARS = ['salt', 'temp', 'oxygen', 'NO3', 'NH4', 'chlorophyll',
            'phytoplankton', 'TIC', 'alkalinity']
LAYER_VARS = ['salt', 'temp', 'oxygen', 'NO3', 'NH4', 'chlorophyll']
HYP_UM = 2.0 / (32.0 / 1000.0)   # 2 mg/L = 62.5 uM


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


def avg_files(day_dir):
    return sorted(day_dir.glob('ocean_avg_*.nc'))


def reduce_day(task):
    """Daily means for one (run, day). Returns a dict ({"error": msg} if a file fails).

    With WET_DRY, ROMS writes a dry cell-hour into ocean_avg as 0 for every
    variable in the column (zeta too). Those hours are left out: a cell-hour
    counts as wet when its surface temp != 0. Cells dry all day come out NaN.
    """
    fn_list, eta_idx, xi_idx, stn_vars, layer_vars = task
    try:
        stn = None
        for fn in fn_list:
            d = nc.Dataset(fn)
            if stn is None:
                N = d.dimensions['s_rho'].size
                stn = {v: np.zeros((len(eta_idx), N)) for v in stn_vars}
                zeta = np.zeros(len(eta_idx))
                nwet_stn = np.zeros(len(eta_idx))
                lay = {v: 0.0 for v in layer_vars}
                oxy3 = 0.0
                nwet = 0
            tsf = d['temp'][0, -1, :, :].filled(np.nan)
            wet = tsf != 0                       # NaN (land) also passes; masked later
            ox = d['oxygen'][0, :, :, :].filled(np.nan)
            oxy3 = oxy3 + np.where(wet, ox, 0.0)
            for v in layer_vars:
                if v == 'oxygen':
                    a = np.stack([ox[0], ox[-1]])
                elif v == 'temp':
                    a = np.stack([d[v][0, 0, :, :].filled(np.nan), tsf])
                else:
                    a = np.stack([d[v][0, 0, :, :].filled(np.nan),
                                  d[v][0, -1, :, :].filled(np.nan)])
                lay[v] = lay[v] + np.where(wet, a, 0.0)
            nwet = nwet + wet
            for si, (j, i) in enumerate(zip(eta_idx, xi_idx)):
                if not wet[j, i]:
                    continue
                for v in stn_vars:
                    stn[v][si] += d[v][0, :, j, i].filled(np.nan)
                zeta[si] += float(d['zeta'][0, j, i])
                nwet_stn[si] += 1
            d.close()
    except Exception as e:
        return {'error': '%s: %s' % (fn, e)}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN / all-dry cells
        nw = np.where(nwet > 0, nwet, np.nan)
        ns = np.where(nwet_stn > 0, nwet_stn, np.nan)
        oxy3 = oxy3 / nw
        out = {
            'stn': {v: stn[v] / ns[:, None] for v in stn_vars},
            'zeta': zeta / ns,
            'lay': {v: (lay[v] / nw).astype(np.float32) for v in layer_vars},
            'hyp': np.nanmin(oxy3, axis=0) < HYP_UM,
            'wetfrac': (nwet / len(fn_list)).astype(np.float32),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gtxs', type=str, default='wb1_t0_xn11abbur00,wb1_t2_xn11abbur00')
    parser.add_argument('-ro', '--roms_out_num', type=int, default=2)
    parser.add_argument('-0', '--ds0', type=str, default='2024.01.01')
    parser.add_argument('-1', '--ds1', type=str, default='2025.12.31')
    parser.add_argument('-navg', type=int, default=24)   # avg files needed for a complete day
    parser.add_argument('-Nproc', type=int, default=10)
    parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)
    args = parser.parse_args()

    gtxs = [g.strip() for g in args.gtxs.split(',') if g.strip()]
    gridname, tag, ex_name = gtxs[0].split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
    if args.roms_out_num > 0:
        Ldir['roms_out'] = Ldir['roms_out' + str(args.roms_out_num)]
    out_dir = Ldir['LOo'] / 'DM_outs' / OUT_FOLDER
    Lfun.make_dir(out_dir)

    # ---- days complete in every run ----------------------------------------
    dt0 = datetime.strptime(args.ds0, Ldir['ds_fmt'])
    dt1 = datetime.strptime(args.ds1, Ldir['ds_fmt'])
    days = []
    dt = dt0
    while dt <= dt1:
        days.append(dt)
        dt += timedelta(days=1)
    fns = {}
    for gtx in gtxs:
        fns[gtx] = {}
        for dt in days:
            fl = avg_files(Ldir['roms_out'] / gtx / ('f' + dt.strftime(Ldir['ds_fmt'])))
            if len(fl) >= args.navg:
                fns[gtx][dt] = fl
        print('%s: %d complete days in %s to %s' % (gtx, len(fns[gtx]), args.ds0, args.ds1))
    common = [dt for dt in days if all(dt in fns[g] for g in gtxs)]
    if len(common) == 0:
        print('*** No days complete in all runs. Check -ro / -0 / -1.')
        sys.exit(1)
    if args.testing:
        common = common[:2]
    print('common days: %d  (%s to %s)' % (len(common), common[0].strftime(Ldir['ds_fmt']),
                                          common[-1].strftime(Ldir['ds_fmt'])))
    sys.stdout.flush()

    # ---- grid + stations from the first run's own output -------------------
    fn0 = fns[gtxs[0]][common[0]][0]
    g = xr.open_dataset(fn0)
    lon2, lat2 = g['lon_rho'].values, g['lat_rho'].values
    h2, mask = g['h'].values, g['mask_rho'].values
    have = set(g.data_vars)
    g.close()
    for gtx in gtxs[1:]:
        gg = xr.open_dataset(fns[gtx][common[0]][0])
        nh = int((gg['h'].values != h2).sum()); nm = int((gg['mask_rho'].values != mask).sum())
        gg.close()
        if nh or nm:
            print('*** WARNING: %s differs from %s at %d h cells, %d mask cells' % (gtx, gtxs[0], nh, nm))
    stn_vars = [v for v in STN_VARS if v in have]
    layer_vars = [v for v in LAYER_VARS if v in have]
    S = zrfun.get_basic_info(fn0, only_S=True)

    names = list(STATIONS.keys())
    eta_idx, xi_idx = [], []
    for nm in names:
        lo, la = STATIONS[nm]
        iy = zfun.find_nearest_ind(lat2[:, 0], la)
        ix = zfun.find_nearest_ind(lon2[0, :], lo)
        iy, ix = nearest_water(iy, ix, mask)
        eta_idx.append(int(iy)); xi_idx.append(int(ix))
    eta_idx = np.array(eta_idx); xi_idx = np.array(xi_idx)

    months = sorted(set(dt.strftime('%Y-%m') for dt in common))
    mi_of = {m: k for k, m in enumerate(months)}
    ny, nx = mask.shape
    N = S['N']
    nd, ns, nmo = len(common), len(names), len(months)

    for gtx in gtxs:
        tt0 = time()
        print((' %s ' % gtx).center(60, '='))
        stn = {v: np.full((nd, ns, N), np.nan) for v in stn_vars}
        zeta = np.full((nd, ns), np.nan)
        oxy_bot = np.full((nd, ny, nx), np.nan, dtype=np.float32)
        lay_sum = {v: np.zeros((nmo, 2, ny, nx)) for v in layer_vars}
        lay_cnt = np.zeros((nmo, ny, nx))          # days with any wet hour, per cell
        wet_sum = np.zeros((nmo, ny, nx))          # sum of daily wet fractions
        hyp = np.zeros((nmo, ny, nx), dtype=np.int16)
        ndays = np.zeros(nmo, dtype=int)
        good = np.zeros(nd, dtype=bool)

        tasks = [(fns[gtx][dt], eta_idx, xi_idx, stn_vars, layer_vars) for dt in common]
        if args.testing or args.Nproc <= 1:
            results = map(reduce_day, tasks)
        else:
            ex = ProcessPoolExecutor(max_workers=args.Nproc)
            results = ex.map(reduce_day, tasks)
        for k, r in enumerate(results):
            if 'error' in r:
                print('  skipped %s (%s)' % (common[k].strftime(Ldir['ds_fmt']), r['error']))
                continue
            good[k] = True
            m = mi_of[common[k].strftime('%Y-%m')]
            ndays[m] += 1
            for v in stn_vars:
                stn[v][k] = r['stn'][v]
            zeta[k] = r['zeta']
            oxy_bot[k] = r['lay']['oxygen'][0]
            for v in layer_vars:
                lay_sum[v][m] += np.nan_to_num(r['lay'][v])
            lay_cnt[m] += np.isfinite(r['lay']['temp'][1])
            wet_sum[m] += r['wetfrac']
            hyp[m] += r['hyp'].astype(np.int16)
            if k % 30 == 0:
                print('  %s done (%d/%d)' % (common[k].strftime(Ldir['ds_fmt']), k+1, nd))
                sys.stdout.flush()
        if not (args.testing or args.Nproc <= 1):
            ex.shutdown()

        # z_rho at the stations from the daily-mean zeta
        z_rho = np.full((nd, ns, N), np.nan)
        hs = h2[eta_idx, xi_idx]
        for k in np.flatnonzero(good):
            z_rho[k] = zrfun.get_z(hs, zeta[k], S, only_rho=True).T

        dv = {}
        for v in stn_vars:
            dv['stn_' + v] = (('time', 'station', 's_rho'), stn[v])
        dv['stn_zeta'] = (('time', 'station'), zeta)
        dv['stn_z_rho'] = (('time', 'station', 's_rho'), z_rho)
        dv['stn_h'] = (('station',), hs)
        dv['stn_lon'] = (('station',), lon2[eta_idx, xi_idx])
        dv['stn_lat'] = (('station',), lat2[eta_idx, xi_idx])
        dv['stn_eta'] = (('station',), eta_idx)
        dv['stn_xi'] = (('station',), xi_idx)
        dv['oxygen_bot_daily'] = (('time', 'eta_rho', 'xi_rho'), oxy_bot)
        cnt_safe = np.where(lay_cnt > 0, lay_cnt, np.nan)
        for v in layer_vars:
            dv[v + '_bot'] = (('month', 'eta_rho', 'xi_rho'),
                              (lay_sum[v][:, 0] / cnt_safe).astype(np.float32))
            dv[v + '_sfc'] = (('month', 'eta_rho', 'xi_rho'),
                              (lay_sum[v][:, 1] / cnt_safe).astype(np.float32))
        dv['wet_frac'] = (('month', 'eta_rho', 'xi_rho'),
                          (wet_sum / np.where(ndays > 0, ndays, np.nan)[:, None, None]).astype(np.float32))
        dv['hyp_days'] = (('month', 'eta_rho', 'xi_rho'), hyp)
        dv['ndays'] = (('month',), ndays)
        dv['day_ok'] = (('time',), good.astype(np.int8))
        dv['lon_rho'] = (('eta_rho', 'xi_rho'), lon2)
        dv['lat_rho'] = (('eta_rho', 'xi_rho'), lat2)
        dv['h'] = (('eta_rho', 'xi_rho'), h2)
        dv['mask_rho'] = (('eta_rho', 'xi_rho'), mask)
        ds = xr.Dataset(dv, coords={'time': pd.to_datetime(common) + pd.Timedelta(hours=12),
                                    'station': names, 'month': months})
        ds.attrs['gtagex'] = gtx
        ds.attrs['note'] = ('daily means over the WET hours of the hourly ocean_avg files '
                            '(dry cell-hours are written as 0 and are excluded); '
                            'hyp_days = days with daily-mean DO < 2 mg/L anywhere in column; '
                            'wet_frac = fraction of hours each cell was wet')
        enc = {k: {'zlib': True, 'complevel': 1} for k in ds.data_vars
               if ds[k].dtype.kind == 'f'}
        out_fn = out_dir / ('reduce_%s_%s_%s.nc' % (gtx, common[0].strftime(Ldir['ds_fmt']),
                                                   common[-1].strftime(Ldir['ds_fmt'])))
        out_fn.unlink(missing_ok=True)
        ds.to_netcdf(out_fn, encoding=enc)
        print('Saved %s  (%d good days, %d sec)' % (out_fn, good.sum(), time() - tt0))
        sys.stdout.flush()

    print('Done.')


if __name__ == '__main__':
    main()
