"""
Reduce pcmap tracks to per-release VERTICAL cohort curves: where in the water
column the particles that started in the surface or the bottom half end up.

Runs on apogee next to the tracks (reads only lon, lat, cs -- no history files,
so it is much faster than 20261005_pcmap_reduce.py). Rerunning skips releases
already reduced unless -clobber.

Same cove, quadrants and starting half as 20261005_pcmap_reduce.py, all from
each particle's INITIAL position: cove = pc_cp_m + pc_cp_p + pc_lp_m, inner =
pc_cp_m, N/S at each column's mean j, surface half if the initial cs >= -0.5.
Particles that start outside the cove are dropped.

Height is FRACTIONAL height in the column, cs + 1 (0 = bed, 1 = surface), not
metres: once a particle leaves the 7-25 m cove for 30-100+ m of Saratoga
Passage, metres above the bed say where it went, not how high in the column it
sits.

Per release and per cohort -- surf, bot (whole cove) and each quadrant-half
(inner-N-surf, ..., outer-S-bot) -- hourly over -days:
  h_mean_all, h_med_all    mean / median height, every particle wherever it is
  h_mean_in,  h_med_in     the same, only particles inside the cove right now
  sw_all, sw_in            fraction now in the OTHER half from where it started
                           (surface starter below cs = -0.5, or the reverse),
                           all / inside-only
  n, n_in                  cohort size, number inside the cove right now
Inside-only values are NaN where no particle of the cohort is inside.

Output: LO_output/DM_outs/20261006_pcmap_vertical_reduce/<gtx>/<dir>__<release>.p

run 20261006_pcmap_vertical_reduce.py
run 20261006_pcmap_vertical_reduce.py -done_only          (while the launcher runs)
run 20261006_pcmap_vertical_reduce.py -dir_glob pcret_3d  (mac test)
"""
import argparse
import pickle
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-dir_glob', default='pcmap_3d*')
p.add_argument('-days', type=float, default=14.0, help='common record length')
p.add_argument('-clobber', action='store_true')
p.add_argument('-done_only', action='store_true',
               help='only releases the launcher logged as finished (returncode 0)')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
trk = Ldir['LOo'] / 'tracks2' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261006_pcmap_vertical_reduce' / args.gtx
Lfun.make_dir(out_dir)

# ------------------------------------------------------------- regions ---
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
g.close()
lon_ax, lat_ax = lon[0, :], lat[:, 0]
dlon, dlat = lon_ax[1] - lon_ax[0], lat_ax[1] - lat_ax[0]
NR, NC = lon.shape
seg = pickle.load(open(sorted((Ldir['LOo'] / 'extract' / 'tef2').glob(
    'seg_info_dict_wb1_pc1_*.p'))[0], 'rb'))


def seg_mask(names):
    m = np.zeros((NR, NC), dtype=bool)
    for s in names:
        a = np.array(seg[s]['ji_list'])
        m[a[:, 0], a[:, 1]] = True
    return m


cove = seg_mask(['pc_cp_m', 'pc_cp_p', 'pc_lp_m'])
inner = seg_mask(['pc_cp_m'])
jj, ii = np.where(cove)
north = np.zeros((NR, NC), dtype=bool)
for i in np.unique(ii):
    north[jj[(ii == i) & (jj > jj[ii == i].mean())], i] = True
QUAD = np.full((NR, NC), -1, dtype=int)
QUAD[cove] = (2 * (~inner) + (~north))[cove]
QNAMES = ['inner-N', 'inner-S', 'outer-N', 'outer-S']

# --------------------------------------------------------------- reduce ---
fns = sorted(f for dd in sorted(trk.glob(args.dir_glob)) if dd.is_dir()
             for f in dd.glob('release_*.nc'))
if args.done_only:
    T = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_launch' / 'pcmap_timing.csv')
    ok = set(T.log[T.returncode == 0].str.replace('.log', '', regex=False))
    fns = [f for f in fns if f.parent.name in ok]
print('%d release files under %s/%s%s' % (len(fns), trk.name, args.dir_glob,
                                        ' (finished only)' if args.done_only else ''))
nf = int(round(args.days * 24)) + 1
for fn in fns:
    out_fn = out_dir / ('%s__%s.p' % (fn.parent.name, fn.stem))
    if out_fn.is_file() and not args.clobber:
        continue
    d = xr.open_dataset(fn)
    if d.sizes['Time'] < nf:
        print('SKIP %s: %d frames < %d' % (out_fn.name, d.sizes['Time'], nf))
        d.close()
        continue
    plon = d.lon.values[:nf]; plat = d.lat.values[:nf]; cs = d.cs.values[:nf]
    t0 = pd.Timestamp(d.ot.values[0])
    d.close()

    okp = np.isfinite(plon) & np.isfinite(plat)
    i = np.zeros(plon.shape, dtype=int); j = np.zeros(plon.shape, dtype=int)
    i[okp] = np.clip(np.round((plon[okp] - lon_ax[0]) / dlon), 0, NC - 1).astype(int)
    j[okp] = np.clip(np.round((plat[okp] - lat_ax[0]) / dlat), 0, NR - 1).astype(int)
    q = np.where(okp, QUAD[j, i], -1)
    keep = q[0] >= 0
    q, cs, okp = q[:, keep], cs[:, keep], okp[:, keep]
    inside = q >= 0
    hgt = np.where(okp & np.isfinite(cs), cs + 1, np.nan)
    surf0 = cs[0] >= -0.5
    in_surf = hgt >= 0.5                                  # current half (NaN -> False)
    valid = np.isfinite(hgt)

    groups = {'surf': surf0, 'bot': ~surf0}
    for k, qn in enumerate(QNAMES):
        groups['%s-surf' % qn] = (q[0] == k) & surf0
        groups['%s-bot' % qn] = (q[0] == k) & ~surf0

    G = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for gname, m in groups.items():
            if m.sum() == 0:
                continue
            H = hgt[:, m]; V = valid[:, m]; INS = inside[:, m]
            other = (in_surf[:, m] != surf0[m][None, :]) & V    # now in the other half
            Hin = np.where(INS, H, np.nan)
            n_in = (INS & V).sum(axis=1)
            G[gname] = dict(
                n=int(m.sum()), n_in=n_in.astype(np.int32),
                h_mean_all=np.nanmean(H, axis=1).astype(np.float32),
                h_med_all=np.nanmedian(H, axis=1).astype(np.float32),
                h_mean_in=np.nanmean(Hin, axis=1).astype(np.float32),
                h_med_in=np.nanmedian(Hin, axis=1).astype(np.float32),
                sw_all=(other.sum(axis=1) / np.maximum(V.sum(axis=1), 1)).astype(np.float32),
                sw_in=np.where(n_in > 0, (other & INS).sum(axis=1) / np.maximum(n_in, 1),
                               np.nan).astype(np.float32))
    meta = dict(file=str(fn.relative_to(trk)), dir=fn.parent.name, t0=t0, nf=nf, days=args.days,
                n_dropped=int((~keep).sum()))
    pickle.dump(dict(meta=meta, groups=G), open(out_fn, 'wb'))
    print('%-45s t0 %s  surf %d bot %d  height at end: surf %.2f bot %.2f  switched: surf %.2f bot %.2f'
          % (out_fn.name, t0, G['surf']['n'], G['bot']['n'], G['surf']['h_mean_all'][-1],
             G['bot']['h_mean_all'][-1], G['surf']['sw_all'][-1], G['bot']['sw_all'][-1]))
