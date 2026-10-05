"""
Reduce pcmap tracker output to small per-release files of particle-level
residence-time metrics, so the year of releases can be analysed without ever
reopening the full tracks. Runs on apogee next to the tracks (one pass, ~650
files); rerunning skips releases already reduced unless -clobber.

COVE AND REGIONS (all from each particle's INITIAL rho cell)
  cove           tef2 wb1_pc1 segments pc_cp_m + pc_cp_p + pc_lp_m, i.e. all
                 water landward of pc_lp (317 cells) -- the release footprint.
                 Particles that start outside it are dropped.
  inner/outer    inner = pc_cp_m (landward of pc_cp, the Coupeville pinch)
  north/south    split at the along-cove centreline: per rho column, the mean
                 j of that column's cove cells; north if j > it
  surf/bot       split at mid-column: surf if the initial cs >= -0.5
The quadrant is inner/outer x north/south; region3 adds surf/bot.

METRICS PER PARTICLE (hourly frames, record trimmed to -days)
  first_exit_h   hours until the particle is first outside the cove
                 (censored = never left within the record; value = record length)
  exp_<T>d_h     exposure: total hours inside the cove up to T days, re-entry
                 counted, for T in -cutoffs. Comparing T says whether 14 d is
                 long enough.
  quad_exit_h    hours until it first leaves its own horizontal quadrant
  reg3_exit_h    hours until it first leaves its own 3-D region (quadrant AND
                 vertical half, judged by cs >= -0.5 at each frame)
Per release also: still-inside and never-left curves for the whole cove and
each quadrant/region, and the release-mean zeta series.

Output: LO_output/DM_outs/20261005_pcmap_reduce/<gtx>/<dir>__<release>.p

run 20261005_pcmap_reduce.py
run 20261005_pcmap_reduce.py -dir_glob 'pcmap_3d*_H_2025.01.*'
run 20261005_pcmap_reduce.py -dir_glob pcret_3d -days 14     (mac test)
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-dir_glob', default='pcmap_3d*')
p.add_argument('-days', type=float, default=14.0, help='common record length')
p.add_argument('-cutoffs', default='7,10,14', help='exposure cutoffs [days]')
p.add_argument('-clobber', action='store_true')
args = p.parse_args()
cutoffs = [float(c) for c in args.cutoffs.split(',')]

Ldir = Lfun.Lstart(gridname='wb1')
trk = Ldir['LOo'] / 'tracks2' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_reduce' / args.gtx
Lfun.make_dir(out_dir)

# ------------------------------------------------------------- regions ---
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
g.close()
lon_ax, lat_ax = lon[0, :], lat[:, 0]
dlon, dlat = lon_ax[1] - lon_ax[0], lat_ax[1] - lat_ax[0]
NR, NC = lon.shape

tef2 = Ldir['LOo'] / 'extract' / 'tef2'
seg = pickle.load(open(sorted(tef2.glob('seg_info_dict_wb1_pc1_*.p'))[0], 'rb'))


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
    jc = jj[ii == i].mean()
    js = jj[(ii == i) & (jj > jc)]
    north[js, i] = True
# quadrant code on the grid: 0 inner-N, 1 inner-S, 2 outer-N, 3 outer-S, -1 not cove
QUAD = np.full((NR, NC), -1, dtype=int)
QUAD[cove] = (2 * (~inner) + (~north))[cove]
QNAMES = ['inner-N', 'inner-S', 'outer-N', 'outer-S']
print('cove %d cells; quadrants %s'
      % (cove.sum(), ', '.join('%s %d' % (n, (QUAD == k).sum()) for k, n in enumerate(QNAMES))))


def ji_of(plon, plat):
    """Nearest rho indices on this plaid grid; NaN positions map to ok=False."""
    ok = np.isfinite(plon) & np.isfinite(plat)
    i = np.zeros(plon.shape, dtype=int)
    j = np.zeros(plon.shape, dtype=int)
    i[ok] = np.clip(np.round((plon[ok] - lon_ax[0]) / dlon), 0, NC - 1).astype(int)
    j[ok] = np.clip(np.round((plat[ok] - lat_ax[0]) / dlat), 0, NR - 1).astype(int)
    return j, i, ok


def first_false(a):
    """Per column, index of the first False (frame 0 excluded); -1 if none."""
    b = ~a[1:, :]
    k = np.argmax(b, axis=0) + 1
    k[~b.any(axis=0)] = -1
    return k


# --------------------------------------------------------------- reduce ---
fns = sorted(f for dd in sorted(trk.glob(args.dir_glob)) if dd.is_dir()
             for f in dd.glob('release_*.nc'))
print('%d release files under %s/%s' % (len(fns), trk.name, args.dir_glob))
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
    sl = slice(0, nf)
    plon = d.lon.values[sl]; plat = d.lat.values[sl]; cs = d.cs.values[sl]
    zeta = d.zeta.values[sl]; h0all = d.h.values[0]
    ot = pd.to_datetime(d.ot.values[sl])
    d.close()

    j, i, ok = ji_of(plon, plat)
    keep = ok[0] & cove[j[0], i[0]]
    plon, plat, cs, zeta = plon[:, keep], plat[:, keep], cs[:, keep], zeta[:, keep]
    j, i, ok = j[:, keep], i[:, keep], ok[:, keep]

    q = np.where(ok, QUAD[j, i], -1)                  # quadrant each frame
    surf = cs >= -0.5
    inside = q >= 0
    q0, s0 = q[0], surf[0]
    in_quad = q == q0[None, :]
    in_reg3 = in_quad & (surf == s0[None, :])

    T_h = nf - 1
    k_exit = first_false(inside)
    P = pd.DataFrame(dict(
        j0=j[0], i0=i[0], cs0=cs[0, :], h0=h0all[keep], quad0=q0, surf0=s0,
        first_exit_h=np.where(k_exit < 0, T_h, k_exit).astype(float),
        censored=k_exit < 0))
    for c in cutoffs:
        n = int(round(c * 24))
        P['exp_%gd_h' % c] = inside[1:n + 1].sum(axis=0).astype(float)
    for name, a in [('quad_exit_h', in_quad), ('reg3_exit_h', in_reg3)]:
        k = first_false(a)
        P[name] = np.where(k < 0, T_h, k).astype(float)

    curves = {}
    groups = {'cove': np.ones(len(q0), dtype=bool)}
    for k, n in enumerate(QNAMES):
        groups[n] = q0 == k
        for sname, sv in [('surf', True), ('bot', False)]:
            groups['%s-%s' % (n, sname)] = (q0 == k) & (s0 == sv)
    for gname, m in groups.items():
        if m.sum() == 0:
            continue
        curves[gname] = dict(n=int(m.sum()), still=inside[:, m].mean(axis=1),
                             never=np.minimum.accumulate(inside[:, m], axis=0).mean(axis=1))

    meta = dict(file=str(fn.relative_to(trk)), dir=fn.parent.name,
                t0=ot[0], hours=np.arange(nf), zeta_mean=np.nanmean(zeta, axis=1),
                n_dropped=int((~keep).sum()), days=args.days, cutoffs=cutoffs)
    pickle.dump(dict(meta=meta, P=P, curves=curves), open(out_fn, 'wb'))
    print('%-45s t0 %s  NP %d (dropped %d)  censored %.1f%%  median exit %.1f h'
          % (out_fn.name, ot[0], len(P), meta['n_dropped'],
             100 * P.censored.mean(), P.first_exit_h.median()))
