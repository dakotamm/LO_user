"""
Reduce pcsrc tracker output to small per-release files, pairing each pcsrc
release with its pcmap twin (same -d, -sh, -dtt, so identical time axes), so
the source-water analysis runs on the Mac without the tracks. Runs on apogee
next to the tracks and history files; rerunning skips releases already reduced
unless -clobber.

TWO COHORTS PER RELEASE (from each particle's INITIAL rho cell)
  spu    from the pcsrc track file (-src, default pcsrc_spu): particles starting
         in upper Saratoga, the pcsrc_regions 'spu' region (= tef2 pc_lp_p)
  cove   from the pcmap twin: particles starting in Penn Cove (pcsrc_regions
         'cove' = tef2 pc_cp_m + pc_cp_p + pc_lp_m)
Particles starting anywhere else are dropped and counted.

VOLUME EACH PARTICLE STANDS FOR (column vol)
  vol = h * A * stride**2 / n,  n = particles starting in that rho cell in this
  file, stride from pcsrc_regions.SEED (spu 2, cove 1). With the seeding rule
  that is exactly pcsrc_regions.particle_volume(), but counting n from the file
  keeps it right if the tracker drops particles on land. h at MEAN SEA LEVEL
  (wb1_t0 bathymetry = grid.nc): 20261013_pcsrc_source.py rescales it to the
  release tide with the segment-mean zeta, which matters for the E set.

PER PARTICLE (DataFrame P)
  cohort, j0, i0, cs0, h0, z0, hab0, salt0, temp0, DO0, vol
  DO0 [mg/L] from the history file at the release hour, interpolated in sigma
  within the starting column (as in 20261005_pcmap_reduce.py); NaN with -no_do.

HOURLY, every particle, full record (arrays shaped (nf, NP), same order as P)
  J, I    int16 rho indices of the nearest cell, -1 where the position is NaN
  CS      int16 fractional depth cs * 10000 (cs in -1..0), -32768 if NaN
  Z       float16 depth [m, negative down]
  S, T    float16 salinity and temperature along the path
Any gate or target (whole cove, inner-cove bottom water, a section crossing)
is applied later from J, I, CS -- nothing about the targets is fixed here.

Output: LO_output/DM_outs/20261013_pcsrc_reduce/<gtx>/<sub_tag>.p
        about 35 MB per release (~7 GB for all 218)

On apogee:
  python 20261013_pcsrc_reduce.py -month 8
  nohup python 20261013_pcsrc_reduce.py -done_only > pcsrc_reduce.log 2>&1 &   (while the launcher runs)
Mac test on pcret (one file holds both cohorts, so it is passed as both):
  python 20261013_pcsrc_reduce.py -test_pcret -no_do
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

import pcsrc_regions as pr

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-src', default='pcsrc_spu', help='pcsrc experiment name')
p.add_argument('-twin', default='pcmap', help='experiment that supplies the cove cohort')
p.add_argument('-month', type=int, default=0, help='0 = all releases found')
p.add_argument('-clobber', action='store_true')
p.add_argument('-no_do', action='store_true', help='skip the DO lookup (no history files)')
p.add_argument('-done_only', action='store_true',
               help='only releases the pcsrc launcher logged as finished (returncode 0)')
p.add_argument('-test_pcret', action='store_true', help='mac test on the pcret tracks')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
trk = Ldir['LOo'] / 'tracks2' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261013_pcsrc_reduce' / (
    args.gtx + ('_test_pcret' if args.test_pcret else ''))
Lfun.make_dir(out_dir)

g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat, h = g.lon_rho.values, g.lat_rho.values, g.h.values
A = 1 / (g.pm.values * g.pn.values)
wet = g.mask_rho.values == 1
g.close()
lon_ax, lat_ax = lon[0, :], lat[:, 0]
dlon, dlat = lon_ax[1] - lon_ax[0], lat_ax[1] - lat_ax[0]
NR, NC = lon.shape
R = pr.regions(Ldir, lon, lat, wet)


def ji_of(plon, plat):
    """Nearest rho indices on this plaid grid; NaN positions map to ok=False."""
    ok = np.isfinite(plon) & np.isfinite(plat)
    i = np.zeros(plon.shape, dtype=int)
    j = np.zeros(plon.shape, dtype=int)
    i[ok] = np.clip(np.round((plon[ok] - lon_ax[0]) / dlon), 0, NC - 1).astype(int)
    j[ok] = np.clip(np.round((plat[ok] - lat_ax[0]) / dlat), 0, NR - 1).astype(int)
    return j, i, ok


def do_at_release(rel_dir, cs0, j0, i0, t_expect):
    """DO [mg/L] at each particle's start, from the history file the tracker
    started on (same as 20261005_pcmap_reduce.py)."""
    info = Lfun.csv_to_dict(rel_dir / 'exp_info.csv')
    L = dict(roms_out=Path(info['roms_out']), gtagex=info['gtagex'])
    fn_his = Lfun.get_fn_list('hourly', L, info['date_string'],
                              info['date_string'])[int(info['start_hour'])]
    ds = xr.open_dataset(fn_his)
    t_his = pd.Timestamp(ds.ocean_time.values[0])
    if abs((t_his - t_expect) / pd.Timedelta(minutes=1)) > 1:
        raise ValueError('%s is at %s but the release starts at %s'
                         % (fn_his, t_his, t_expect))
    s_rho = ds.s_rho.values
    ox = ds.oxygen.values[0][:, j0, i0]
    ds.close()
    c = np.clip(cs0, s_rho[0], s_rho[-1])
    k = np.clip(np.searchsorted(s_rho, c), 1, len(s_rho) - 1)
    w = (c - s_rho[k - 1]) / (s_rho[k] - s_rho[k - 1])
    n = np.arange(len(c))
    oxp = (1 - w) * ox[k - 1, n] + w * ox[k, n]
    return oxp * 32 / 1000, fn_his


def read_cohort(fn, rn):
    """Particles of track file fn that start in region rn."""
    d = xr.open_dataset(fn)
    plon = d.lon.values; plat = d.lat.values; cs = d.cs.values
    z = d.z.values; S = d.salt.values; T = d.temp.values
    h0all = d.h.values[0]; zeta0 = d.zeta.values[0]
    ot = pd.to_datetime(d.ot.values)
    d.close()
    j, i, ok = ji_of(plon, plat)
    keep = ok[0] & R[rn][j[0], i[0]]
    j0, i0 = j[0, keep], i[0, keep]
    # volume per particle from the count actually in each starting cell
    cell = j0 * NC + i0
    _, inv, cnt = np.unique(cell, return_inverse=True, return_counts=True)
    vol = h[j0, i0] * A[j0, i0] * pr.SEED[rn]['stride'] ** 2 / cnt[inv]
    if args.test_pcret and rn == 'spu':
        # pcret thinned pc_lp_p by a step through its particle list, not by a
        # cell stride, so give each particle an equal share of the region
        vol = np.full(len(j0), (h * A)[R[rn]].sum() / len(j0))
    cs0 = cs[0, keep]
    P = pd.DataFrame(dict(
        cohort=rn, j0=j0, i0=i0, cs0=cs0, h0=h0all[keep], z0=z[0, keep],
        hab0=(cs0 + 1) * (h0all[keep] + zeta0[keep]),
        salt0=S[0, keep], temp0=T[0, keep], vol=vol))
    H = dict(J=np.where(ok, j, -1)[:, keep].astype(np.int16),
             I=np.where(ok, i, -1)[:, keep].astype(np.int16),
             CS=np.where(np.isfinite(cs), np.round(cs * 10000), -32768)[:, keep].astype(np.int16),
             Z=z[:, keep].astype(np.float16),
             S=S[:, keep].astype(np.float16),
             T=T[:, keep].astype(np.float16))
    return P, H, ot, int((~keep).sum())


# ------------------------------------------------------------ releases ---
if args.test_pcret:
    pairs = [(f, f, f.stem.replace('release_', 'pcret_')) for f in
             sorted((trk / 'pcret_3d').glob('release_*.nc'))]
else:
    pairs = []
    for dd in sorted(trk.glob(args.src + '_3d*')):
        sub_tag = dd.name.split('_3d')[1].split('_', 2)[-1] if '_sh' in dd.name else dd.name.split('_3d_')[1]
        twin_dir = trk / dd.name.replace(args.src, args.twin, 1)
        fs = sorted(dd.glob('release_*.nc'))
        if len(fs) != 1:
            continue
        ft = twin_dir / fs[0].name
        if not ft.is_file():
            print('NO TWIN for %s (looked for %s)' % (dd.name, ft))
            continue
        pairs.append((fs[0], ft, sub_tag))
    if args.month:
        pairs = [pp for pp in pairs if int(pp[2].split('_')[1].split('.')[1]) == args.month]
    if args.done_only:
        tfn = Ldir['LOo'] / 'DM_outs' / '20261012_pcsrc_launch' / ('%s_timing.csv' % args.src)
        Tt = pd.read_csv(tfn)
        okset = set(Tt.sub_tag[Tt.returncode == 0])
        pairs = [pp for pp in pairs if pp[2] in okset]
print('%d releases to consider -> %s' % (len(pairs), out_dir))
sys.stdout.flush()

for fs, ft, sub_tag in pairs:
    out_fn = out_dir / ('%s.p' % sub_tag)
    if out_fn.is_file() and not args.clobber:
        continue
    Ps, Hs, ots, drop_s = read_cohort(fs, 'spu')
    Pc, Hc, otc, drop_c = read_cohort(ft, 'cove')
    nf = min(len(ots), len(otc))
    if not (ots[:nf] == otc[:nf]).all():
        raise ValueError('%s and its twin %s have different times' % (fs, ft))
    P = pd.concat([Ps, Pc], ignore_index=True)
    H = {k: np.concatenate([Hs[k][:nf], Hc[k][:nf]], axis=1) for k in Hs}
    fn_his = None
    if args.no_do:
        P['DO0'] = np.nan
    else:
        do_s, fn_his = do_at_release(fs.parent, Ps.cs0.values, Ps.j0.values, Ps.i0.values, ots[0])
        do_c, _ = do_at_release(ft.parent, Pc.cs0.values, Pc.j0.values, Pc.i0.values, otc[0])
        P['DO0'] = np.concatenate([do_s, do_c])
    meta = dict(sub_tag=sub_tag, set=sub_tag.split('_')[0], t0=ots[0], ot=ots[:nf], nf=nf,
                src=str(fs.relative_to(trk)), twin=str(ft.relative_to(trk)),
                dropped=dict(spu=drop_s, cove=drop_c), fn_his=str(fn_his),
                SEED=pr.SEED)
    pickle.dump(dict(meta=meta, P=P, H=H), open(out_fn, 'wb'), protocol=4)
    print('%-16s t0 %s  nf %d  spu %d  cove %d  (dropped %d/%d)  V spu %.2e cove %.2e  '
          'DO0 %.2f-%.2f  %.1f MB'
          % (sub_tag, ots[0], nf, len(Ps), len(Pc), drop_s, drop_c,
             Ps.vol.sum(), Pc.vol.sum(), np.nanmin(P.DO0) if not args.no_do else np.nan,
             np.nanmax(P.DO0) if not args.no_do else np.nan, out_fn.stat().st_size / 1e6))
    sys.stdout.flush()
