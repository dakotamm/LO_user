"""
Dissolved oxygen along every pcsrc particle path, sampled from the ROMS history
files, for the reduced releases of 20261013_pcsrc_reduce.py. The tracker does
not carry oxygen and the reduce only stores it at release (DO0), so this is a
separate pass. Runs on apogee next to the history files.

For each hour of each release, every particle with a valid position gets the
oxygen of its reduced cell (J, I), interpolated linearly in sigma at its cs --
the same rule as DO0 in the reduce, so DO[0] reproduces DO0 (checked and
printed). oxygen [mmol/m3] * 32/1000 = mg/L.

EFFICIENT ORDER: releases overlap in time (one every ~3 lunar days, each 15 d
long), so the history files are walked ONCE in time order and each hour is
sampled for every release active then, instead of reopening the same file for
each release. -month limits the releases to those starting in that month.

Output: LO_output/DM_outs/20261013_pcsrc_do_path/<gtx>/<sub_tag>_DO.p
        dict(DO = float16 (nf, NP) mg/L in the reduced particle order, NaN
        where the position is invalid; meta), about 5 MB per release

On apogee:
  python 20261013_pcsrc_do_path.py -month 8
  nohup python 20261013_pcsrc_do_path.py > pcsrc_do_path.log 2>&1 &
"""
import argparse
import pickle
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-ro', type=int, default=2, help='roms_out number, as for tracker.py')
p.add_argument('-month', type=int, default=0, help='0 = all reduced releases')
p.add_argument('-clobber', action='store_true')
args = p.parse_args()

gridname, tag, ex_name = args.gtx.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
Ldir['roms_out'] = Ldir['roms_out' + str(args.ro)] if args.ro > 0 else Ldir['roms_out']
red_dir = Ldir['LOo'] / 'DM_outs' / '20261013_pcsrc_reduce' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261013_pcsrc_do_path' / args.gtx
Lfun.make_dir(out_dir)

# ------------------------------------------------------ releases to do ---
rel = {}
for fn in sorted(red_dir.glob('*.p')):
    out_fn = out_dir / (fn.stem + '_DO.p')
    if out_fn.is_file() and not args.clobber:
        continue
    D = pickle.load(open(fn, 'rb'))
    t0 = pd.Timestamp(D['meta']['t0'])
    if args.month and t0.month != args.month:
        continue
    nf = D['meta']['nf']
    rel[fn.stem] = dict(t0=t0, nf=nf, out_fn=out_fn, DO0=D['P'].DO0.values,
                        J=D['H']['J'], I=D['H']['I'], CS=D['H']['CS'],
                        DO=np.full(D['H']['J'].shape, np.nan, dtype=np.float32))
if not rel:
    sys.exit('nothing to do in %s' % red_dir)
t_first = min(r['t0'] for r in rel.values())
t_last = max(r['t0'] + pd.Timedelta(hours=r['nf'] - 1) for r in rel.values())
print('%d releases, hours %s to %s' % (len(rel), t_first, t_last))

# hourly file list: with 'hourly' (his_num 2) index k is hour k after 00:00 of ds0
day0 = t_first.normalize()
fn_list = Lfun.get_fn_list('hourly', Ldir, day0.strftime(Lfun.ds_fmt),
                           t_last.normalize().strftime(Lfun.ds_fmt))
hours = pd.date_range(t_first, t_last, freq='h')
print('%d history files to read' % len(hours))
sys.stdout.flush()

tic = time.time()
for nh, T in enumerate(hours):
    k = int((T - day0) / pd.Timedelta(hours=1))
    ds = xr.open_dataset(fn_list[k])
    t_his = pd.Timestamp(ds.ocean_time.values[0])
    if abs((t_his - T) / pd.Timedelta(minutes=1)) > 1:
        raise ValueError('%s is at %s, expected %s' % (fn_list[k], t_his, T))
    s_rho = ds.s_rho.values
    ox = ds.oxygen.values[0]                                   # (N, NR, NC)
    ds.close()
    for r in rel.values():
        kf = int(round((T - r['t0']) / pd.Timedelta(hours=1)))
        if kf < 0 or kf >= r['nf']:
            continue
        j = r['J'][kf].astype(int); i = r['I'][kf].astype(int)
        ok = (j >= 0) & (r['CS'][kf] > -32768)
        if not ok.any():
            continue
        c = np.clip(r['CS'][kf][ok] / 10000, s_rho[0], s_rho[-1])
        col = ox[:, j[ok], i[ok]]                              # (N, n_ok)
        kk = np.clip(np.searchsorted(s_rho, c), 1, len(s_rho) - 1)
        w = (c - s_rho[kk - 1]) / (s_rho[kk] - s_rho[kk - 1])
        n = np.arange(len(c))
        r['DO'][kf, ok] = ((1 - w) * col[kk - 1, n] + w * col[kk, n]) * 32 / 1000
    if nh % 100 == 0:
        print('  %s  (%d/%d, %.0f s)' % (T, nh + 1, len(hours), time.time() - tic))
        sys.stdout.flush()

for st, r in rel.items():
    d0 = np.nanmax(np.abs(r['DO'][0] - r['DO0'])) if np.isfinite(r['DO0']).any() else np.nan
    meta = dict(sub_tag=st, t0=r['t0'], nf=r['nf'], units='mg/L', ro=args.ro,
                max_abs_diff_vs_DO0=float(d0))
    pickle.dump(dict(DO=r['DO'].astype(np.float16), meta=meta), open(r['out_fn'], 'wb'), protocol=4)
    print('%-16s DO %.2f-%.2f mg/L, frac < 2 mg/L %.3f, |DO[0]-DO0| max %.3g'
          % (st, np.nanmin(r['DO']), np.nanmax(r['DO']), np.nanmean(r['DO'] < 2), d0))
print('done in %.1f min' % ((time.time() - tic) / 60))
