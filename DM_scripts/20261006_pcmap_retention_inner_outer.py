"""
Retention curves for particles that started in the INNER vs the OUTER half of
Penn Cove, with the releases classed by stratification and, separately, by
along-cove wind -- the same tercile classes as 20261006_pcmap_retention_bulk.py.

  inner  inner-N + inner-S (tef2 segment pc_cp_m, landward of pc_cp)
  outer  outer-N + outer-S
The curve is the fraction still inside / never having left the whole COVE.
Inner and outer curves are the particle-weighted combination of the two
quadrant curves stored by 20261005_pcmap_reduce.py.

Forcing per release, averaged over its first -win_days (default 3):
  stratification  bottom-minus-surface d(sigma0), mean of pc_cp/pc_lj/pc_lp
                  (tef2 strat file, gsw with salt ~ SA, temp ~ CT)
  wind            w_along (tef2 wind file, + = into the cove, mouth -> head)
Terciles over the whole year: weak / mid / strong and down-cove / mid / up-cove.

  pcmap_retention_inner_outer_strat.png   rows inner / outer, cols still / never
  pcmap_retention_inner_outer_wind.png    the same, classed by wind

Output: LO_output/DM_outs/20261006_pcmap_retention_inner_outer/<gtx>/

run 20261006_pcmap_retention_inner_outer.py
"""
import argparse
import pickle
import re

import gsw
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-glob', default='pcmap_3d*', help='reduced files to use')
p.add_argument('-every', type=int, default=3, help='release table to keep; 0 = all files')
p.add_argument('-year', type=int, default=2025)
p.add_argument('-win_days', type=float, default=3.0, help='forcing averaging window after release')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
red_dir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_reduce' / args.gtx
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261006_pcmap_retention_inner_outer' / args.gtx
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
HALVES = {'inner': ['inner-N', 'inner-S'], 'outer': ['outer-N', 'outer-S']}

keep_tags = None
if args.every > 0:
    tbl = (Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_release_times'
           / ('pcmap_release_times_%d%s.csv' % (args.year, '_every%d' % args.every if args.every > 1 else '')))
    keep_tags = set(pd.read_csv(tbl).sub_tag)

# ------------------------------------------------------- retention curves ---
C = {h: dict(still=[], never=[]) for h in HALVES}
t0s = []
for fn in sorted(red_dir.glob(args.glob + '.p')):
    D = pickle.load(open(fn, 'rb'))
    m = re.search(r'_([EF])_(\d{4}\.\d{2}\.\d{2})$', D['meta']['dir'])
    if keep_tags is not None and (not m or '%s_%s' % (m.group(1), m.group(2)) not in keep_tags):
        continue
    t0s.append(pd.Timestamp(D['meta']['t0']))
    for h, qs in HALVES.items():
        n = np.array([D['curves'][q]['n'] for q in qs], dtype=float)
        for k in ['still', 'never']:
            C[h][k].append(sum(n[i] * D['curves'][q][k] for i, q in enumerate(qs)) / n.sum())
if not t0s:
    raise SystemExit('no releases found in %s' % red_dir)
nf = min(len(c) for h in HALVES for c in C[h]['still'])
for h in HALVES:
    for k in ['still', 'never']:
        C[h][k] = np.array([c[:nf] for c in C[h][k]])
days = np.arange(nf) / 24
print('%d releases, record %.1f d' % (len(t0s), days[-1]))


def efold(c):
    k = np.where(c < 1 / np.e)[0]
    return days[k[0]] if len(k) else np.nan


# ---------------------------------------------------- forcing per release ---
win = np.array([np.arange(t, t + pd.Timedelta(days=args.win_days), pd.Timedelta(hours=1)) for t in t0s])
win_ns = win.astype('datetime64[ns]').astype('int64')


def window_mean(t, v):
    ok = np.isfinite(v)
    return np.interp(win_ns, pd.to_datetime(t)[ok].values.astype('int64'), v[ok]).mean(axis=1)


ds = xr.open_dataset(tef2 / 'strat_2024.01.01_2025.12.31_wb1_pc1.nc')
drho = np.nanmean([gsw.sigma0(ds.s_bot.sel(sect=k).values, ds.t_bot.sel(sect=k).values)
                   - gsw.sigma0(ds.s_top.sel(sect=k).values, ds.t_top.sel(sect=k).values)
                   for k in ['pc_cp', 'pc_lj', 'pc_lp']], axis=0)
strat_rel = window_mean(ds.time.values, drho)
ds.close()
dw = xr.open_dataset(tef2 / 'wind_2024.01.01_2025.12.31_wb1_pc1.nc')
wind_rel = window_mean(pd.to_datetime(dw.day.values) + pd.Timedelta(hours=12), dw.w_along.values)
dw.close()

FAMILIES = [
    ('strat', strat_rel, ('weak', 'strong'), {'weak': '#8fbcd4', 'strong': '#08306b', 'mid': '0.65'},
     'd$\\sigma_0$', 'kg m$^{-3}$', 'Penn Cove stratification'),
    ('wind', wind_rel, ('down-cove', 'up-cove'), {'down-cove': '#7b3294', 'up-cove': '#e66101', 'mid': '0.65'},
     'w_along', 'm/s', 'along-cove wind (+ into the cove)'),
]

for key, vrel, (lo_lab, hi_lab), cols, vname, vunit, ttl_v in FAMILIES:
    lo, hi = np.percentile(vrel, [100 / 3, 200 / 3])
    cls = np.where(vrel <= lo, lo_lab, np.where(vrel >= hi, hi_lab, 'mid')).astype(object)
    print('\n%s: terciles %.2f / %.2f %s' % (key, lo, hi, vunit.replace('$', '').replace('^{-3}', '-3')))
    rows = []
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for r, h in enumerate(HALVES):
        for c, (k, lab) in enumerate([('still', 'still inside the cove'), ('never', 'never left the cove')]):
            ax = axs[r, c]
            A = C[h][k]
            for kk in ['mid', lo_lab, hi_lab]:
                for cc in A[cls == kk]:
                    ax.plot(days, cc, color=cols[kk], lw=0.4, alpha=0.35 if kk != 'mid' else 0.25)
            txt = []
            for kk in [lo_lab, hi_lab]:
                m = cls == kk
                ax.plot(days, A[m].mean(0), color=cols[kk], lw=3,
                        label='%s mean (n %d, %.2f)' % (kk, m.sum(), vrel[m].mean()))
                ax.plot(days, np.median(A[m], axis=0), color=cols[kk], lw=1.6, ls='--',
                        label='%s median' % kk)
                txt.append('%s %.2f' % (kk, efold(A[m].mean(0))))
                rows.append(dict(half=h, curve=k, cls=kk, n=int(m.sum()), mean_forcing=vrel[m].mean(),
                                 efold_of_mean_d=efold(A[m].mean(0))))
            rows.append(dict(half=h, curve=k, cls='mid', n=int((cls == 'mid').sum()),
                             mean_forcing=vrel[cls == 'mid'].mean(),
                             efold_of_mean_d=efold(A[cls == 'mid'].mean(0))))
            ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
            ax.set_title('started in the %s cove, %s\n1/e of mean: %s d' % (h, lab, ', '.join(txt)),
                         fontsize=10)
            ax.set_xlim(0, days[-1])
            ax.grid(**GRID)
            if r == 1:
                ax.set_xlabel('days from release')
            if c == 0:
                ax.set_ylabel('fraction of particles')
            if r == 0 and c == 0:
                ax.plot([], [], color=cols['mid'], lw=1, label='mid tercile')
                ax.legend(fontsize=8, loc='upper right')
    axs[0, 0].set_ylim(0, 1.02)
    fig.suptitle('%s pcmap retention, inner vs outer starting position, by %s (%s terciles over the year, '
                 'first %g d)' % (args.gtx, ttl_v, vname, args.win_days), fontsize=11.5)
    fig.tight_layout()
    fn_out = out_dir / ('pcmap_retention_inner_outer_%s.png' % key)
    fig.savefig(fn_out, dpi=200, transparent=True)
    plt.close(fig)
    T = pd.DataFrame(rows)
    T.to_csv(out_dir / ('pcmap_retention_inner_outer_%s.csv' % key), index=False)
    print(T.pivot_table(index=['half', 'curve'], columns='cls', values='efold_of_mean_d')
          [[lo_lab, 'mid', hi_lab]].round(2).to_string())
    print('wrote %s' % fn_out)
