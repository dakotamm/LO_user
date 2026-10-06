"""
Bulk retention curves for the pcmap releases: every release as a thin grey
curve and the release-mean curve in thick black.

  left   fraction still inside the cove (re-entry counted: a particle that
         leaves and comes back is inside again)
  right  fraction that has never left (running minimum)

Each release is one curve with equal weight in the mean, so the black curve is
"what a typical release does", not a particle-pooled curve dominated by
whichever releases had the most particles in the group.

-group picks which particles, by where they STARTED (the groups stored by
20261005_pcmap_reduce.py): cove (all), inner-N, inner-S, outer-N, outer-S, or
a quadrant with -surf / -bot, e.g. inner-S-bot. The curve is always "inside
the COVE", whichever group is chosen.

By default only the releases of the every-3rd-lunar-day table are used, so the
extra January releases left over from the original daily run (Jan 2, 3, 5) do
not enter the mean. -every 0 uses every reduced file found.

Output: LO_output/DM_outs/20261006_pcmap_retention_bulk/<gtx>/

run 20261006_pcmap_retention_bulk.py
run 20261006_pcmap_retention_bulk.py -group inner-S-bot
run 20261006_pcmap_retention_bulk.py -set E
"""
import argparse
import pickle
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-glob', default='pcmap_3d*', help='reduced files to use')
p.add_argument('-group', default='cove')
p.add_argument('-set', default='all', choices=['all', 'E', 'F'])
p.add_argument('-every', type=int, default=3, help='release table to keep; 0 = all files')
p.add_argument('-year', type=int, default=2025)
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
red_dir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_reduce' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261006_pcmap_retention_bulk' / args.gtx
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)

keep_tags = None
if args.every > 0:
    tbl = (Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_release_times'
           / ('pcmap_release_times_%d%s.csv' % (args.year, '_every%d' % args.every if args.every > 1 else '')))
    keep_tags = set(pd.read_csv(tbl).sub_tag)

still, never, t0s, sets = [], [], [], []
nf = None
for fn in sorted(red_dir.glob(args.glob + '.p')):
    D = pickle.load(open(fn, 'rb'))
    m = re.search(r'_([EF])_(\d{4}\.\d{2}\.\d{2})$', D['meta']['dir'])
    s = m.group(1) if m else 'other'
    if keep_tags is not None and (not m or '%s_%s' % (s, m.group(2)) not in keep_tags):
        continue
    if args.set != 'all' and s != args.set:
        continue
    if args.group not in D['curves']:
        continue
    c = D['curves'][args.group]
    still.append(c['still']); never.append(c['never'])
    t0s.append(pd.Timestamp(D['meta']['t0'])); sets.append(s)
if not still:
    raise SystemExit('no releases found (glob %s, group %s, set %s, every %d)'
                     % (args.glob, args.group, args.set, args.every))
nf = min(len(c) for c in still)
S = np.array([c[:nf] for c in still]); N = np.array([c[:nf] for c in never])
days = np.arange(nf) / 24
print('%d releases (%s), %s to %s, group %s, record %.1f d'
      % (len(S), ', '.join('%s %d' % (k, sets.count(k)) for k in sorted(set(sets))),
         min(t0s).date(), max(t0s).date(), args.group, days[-1]))


def efold(c):
    k = np.where(c < 1 / np.e)[0]
    return days[k[0]] if len(k) else np.nan


rows = []
for lab, A in [('still inside', S), ('never left', N)]:
    ef = np.array([efold(c) for c in A])
    mean = A.mean(axis=0)
    rows.append(dict(curve=lab, efold_of_mean_d=efold(mean), efold_median_d=np.nanmedian(ef),
                     efold_p10_d=np.nanpercentile(ef, 10), efold_p90_d=np.nanpercentile(ef, 90),
                     never_reach_1e=int(np.isnan(ef).sum()),
                     mean_at_7d=mean[7 * 24], mean_at_end=mean[-1]))
T = pd.DataFrame(rows)
print(T.to_string(index=False, float_format=lambda v: '%.2f' % v))
tag = args.group + ('' if args.set == 'all' else '_' + args.set)
T.to_csv(out_dir / ('pcmap_retention_bulk_%s.csv' % tag), index=False)

fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, A, lab in zip(axs, [S, N], ['still inside the cove', 'never left the cove']):
    for c in A:
        ax.plot(days, c, color='0.6', lw=0.5, alpha=0.5)
    ax.plot(days, A.mean(axis=0), color='k', lw=2.5, label='mean of %d releases' % len(A))
    ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':', label='1/e')
    ax.set_title(lab, fontsize=11)
    ax.set_xlabel('days from release')
    ax.set_xlim(0, days[-1])
    ax.grid(**GRID)
axs[0].set_ylim(0, 1.02)
axs[0].set_ylabel('fraction of particles')
axs[0].legend(fontsize=9, loc='upper right')
grp = 'all particles' if args.group == 'cove' else 'particles starting in %s' % args.group
fig.suptitle('%s pcmap retention: %s, %s releases %s to %s'
             % (args.gtx, grp, 'E+F' if args.set == 'all' else args.set,
                min(t0s).strftime('%Y-%m-%d'), max(t0s).strftime('%Y-%m-%d')), fontsize=12)
fig.tight_layout()
fn_out = out_dir / ('pcmap_retention_bulk_%s.png' % tag)
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)
