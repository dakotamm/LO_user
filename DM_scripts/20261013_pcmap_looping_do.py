"""
Are the pcmap loopers the low-DO particles?

Joins each particle's starting dissolved oxygen (DO0 [mg/L], from the history
file at the release hour; 20261005_pcmap_reduce.py) to the looping table of
20261013_pcmap_looping.py (looper = |loops| >= 1 on the tidal-day-mean path
before first exit).

Absolute DO is confounded with season and place (the inner-cove bottom is the
lowest-DO water AND where the gyre is), so DO is ranked WITHIN each release:
  DOrel   tercile of DO0 among that release's particles (low / mid / high),
          the same idea as -by DOrel in 20261005_pcmap_partition.py
  DOrel_q the same, but within release x starting quadrant x starting half,
          so it compares particles that started in the same place

Output: LO_output/DM_outs/20261013_pcmap_looping_do/
  pcmap_looping_do.png    fraction loopers by DO tercile; columns = all
      releases / Aug-Nov only; rows = DOrel (whole cove, by starting half) /
      DOrel_q (by starting quadrant, surface and bottom starters)
  pcmap_looping_do.csv    the numbers, plus median DO0 of loopers vs
      non-loopers

  python 20261013_pcmap_looping_do.py
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun
from pcmap_regions import QNAMES

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261013_pcmap_looping_do'
Lfun.make_dir(out_dir)
rdir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_reduce' / args.gtx

A = pd.read_pickle(Ldir['LOo'] / 'DM_outs' / '20261013_pcmap_looping' / 'pcmap_looping_particles.p')
do = []
for rel, a in A.groupby('release', sort=False):
    fns = list(rdir.glob(rel + '__*.p'))
    assert len(fns) == 1, rel
    P = pd.read_pickle(fns[0])['P']
    assert len(P) == len(a) and (P.i0.values == a.i0.values).all(), rel
    do.append(pd.Series(P.DO0.values, index=a.index))
A['DO0'] = pd.concat(do)
print('%d particles, DO0 missing %.3f' % (len(A), A.DO0.isna().mean()))
TER = ['low', 'mid', 'high']


def terc(s):
    return pd.qcut(s.rank(method='first'), 3, labels=TER)


A['DOrel'] = A.groupby('release').DO0.transform(terc)
A['DOrel_q'] = A.groupby(['release', 'quad0', 'surf0']).DO0.transform(terc)

rows = []
for per, sel in [('all', np.ones(len(A), bool)), ('Aug-Nov', (A.season == 'Aug-Nov').values)]:
    a = A[sel]
    for half, hs in [('cove', None), ('surf', True), ('bot', False)]:
        b = a if hs is None else a[a.surf0 == hs]
        for t in TER:
            c = b[b.DOrel == t]
            rows.append(dict(period=per, rank='DOrel', half=half, quad='cove', tercile=t, n=len(c), frac_looper=c.looper.mean(),
                             med_DO0=c.DO0.median(), med_first_exit_d=c.first_exit_h.median() / 24))
    for qn, qq in enumerate(QNAMES):
        for half, hs in [('surf', True), ('bot', False)]:
            b = a[(a.quad0 == qn) & (a.surf0 == hs)]
            for t in TER:
                c = b[b.DOrel_q == t]
                rows.append(dict(period=per, rank='DOrel_q', half=half, quad=qq, tercile=t, n=len(c), frac_looper=c.looper.mean(),
                                 med_DO0=c.DO0.median(), med_first_exit_d=c.first_exit_h.median() / 24))
    for half, hs in [('surf', True), ('bot', False)]:
        b = a[a.surf0 == hs]
        # paired by release: median DO0 of loopers minus non-loopers, then the median over releases
        d = b.groupby('release').apply(lambda g: g.DO0[g.looper].median() - g.DO0[~g.looper].median() if g.looper.sum() >= 5 else np.nan)
        rows.append(dict(period=per, rank='looper-minus-nonlooper DO0 (median over releases)', half=half, quad='cove',
                         tercile='', n=int(d.notna().sum()), med_DO0=d.median()))
R = pd.DataFrame(rows)
R.to_csv(out_dir / 'pcmap_looping_do.csv', index=False, float_format='%.4g')
pd.set_option('display.width', 220)
print(R.round(3).to_string(index=False))

# ------------------------------------------------------------------ figure ---
col = {'low': 'tab:red', 'mid': '0.6', 'high': 'tab:blue'}
fig, axs = plt.subplots(2, 2, figsize=(13, 8), layout='constrained', sharey=True)
for cc, per in enumerate(['all', 'Aug-Nov']):
    ax = axs[0, cc]
    groups = ['cove', 'surf', 'bot']
    for k, t in enumerate(TER):
        v = [R[(R.period == per) & (R['rank'] == 'DOrel') & (R.half == h) & (R.tercile == t)].frac_looper.iloc[0] for h in groups]
        ax.bar(np.arange(3) + (k - 1) * 0.27, v, 0.27, color=col[t], label='%s DO (within release)' % t)
    ax.set_xticks(range(3), ['all particles', 'surface starters', 'bottom starters'])
    ax.set_title('%s: DO tercile within each release' % ('All releases' if per == 'all' else 'Aug-Nov releases'), fontsize=11)
    ax = axs[1, cc]
    labs = ['%s\n%s' % (q, h) for q in QNAMES for h in ['surf', 'bot']]
    for k, t in enumerate(TER):
        v = [R[(R.period == per) & (R['rank'] == 'DOrel_q') & (R.quad == q) & (R.half == h) & (R.tercile == t)].frac_looper.iloc[0]
             for q in QNAMES for h in ['surf', 'bot']]
        ax.bar(np.arange(len(labs)) + (k - 1) * 0.27, v, 0.27, color=col[t])
    ax.set_xticks(range(len(labs)), labs, fontsize=8)
    ax.set_title('DO tercile within release x starting quadrant x half', fontsize=11)
for ax in axs.ravel():
    ax.grid(axis='y', color='lightgray', linestyle='--', alpha=0.5)
axs[0, 0].set_ylabel('Fraction loopers (|loops| >= 1)')
axs[1, 0].set_ylabel('Fraction loopers (|loops| >= 1)')
axs[0, 0].legend(fontsize=9)
fig.suptitle('Do the low-DO particles loop? (DO at release, ranked within each release)')
fn = out_dir / 'pcmap_looping_do.png'
fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('saved ' + str(fn))
