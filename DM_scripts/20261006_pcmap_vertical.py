"""
Vertical fate of the pcmap cohorts: particles that started in the surface vs
the bottom half of the Penn Cove water column, and where in the column they
are afterwards. Reads the files of 20261006_pcmap_vertical_reduce.py.

Height is fractional height in the column (cs + 1: 0 = bed, 1 = surface).
"switched" is the fraction of the cohort now in the other half from where it
started (cs = -0.5 divides the halves).

  fig 1  inside the cove only: rows = cohort mean height / fraction switched,
         columns = surface starters / bottom starters; every release thin grey,
         release mean thick black, release median dashed
  fig 2  the same for every particle wherever it is (cove + outside)
  fig 3  surface vs bottom starters overlaid: rows height / switched, columns
         inside the cove / everywhere; means solid, medians dashed
  figs 4-5  figs 1-2 coloured by season of release (Dec-Mar winter, Apr-Jul
         spring, Aug-Nov low DO): releases thin, season means thick, season
         medians dashed

Inside-only values of a release are dropped (NaN) once fewer than -min_in of
its cohort are still inside, so the late, nearly empty cohorts do not add
noise. Each release has equal weight in a mean or median. Only the releases of
the every-3rd-lunar-day table are used (-every 0 for all reduced files).

Output: LO_output/DM_outs/20261006_pcmap_vertical/<gtx>/

run 20261006_pcmap_vertical.py
"""
import argparse
import pickle
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-glob', default='pcmap_3d*', help='reduced files to use')
p.add_argument('-every', type=int, default=3, help='release table to keep; 0 = all files')
p.add_argument('-year', type=int, default=2025)
p.add_argument('-min_in', type=int, default=20,
               help='inside-only values need at least this many cohort particles inside')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
red_dir = Ldir['LOo'] / 'DM_outs' / '20261006_pcmap_vertical_reduce' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261006_pcmap_vertical' / args.gtx
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
HALVES = ['surf', 'bot']
HLAB = {'surf': 'surface half', 'bot': 'bottom half'}
HCOL = {'surf': '#f0a04b', 'bot': '#3b0f70'}
VARS = ['h_mean_in', 'sw_in', 'h_mean_all', 'sw_all']

keep_tags = None
if args.every > 0:
    tbl = (Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_release_times'
           / ('pcmap_release_times_%d%s.csv' % (args.year, '_every%d' % args.every if args.every > 1 else '')))
    keep_tags = set(pd.read_csv(tbl).sub_tag)

SORDER = ['Dec-Mar', 'Apr-Jul', 'Aug-Nov']
SEASON = {m: 'Dec-Mar' for m in [12, 1, 2, 3]}
SEASON.update({m: 'Apr-Jul' for m in [4, 5, 6, 7]})
SEASON.update({m: 'Aug-Nov' for m in [8, 9, 10, 11]})
SLAB = {'Dec-Mar': 'Dec-Mar (winter)', 'Apr-Jul': 'Apr-Jul (spring)', 'Aug-Nov': 'Aug-Nov (low DO)'}
SCOL = {'Dec-Mar': '#4565e8', 'Apr-Jul': '#45a85b', 'Aug-Nov': '#e8455e'}

V = {h: {v: [] for v in VARS} for h in HALVES}
seas = []
nrel = 0
for fn in sorted(red_dir.glob(args.glob + '.p')):
    D = pickle.load(open(fn, 'rb'))
    m = re.search(r'_([EF])_(\d{4}\.\d{2}\.\d{2})$', D['meta']['dir'])
    if keep_tags is not None and (not m or '%s_%s' % (m.group(1), m.group(2)) not in keep_tags):
        continue
    nrel += 1
    seas.append(SEASON[pd.Timestamp(D['meta']['t0']).month])
    for h in HALVES:
        G = D['groups'][h]
        few = G['n_in'] < args.min_in
        for v in VARS:
            a = G[v].astype(float)
            if v.endswith('_in'):
                a = np.where(few, np.nan, a)
            V[h][v].append(a)
if nrel == 0:
    raise SystemExit('no vertical-reduce files found in %s' % red_dir)
nf = min(len(a) for h in HALVES for a in V[h]['h_mean_all'])
for h in HALVES:
    for v in VARS:
        V[h][v] = np.array([a[:nf] for a in V[h][v]])
days = np.arange(nf) / 24
seas = np.array(seas)


def mmed(A):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(A, axis=0), np.nanmedian(A, axis=0)


def at(A, d):
    return mmed(A)[0][int(round(d * 24))]


print('%d releases, record %.1f d' % (nrel, days[-1]))
rows = []
for h in HALVES:
    for v in VARS:
        rows.append(dict(cohort=h, var=v, at_0d=at(V[h][v], 0), at_1d=at(V[h][v], 1), at_3d=at(V[h][v], 3),
                         at_7d=at(V[h][v], 7), at_end=mmed(V[h][v])[0][-1]))
T = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print('release-mean values:')
print(T.to_string(index=False, float_format=lambda x: '%.2f' % x))
T.to_csv(out_dir / 'pcmap_vertical.csv', index=False)

VLAB = {'h_mean_in': 'cohort mean height\n[0 bed, 1 surface]', 'h_mean_all': 'cohort mean height\n[0 bed, 1 surface]',
        'sw_in': 'fraction now in\nthe other half', 'sw_all': 'fraction now in\nthe other half'}

# ----------------------------------------- figs 1-2: one column per cohort ---
for fig_tag, rows_v, where in [('inside', ['h_mean_in', 'sw_in'], 'particles inside the cove'),
                               ('all', ['h_mean_all', 'sw_all'], 'all particles, inside or outside the cove')]:
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey='row')
    for r, v in enumerate(rows_v):
        for c, h in enumerate(HALVES):
            ax = axs[r, c]
            A = V[h][v]
            for a in A:
                ax.plot(days, a, color='0.6', lw=0.4, alpha=0.4)
            mean, med = mmed(A)
            ax.plot(days, mean, color='k', lw=2.5, label='mean of %d releases' % len(A))
            ax.plot(days, med, color='k', lw=1.6, ls='--', label='median')
            ax.axhline(0.5, color='0.4', lw=0.8, ls=':')
            ax.set_xlim(0, days[-1])
            ax.set_ylim(0, 1)
            ax.grid(**GRID)
            if r == 0:
                ax.set_title('started in the %s\n%s' % (HLAB[h], where), fontsize=10, color=HCOL[h])
            if r == 1:
                ax.set_xlabel('days from release')
            if c == 0:
                ax.set_ylabel(VLAB[v])
    axs[0, 0].legend(fontsize=8, loc='upper right')
    fig.suptitle('%s pcmap vertical fate of surface vs bottom starters (%s), %d releases'
                 % (args.gtx, 'inside the cove only' if fig_tag == 'inside' else 'all particles', nrel),
                 fontsize=12)
    fig.tight_layout()
    fn_out = out_dir / ('pcmap_vertical_panels_%s.png' % fig_tag)
    fig.savefig(fn_out, dpi=200, transparent=True)
    plt.close(fig)
    print('wrote %s' % fn_out)

# --------------------------------------------- fig 3: cohorts overlaid ---
fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey='row')
for c, (vh, vs, where) in enumerate([('h_mean_in', 'sw_in', 'inside the cove'),
                                     ('h_mean_all', 'sw_all', 'everywhere')]):
    for r, v in enumerate([vh, vs]):
        ax = axs[r, c]
        for h in HALVES:
            mean, med = mmed(V[h][v])
            ax.plot(days, mean, color=HCOL[h], lw=2.5, label='%s starters, mean' % HLAB[h])
            ax.plot(days, med, color=HCOL[h], lw=1.6, ls='--', label='%s starters, median' % HLAB[h])
        ax.axhline(0.5, color='0.4', lw=0.8, ls=':')
        ax.set_xlim(0, days[-1]); ax.set_ylim(0, 1)
        ax.grid(**GRID)
        if r == 0:
            ax.set_title('particles %s, across releases' % where, fontsize=10)
        if r == 1:
            ax.set_xlabel('days from release')
        if c == 0:
            ax.set_ylabel(VLAB[v])
axs[0, 0].legend(fontsize=8, loc='upper right')
fig.suptitle('%s pcmap vertical fate: surface vs bottom starters, %d releases' % (args.gtx, nrel), fontsize=12)
fig.tight_layout()
fn_out = out_dir / 'pcmap_vertical_means.png'
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)

# --------------------------------------- figs 4-5: coloured by season ---
for fig_tag, rows_v, where in [('inside', ['h_mean_in', 'sw_in'], 'particles inside the cove'),
                               ('all', ['h_mean_all', 'sw_all'], 'all particles, inside or outside the cove')]:
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey='row')
    for r, v in enumerate(rows_v):
        for c, h in enumerate(HALVES):
            ax = axs[r, c]
            A = V[h][v]
            for a, sn in zip(A, seas):
                ax.plot(days, a, color=SCOL[sn], lw=0.3, alpha=0.2)
            for sn in SORDER:
                mean, med = mmed(A[seas == sn])
                ax.plot(days, mean, color=SCOL[sn], lw=2.8,
                        label='%s mean (n %d)' % (SLAB[sn], (seas == sn).sum()))
                ax.plot(days, med, color=SCOL[sn], lw=1.6, ls='--', label='%s median' % sn)
            ax.axhline(0.5, color='0.4', lw=0.8, ls=':')
            ax.set_xlim(0, days[-1]); ax.set_ylim(0, 1)
            ax.grid(**GRID)
            if r == 0:
                ax.set_title('started in the %s\n%s' % (HLAB[h], where), fontsize=10, color=HCOL[h])
            if r == 1:
                ax.set_xlabel('days from release')
            if c == 0:
                ax.set_ylabel(VLAB[v])
    axs[0, 0].legend(fontsize=7.5, loc='upper right')
    fig.suptitle('%s pcmap vertical fate by season of release (%s), %d releases'
                 % (args.gtx, 'inside the cove only' if fig_tag == 'inside' else 'all particles', nrel),
                 fontsize=12)
    fig.tight_layout()
    fn_out = out_dir / ('pcmap_vertical_panels_%s_season.png' % fig_tag)
    fig.savefig(fn_out, dpi=200, transparent=True)
    plt.close(fig)
    print('wrote %s' % fn_out)

rowsS = []
for h in HALVES:
    for v in VARS:
        for sn in SORDER:
            A = V[h][v][seas == sn]
            rowsS.append(dict(cohort=h, var=v, season=sn, at_1d=at(A, 1), at_3d=at(A, 3), at_7d=at(A, 7),
                              at_end=mmed(A)[0][-1]))
TS = pd.DataFrame(rowsS)
print('\nrelease-mean values by season:')
print(TS.to_string(index=False, float_format=lambda x: '%.2f' % x))
TS.to_csv(out_dir / 'pcmap_vertical_season.csv', index=False)
