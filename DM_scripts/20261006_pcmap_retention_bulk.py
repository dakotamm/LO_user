"""
Bulk retention curves for the pcmap releases: every release as a thin grey
curve, the release-mean curve in thick black and the release-median in
dashed black. Further figures break the releases out by season of release.

SEASONS (-seasons)
  tri  (default) four-month blocks matched to the Penn Cove oxygen cycle
       (2025 monthly bottom DO: Aug-Oct 1.7-3.7 mg/L, Dec-Mar 7-8):
         Aug-Nov  low-DO season
         Dec-Mar  winter
         Apr-Jul  spring (drawdown)
  djf  DJF / MAM / JJA / SON
Either way a winter season is Jan-Mar plus Dec of the SAME year (2025), not a
contiguous winter.

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
  pcmap_retention_bulk_<group>.png          grey + mean + median
  pcmap_retention_bulk_<group>_<seasons>.png
                                            all releases coloured by season,
                                            season means thick
  pcmap_retention_bulk_<group>_<seasons>_panels.png
                                            one season per column (rows: still
                                            inside / never left); that season's
                                            releases coloured with its mean thick
                                            and median dashed, the other seasons
                                            in light grey, the all-release mean
                                            in thin black
  pcmap_retention_bulk_<group>_<seasons>.csv
                                            e-folds, all releases and by season
  pcmap_retention_bulk_<group>_EF.png       ebb (E) vs flood (F) releases, all
                                            seasons together: thin = releases,
                                            thick = set mean, dashed = set median
  pcmap_retention_bulk_<group>_<seasons>_EF_panels.png
                                            the same, one season per column
(the two E/F figures need both sets, i.e. no -set)
  pcmap_retention_bulk_<group>_SN_<terc>.png
                                            spring vs neap releases, all seasons
  pcmap_retention_bulk_<group>_<seasons>_SN_panels.png
                                            the same, one season per column
                                            (only with -sn_terc season)

SPRING / NEAP
Each release is classed by qprism at pc_lp (tef2 bulk_avg, Godin-filtered,
daily) averaged over its first -sn_days (default 3) -- the window in which
most of the flushing happens, so a release is "spring" if spring tides act on
it, not merely if it started on one. qprism, not the sea-level envelope: the
two drift 2-3 d apart in this mixed tide. spring = top third, neap = bottom
third, mid = the rest. -sn_terc year (default) takes the terciles over all
releases; -sn_terc season takes them within each season instead, because
qprism also has a solstitial cycle and year-wide terciles partly sort releases
by season.

STRATIFICATION
  pcmap_retention_bulk_<group>_strat.png    weak vs strong stratification
  pcmap_retention_bulk_<group>_strat_series.png
                                            d(sigma0) through the year (Godin)
                                            with the tercile bands labelled and
                                            each release marked at its window
                                            mean; below, each release's 1/e time
Each release is classed by the bottom-minus-surface density difference
d(sigma0) in Penn Cove: tef2 strat file (width-weighted top and bottom salt and
temperature at each section), sigma0 from gsw with salt ~ SA and temp ~ CT,
averaged over the three cove sections pc_cp, pc_lj, pc_lp and over the same
first -sn_days as qprism. Terciles over the whole year: weak / mid / strong.
Stratification is strongly seasonal, so the printed season x tercile counts
show how much the classes are also sorting by season.
  pcmap_retention_bulk_<group>_<seasons>_strat_season_panels.png
  pcmap_retention_bulk_<group>_<seasons>_strat_season_series.png
The same with terciles taken WITHIN each season, so "weak" means weak for that
season: one column per season, and the time series with each season's own
tercile edges drawn over its months and the season marked along the top.

WIND
  pcmap_retention_bulk_<group>_wind.png, _wind_series.png
  pcmap_retention_bulk_<group>_<seasons>_wind_season_panels.png, _season_series.png
Each release is classed by the along-cove wind w_along (tef2 wind file: daily
mean over the Penn Cove segments, positive INTO the cove, mouth -> head)
averaged over its first -sn_days. Terciles: down-cove (most negative), mid,
up-cove (most positive); year-wide and within season, as for stratification.
Up-cove wind is what reversed the vertical exchange at the mouth in
20260818_pc_exchange_reversal.py.

INITIAL DO
  pcmap_retention_bulk_<group>_DO.png
Every release's curve coloured by the mean DO at release of the particles in
the group (DO0 from the reduce: oxygen at each particle's starting position,
from the history file at the release hour). The release is volume-uniform, so
for -group cove this is the cove volume-mean DO at release.
  pcmap_retention_bulk_<group>_DO_end.png
The same curves coloured by Penn Cove volume-mean DO at the END of each
release's flushing time, t0 + its own still-inside 1/e time, from the daily
lowpassed do_vol_mean of 20260806_hypoxia (pc polygon). The release mean at
t0 from that series is printed against the particle-based initial DO as a
consistency check of the two sources.

run 20261006_pcmap_retention_bulk.py
run 20261006_pcmap_retention_bulk.py -group inner-S-bot
run 20261006_pcmap_retention_bulk.py -set E
run 20261006_pcmap_retention_bulk.py -seasons djf
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
p.add_argument('-seasons', default='tri', choices=['tri', 'djf'])
p.add_argument('-sn_days', type=float, default=3.0,
               help='days after release over which qprism is averaged for spring/neap')
p.add_argument('-sn_terc', default='year', choices=['year', 'season'],
               help='spring/neap terciles over the whole year or within each season')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
red_dir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_reduce' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261006_pcmap_retention_bulk' / args.gtx
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
if args.seasons == 'tri':
    SORDER = ['Dec-Mar', 'Apr-Jul', 'Aug-Nov']
    SEASON = {m: 'Dec-Mar' for m in [12, 1, 2, 3]}
    SEASON.update({m: 'Apr-Jul' for m in [4, 5, 6, 7]})
    SEASON.update({m: 'Aug-Nov' for m in [8, 9, 10, 11]})
    SLAB = {'Dec-Mar': 'Dec-Mar (winter)', 'Apr-Jul': 'Apr-Jul (spring)',
            'Aug-Nov': 'Aug-Nov (low DO)'}
    SCOL = {'Dec-Mar': '#4565e8', 'Apr-Jul': '#45a85b', 'Aug-Nov': '#e8455e'}
else:
    SORDER = ['DJF', 'MAM', 'JJA', 'SON']
    SEASON = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
              6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
    SLAB = {k: k for k in SORDER}
    SCOL = {'DJF': '#4565e8', 'MAM': '#45a85b', 'JJA': '#e8455e', 'SON': '#c68a2e'}

keep_tags = None
if args.every > 0:
    tbl = (Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_release_times'
           / ('pcmap_release_times_%d%s.csv' % (args.year, '_every%d' % args.every if args.every > 1 else '')))
    keep_tags = set(pd.read_csv(tbl).sub_tag)

still, never, t0s, sets, do0 = [], [], [], [], []
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
    P_ = D['P']
    if args.group == 'cove':
        gm = np.ones(len(P_), dtype=bool)
    else:
        q_ = ['inner-N', 'inner-S', 'outer-N', 'outer-S'].index(args.group[:7])
        gm = P_.quad0.values == q_
        if args.group.endswith('-surf'):
            gm &= P_.surf0.values
        elif args.group.endswith('-bot'):
            gm &= ~P_.surf0.values
    do0.append(np.nanmean(P_.DO0.values[gm]) if 'DO0' in P_ else np.nan)
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


seas = np.array([SEASON[t.month] for t in t0s])
rows = []
for sname in ['all'] + SORDER:
    m = np.ones(len(S), dtype=bool) if sname == 'all' else seas == sname
    if m.sum() == 0:
        continue
    for lab, A in [('still inside', S[m]), ('never left', N[m])]:
        ef = np.array([efold(c) for c in A])
        mean = A.mean(axis=0)
        rows.append(dict(season=sname, n=int(m.sum()), curve=lab, efold_of_mean_d=efold(mean),
                         efold_of_median_d=efold(np.median(A, axis=0)), efold_median_d=np.nanmedian(ef),
                         efold_p10_d=np.nanpercentile(ef, 10), efold_p90_d=np.nanpercentile(ef, 90),
                         never_reach_1e=int(np.isnan(ef).sum()),
                         mean_at_7d=mean[7 * 24], mean_at_end=mean[-1]))
T = pd.DataFrame(rows)
print(T.to_string(index=False, float_format=lambda v: '%.2f' % v))
tag = args.group + ('' if args.set == 'all' else '_' + args.set)
T.to_csv(out_dir / ('pcmap_retention_bulk_%s_%s.csv' % (tag, args.seasons)), index=False)

fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, A, lab in zip(axs, [S, N], ['still inside the cove', 'never left the cove']):
    for c in A:
        ax.plot(days, c, color='0.6', lw=0.5, alpha=0.5)
    ax.plot(days, A.mean(axis=0), color='k', lw=2.5, label='mean of %d releases' % len(A))
    ax.plot(days, np.median(A, axis=0), color='k', lw=1.8, ls='--', label='median')
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

# ------------------------------------------------------------ by season ---
fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, A, lab in zip(axs, [S, N], ['still inside the cove', 'never left the cove']):
    for c, sn in zip(A, seas):
        ax.plot(days, c, color=SCOL[sn], lw=0.4, alpha=0.35)
    for sn in SORDER:
        m = seas == sn
        if m.any():
            ax.plot(days, A[m].mean(axis=0), color=SCOL[sn], lw=2.8,
                    label='%s mean (n %d)' % (SLAB[sn], m.sum()))
    ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
    ax.set_title(lab, fontsize=11)
    ax.set_xlabel('days from release')
    ax.set_xlim(0, days[-1])
    ax.grid(**GRID)
axs[0].set_ylim(0, 1.02)
axs[0].set_ylabel('fraction of particles')
axs[0].legend(fontsize=9, loc='upper right')
fig.suptitle('%s pcmap retention by season of release: %s, %s releases'
             % (args.gtx, grp, 'E+F' if args.set == 'all' else args.set), fontsize=12)
fig.tight_layout()
fn_out = out_dir / ('pcmap_retention_bulk_%s_%s.png' % (tag, args.seasons))
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)

# ------------------------------------------------- one panel per season ---
SEAS4 = [sn for sn in SORDER if (seas == sn).any()]
fig, axs = plt.subplots(2, len(SEAS4), figsize=(4.2 * len(SEAS4), 8), sharex=True, sharey=True,
                        squeeze=False)
for r, (A, lab) in enumerate([(S, 'still inside the cove'), (N, 'never left the cove')]):
    for c, sn in enumerate(SEAS4):
        ax = axs[r, c]
        m = seas == sn
        for cc in A[~m]:
            ax.plot(days, cc, color='0.85', lw=0.4)
        for cc in A[m]:
            ax.plot(days, cc, color=SCOL[sn], lw=0.5, alpha=0.5)
        ax.plot(days, A.mean(axis=0), color='k', lw=1.2, label='all-release mean')
        ax.plot(days, A[m].mean(axis=0), color=SCOL[sn], lw=3, label='season mean')
        ax.plot(days, np.median(A[m], axis=0), color=SCOL[sn], lw=1.8, ls='--', label='season median')
        ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
        ef = efold(A[m].mean(axis=0))
        ax.set_title('%s, n %d\n%s: 1/e of mean %.2f d' % (SLAB[sn], m.sum(), lab, ef), fontsize=10)
        ax.set_xlim(0, days[-1])
        ax.grid(**GRID)
        if r == 1:
            ax.set_xlabel('days from release')
        if c == 0:
            ax.set_ylabel('fraction of particles')
        if r == 0:
            ax.legend(fontsize=8, loc='upper right')
axs[0, 0].set_ylim(0, 1.02)
fig.suptitle('%s pcmap retention by season of release: %s, %s releases'
             % (args.gtx, grp, 'E+F' if args.set == 'all' else args.set), fontsize=12)
fig.tight_layout()
fn_out = out_dir / ('pcmap_retention_bulk_%s_%s_panels.png' % (tag, args.seasons))
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)

# --------------------------------------------------------- ebb vs flood ---
SETC = {'E': '#e8455e', 'F': '#4565e8'}
SETL = {'E': 'E (strongest ebb)', 'F': 'F (strongest flood)'}
sets_a = np.array(sets)


def ef_panel(ax, A, ms, lab, ttl, legend):
    txt = []
    for st in ['E', 'F']:
        for cc in A[ms & (sets_a == st)]:
            ax.plot(days, cc, color=SETC[st], lw=0.4, alpha=0.3)
    for st in ['E', 'F']:
        m = ms & (sets_a == st)
        ax.plot(days, A[m].mean(axis=0), color=SETC[st], lw=3,
                label='%s mean (n %d)' % (SETL[st], m.sum()))
        ax.plot(days, np.median(A[m], axis=0), color=SETC[st], lw=1.6, ls='--',
                label='%s median' % st)
        txt.append('%s %.2f' % (st, efold(A[m].mean(axis=0))))
    ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
    ax.set_title('%s\n%s: 1/e of mean %s d' % (ttl, lab, ', '.join(txt)), fontsize=10)
    ax.set_xlim(0, days[-1])
    ax.grid(**GRID)
    if legend:
        ax.legend(fontsize=8, loc='upper right')


if args.set == 'all' and {'E', 'F'} <= set(sets):
    print('\nE vs F, 1/e of the mean curve [d]:')
    for sn in ['all'] + SORDER:
        ms = np.ones(len(S), dtype=bool) if sn == 'all' else seas == sn
        print('  %-8s still E %.2f  F %.2f   never E %.2f  F %.2f'
              % (sn, efold(S[ms & (sets_a == 'E')].mean(0)), efold(S[ms & (sets_a == 'F')].mean(0)),
                 efold(N[ms & (sets_a == 'E')].mean(0)), efold(N[ms & (sets_a == 'F')].mean(0))))

    # all seasons together
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    allm = np.ones(len(S), dtype=bool)
    ef_panel(axs[0], S, allm, 'still inside the cove', 'all releases', True)
    ef_panel(axs[1], N, allm, 'never left the cove', 'all releases', False)
    for ax in axs:
        ax.set_xlabel('days from release')
    axs[0].set_ylim(0, 1.02)
    axs[0].set_ylabel('fraction of particles')
    fig.suptitle('%s pcmap retention, ebb vs flood releases: %s' % (args.gtx, grp), fontsize=12)
    fig.tight_layout()
    fn_out = out_dir / ('pcmap_retention_bulk_%s_EF.png' % tag)
    fig.savefig(fn_out, dpi=200, transparent=True)
    plt.close(fig)
    print('wrote %s' % fn_out)

    # one season per column
    fig, axs = plt.subplots(2, len(SEAS4), figsize=(4.2 * len(SEAS4), 8), sharex=True, sharey=True,
                            squeeze=False)
    for r, (A, lab) in enumerate([(S, 'still inside the cove'), (N, 'never left the cove')]):
        for c, sn in enumerate(SEAS4):
            ef_panel(axs[r, c], A, seas == sn, lab, SLAB[sn], r == 0 and c == 0)
            if r == 1:
                axs[r, c].set_xlabel('days from release')
            if c == 0:
                axs[r, c].set_ylabel('fraction of particles')
    axs[0, 0].set_ylim(0, 1.02)
    fig.suptitle('%s pcmap retention, ebb vs flood releases by season: %s' % (args.gtx, grp),
                 fontsize=12)
    fig.tight_layout()
    fn_out = out_dir / ('pcmap_retention_bulk_%s_%s_EF_panels.png' % (tag, args.seasons))
    fig.savefig(fn_out, dpi=200, transparent=True)
    plt.close(fig)
    print('wrote %s' % fn_out)

# ---------------------------------------------------------- spring/neap ---
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
dq = __import__('xarray').open_dataset(tef2 / 'bulk_avg_2024.01.01_2025.12.31' / 'pc_lp.nc')
tq = pd.to_datetime(dq.time.values); qp = dq.qprism.values
dq.close()
okq = np.isfinite(qp)
th = np.array([np.arange(t, t + pd.Timedelta(days=args.sn_days), pd.Timedelta(hours=1))
               for t in t0s])                                   # hourly over the window
qrel = np.interp(th.astype('datetime64[ns]').astype('int64'),
                 tq[okq].values.astype('int64'), qp[okq]).mean(axis=1)
sn = np.full(len(S), 'mid', dtype=object)
blocks = [np.ones(len(S), dtype=bool)] if args.sn_terc == 'year' else [seas == k for k in SORDER]
for m in blocks:
    if m.sum() < 3:
        continue
    lo, hi = np.percentile(qrel[m], [100 / 3, 200 / 3])
    sn[m & (qrel <= lo)] = 'neap'
    sn[m & (qrel >= hi)] = 'spring'
SNC = {'spring': '#d95f02', 'neap': '#1b9e77', 'mid': '0.65'}
print('\nspring/neap by qprism at pc_lp over the first %g d (terciles over the %s):'
      % (args.sn_days, 'whole year' if args.sn_terc == 'year' else 'season'))
for snm in ['all'] + SORDER:
    ms = np.ones(len(S), dtype=bool) if snm == 'all' else seas == snm
    parts = []
    for k in ['neap', 'mid', 'spring']:
        m = ms & (sn == k)
        parts.append('%s n %2d qprism %4.0f  still %.2f never %.2f'
                     % (k, m.sum(), qrel[m].mean(), efold(S[m].mean(0)), efold(N[m].mean(0))))
    print('  %-8s %s' % (snm, ' | '.join(parts)))
pd.DataFrame(dict(t0=t0s, set=sets, season=seas, qprism_mean=qrel, springneap=sn,
                  efold_still=[efold(c) for c in S], efold_never=[efold(c) for c in N])
             ).to_csv(out_dir / ('pcmap_retention_bulk_%s_%s_SN_releases.csv' % (tag, args.seasons)),
                      index=False)


def sn_panel(ax, A, ms, lab, ttl, legend):
    for k in ['mid', 'neap', 'spring']:
        for cc in A[ms & (sn == k)]:
            ax.plot(days, cc, color=SNC[k], lw=0.4, alpha=0.3 if k != 'mid' else 0.25)
    txt = []
    for k in ['neap', 'spring']:
        m = ms & (sn == k)
        ax.plot(days, A[m].mean(axis=0), color=SNC[k], lw=3,
                label='%s mean (n %d, qprism %.0f)' % (k, m.sum(), qrel[m].mean()))
        ax.plot(days, np.median(A[m], axis=0), color=SNC[k], lw=1.6, ls='--', label='%s median' % k)
        txt.append('%s %.2f' % (k, efold(A[m].mean(axis=0))))
    ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
    ax.set_title('%s, %s\n1/e of mean: %s d' % (ttl, lab, ', '.join(txt)), fontsize=10)
    ax.set_xlim(0, days[-1])
    ax.grid(**GRID)
    if legend:
        ax.plot([], [], color=SNC['mid'], lw=1, label='mid tercile')
        ax.legend(fontsize=7.5, loc='upper right')


fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
allm = np.ones(len(S), dtype=bool)
sn_panel(axs[0], S, allm, 'still inside the cove', 'all releases', True)
sn_panel(axs[1], N, allm, 'never left the cove', 'all releases', False)
for ax in axs:
    ax.set_xlabel('days from release')
axs[0].set_ylim(0, 1.02)
axs[0].set_ylabel('fraction of particles')
fig.suptitle('%s pcmap retention, spring vs neap (qprism terciles over the %s, first %g d): %s'
             % (args.gtx, 'whole year' if args.sn_terc == 'year' else 'season', args.sn_days, grp),
             fontsize=12)
fig.tight_layout()
fn_out = out_dir / ('pcmap_retention_bulk_%s_SN_%s.png' % (tag, args.sn_terc))
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)

if args.sn_terc == 'season':
    fig, axs = plt.subplots(2, len(SEAS4), figsize=(4.2 * len(SEAS4), 8), sharex=True,
                            sharey=True, squeeze=False)
    for r, (A, lab) in enumerate([(S, 'still inside the cove'), (N, 'never left the cove')]):
        for c, snm in enumerate(SEAS4):
            sn_panel(axs[r, c], A, seas == snm, lab, SLAB[snm], r == 0)
            if r == 1:
                axs[r, c].set_xlabel('days from release')
            if c == 0:
                axs[r, c].set_ylabel('fraction of particles')
    axs[0, 0].set_ylim(0, 1.02)
    fig.suptitle('%s pcmap retention, spring vs neap by season: %s' % (args.gtx, grp), fontsize=12)
    fig.tight_layout()
    fn_out = out_dir / ('pcmap_retention_bulk_%s_%s_SN_panels.png' % (tag, args.seasons))
    fig.savefig(fn_out, dpi=200, transparent=True)
    plt.close(fig)
    print('wrote %s' % fn_out)

# ------------------------------------------------------- stratification ---
import gsw
dsx = __import__('xarray').open_dataset(tef2 / 'strat_2024.01.01_2025.12.31_wb1_pc1.nc')
tst = pd.to_datetime(dsx.time.values)
secs = ['pc_cp', 'pc_lj', 'pc_lp']
drho_h = np.nanmean([gsw.sigma0(dsx.s_bot.sel(sect=k).values, dsx.t_bot.sel(sect=k).values)
                     - gsw.sigma0(dsx.s_top.sel(sect=k).values, dsx.t_top.sel(sect=k).values)
                     for k in secs], axis=0)
ds_h = np.nanmean([dsx.dstrat.sel(sect=k).values for k in secs], axis=0)
dsx.close()
okd = np.isfinite(drho_h)
th_ns = th.astype('datetime64[ns]').astype('int64')
drel = np.interp(th_ns, tst[okd].values.astype('int64'), drho_h[okd]).mean(axis=1)
dsrel = np.interp(th_ns, tst[okd].values.astype('int64'), ds_h[okd]).mean(axis=1)
lo, hi = np.percentile(drel, [100 / 3, 200 / 3])
stc = np.where(drel <= lo, 'weak', np.where(drel >= hi, 'strong', 'mid')).astype(object)
STC = {'weak': '#8fbcd4', 'strong': '#08306b', 'mid': '0.65'}
print('\nstratification: d(sigma0) bottom-surface, mean of pc_cp/pc_lj/pc_lp over the first %g d'
      % args.sn_days)
print('  d(sigma0) vs d(salt) across releases r = %.2f; tercile edges %.2f / %.2f kg/m3'
      % (np.corrcoef(drel, dsrel)[0, 1], lo, hi))
for k in ['weak', 'mid', 'strong']:
    m = stc == k
    print('  %-6s n %2d  d(sigma0) %.2f kg/m3  still %.2f  never %.2f  | by season: %s'
          % (k, m.sum(), drel[m].mean(), efold(S[m].mean(0)), efold(N[m].mean(0)),
             '  '.join('%s %d' % (sn_, (m & (seas == sn_)).sum()) for sn_ in SORDER)))
print('  corr(d(sigma0), e-fold) over releases: still r = %+.2f, never r = %+.2f'
      % (np.corrcoef(drel, [efold(c) for c in S])[0, 1], np.corrcoef(drel, [efold(c) for c in N])[0, 1]))
pd.DataFrame(dict(t0=t0s, set=sets, season=seas, drho=drel, dsalt=dsrel, strat=stc,
                  efold_still=[efold(c) for c in S], efold_never=[efold(c) for c in N])
             ).to_csv(out_dir / ('pcmap_retention_bulk_%s_strat_releases.csv' % tag), index=False)

fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, A, lab in zip(axs, [S, N], ['still inside the cove', 'never left the cove']):
    for k in ['mid', 'weak', 'strong']:
        for cc in A[stc == k]:
            ax.plot(days, cc, color=STC[k], lw=0.4, alpha=0.3 if k != 'mid' else 0.25)
    txt = []
    for k in ['weak', 'strong']:
        m = stc == k
        ax.plot(days, A[m].mean(0), color=STC[k], lw=3,
                label='%s mean (n %d, d$\\sigma_0$ %.2f)' % (k, m.sum(), drel[m].mean()))
        ax.plot(days, np.median(A[m], axis=0), color=STC[k], lw=1.6, ls='--', label='%s median' % k)
        txt.append('%s %.2f' % (k, efold(A[m].mean(0))))
    ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
    ax.set_title('all releases, %s\n1/e of mean: %s d' % (lab, ', '.join(txt)), fontsize=10)
    ax.set_xlim(0, days[-1]); ax.set_xlabel('days from release')
    ax.grid(**GRID)
axs[0].plot([], [], color=STC['mid'], lw=1, label='mid tercile')
axs[0].legend(fontsize=7.5, loc='upper right')
axs[0].set_ylim(0, 1.02); axs[0].set_ylabel('fraction of particles')
fig.suptitle('%s pcmap retention by Penn Cove stratification (d$\\sigma_0$ terciles, first %g d): %s'
             % (args.gtx, args.sn_days, grp), fontsize=12)
fig.tight_layout()
fn_out = out_dir / ('pcmap_retention_bulk_%s_strat.png' % tag)
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)

# ------------------------------------------- stratification time series ---
from lo_tools import zfun
yr0, yr1 = pd.Timestamp('%d-01-01' % args.year), pd.Timestamp('%d-01-01' % (args.year + 1))
drho_lp = zfun.lowpass(np.where(okd, drho_h, np.nan), f='godin')
mt = (tst >= yr0) & (tst < yr1)
efs = np.array([efold(c) for c in S]); efn = np.array([efold(c) for c in N])
fig, axs = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
ax = axs[0]
ymax = np.nanpercentile(drho_lp[mt], 99.5) * 1.1
# solid pale fills, not alpha: translucent fills on a transparent savefig
# render as solid colour in some viewers
ax.axhspan(0, lo, color='#e4f0f7', lw=0, zorder=0)
ax.axhspan(hi, ymax, color='#dfe5ef', lw=0, zorder=0)
ax.plot(tst[mt], drho_lp[mt], color='0.25', lw=1, label='cove d$\\sigma_0$ (Godin)')
for v in [lo, hi]:
    ax.axhline(v, color='0.3', lw=0.8, ls='--')
xt = yr1 - pd.Timedelta(days=4)
ax.text(xt, lo / 2, 'weak', ha='right', va='center', fontsize=10, color='#3d7fa6')
ax.text(xt, (lo + hi) / 2, 'mid', ha='right', va='center', fontsize=10, color='0.35')
ax.text(xt, min(hi + (ymax - hi) / 4, ymax * 0.9), 'strong', ha='right', va='center', fontsize=10,
        color=STC['strong'])
for k in ['weak', 'mid', 'strong']:
    m = stc == k
    ax.scatter(np.array(t0s)[m], drel[m], s=18, color=STC[k], edgecolor='k', lw=0.4, zorder=5,
               label='%s releases (n %d)' % (k, m.sum()))
ax.set_ylim(0, ymax)
ax.set_ylabel('d$\\sigma_0$ bottom - surface [kg m$^{-3}$]')
ax.set_title('Penn Cove stratification (mean of pc_cp, pc_lj, pc_lp); markers = release, '
             'at its first-%g-d mean; terciles %.2f / %.2f' % (args.sn_days, lo, hi), fontsize=10)
ax.grid(**GRID)
ax.legend(fontsize=8, loc='upper left', ncol=4)
ax = axs[1]
for k in ['weak', 'mid', 'strong']:
    m = stc == k
    ax.scatter(np.array(t0s)[m], efs[m], s=18, color=STC[k], edgecolor='k', lw=0.4, zorder=5,
               label='still inside, %s' % k)
    ax.scatter(np.array(t0s)[m], efn[m], s=14, marker='^', color=STC[k], edgecolor='0.4', lw=0.3,
               zorder=4)
ax.scatter([], [], s=14, marker='^', color='w', edgecolor='k', lw=0.6, label='never left (triangles)')
ax.set_ylabel('1/e time of release [d]')
ax.set_title('each release: circles = still inside the cove, triangles = never left', fontsize=10)
ax.grid(**GRID)
ax.legend(fontsize=8, loc='upper left', ncol=4)
ax.set_xlim(yr0, yr1)
fig.suptitle('%s pcmap: stratification classes through %d (%s)' % (args.gtx, args.year, grp), fontsize=12)
fig.tight_layout()
fn_out = out_dir / ('pcmap_retention_bulk_%s_strat_series.png' % tag)
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)

# ------------------------------------- stratification, terciles by season ---
stc_s = np.full(len(S), 'mid', dtype=object)
edges_s = {}
for snm in SORDER:
    m = seas == snm
    l_, h_ = np.percentile(drel[m], [100 / 3, 200 / 3])
    edges_s[snm] = (l_, h_)
    stc_s[m & (drel <= l_)] = 'weak'
    stc_s[m & (drel >= h_)] = 'strong'
print('\nstratification terciles WITHIN season:')
for snm in SORDER:
    m = seas == snm
    parts = []
    for k in ['weak', 'mid', 'strong']:
        mk = m & (stc_s == k)
        parts.append('%s n %2d dsig %.2f still %.2f never %.2f'
                     % (k, mk.sum(), drel[mk].mean(), efold(S[mk].mean(0)), efold(N[mk].mean(0))))
    print('  %-8s edges %.2f / %.2f | %s' % (snm, *edges_s[snm], ' | '.join(parts)))

fig, axs = plt.subplots(2, len(SEAS4), figsize=(4.2 * len(SEAS4), 8), sharex=True, sharey=True,
                        squeeze=False)
for r, (A, lab) in enumerate([(S, 'still inside the cove'), (N, 'never left the cove')]):
    for c, snm in enumerate(SEAS4):
        ax = axs[r, c]
        ms = seas == snm
        for k in ['mid', 'weak', 'strong']:
            for cc in A[ms & (stc_s == k)]:
                ax.plot(days, cc, color=STC[k], lw=0.4, alpha=0.35 if k != 'mid' else 0.25)
        txt = []
        for k in ['weak', 'strong']:
            m = ms & (stc_s == k)
            ax.plot(days, A[m].mean(0), color=STC[k], lw=3,
                    label='%s mean (n %d, d$\\sigma_0$ %.2f)' % (k, m.sum(), drel[m].mean()))
            ax.plot(days, np.median(A[m], axis=0), color=STC[k], lw=1.6, ls='--',
                    label='%s median' % k)
            txt.append('%s %.2f' % (k, efold(A[m].mean(0))))
        ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
        ax.set_title('%s, %s\n1/e of mean: %s d' % (SLAB[snm], lab, ', '.join(txt)), fontsize=10)
        ax.set_xlim(0, days[-1])
        ax.grid(**GRID)
        if r == 0:
            ax.plot([], [], color=STC['mid'], lw=1, label='mid tercile')
            ax.legend(fontsize=7.5, loc='upper right')
        if r == 1:
            ax.set_xlabel('days from release')
        if c == 0:
            ax.set_ylabel('fraction of particles')
axs[0, 0].set_ylim(0, 1.02)
fig.suptitle('%s pcmap retention by stratification, terciles within each season: %s' % (args.gtx, grp),
             fontsize=12)
fig.tight_layout()
fn_out = out_dir / ('pcmap_retention_bulk_%s_%s_strat_season_panels.png' % (tag, args.seasons))
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)

# time series with each season's own tercile edges
months = pd.date_range(yr0, yr1, freq='MS')
fig, axs = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
ax = axs[0]
ax.plot(tst[mt], drho_lp[mt], color='0.25', lw=1, label='cove d$\\sigma_0$ (Godin)', zorder=3)
for a, b in zip(months[:-1], months[1:]):
    snm = SEASON[a.month]
    l_, h_ = edges_s[snm]
    ax.fill_between([a, b], 0, l_, color='#e4f0f7', lw=0, zorder=0)
    ax.fill_between([a, b], h_, ymax, color='#dfe5ef', lw=0, zorder=0)
    ax.plot([a, b], [l_, l_], color='0.3', lw=0.8, ls='--', zorder=2)
    ax.plot([a, b], [h_, h_], color='0.3', lw=0.8, ls='--', zorder=2)
    # season strip along the top
    ax.fill_between([a, b], ymax * 0.97, ymax, color=SCOL[snm], lw=0, zorder=4)
for k in ['weak', 'mid', 'strong']:
    m = stc_s == k
    ax.scatter(np.array(t0s)[m], drel[m], s=18, color=STC[k], edgecolor='k', lw=0.4, zorder=5,
               label='%s for its season (n %d)' % (k, m.sum()))
for snm in SORDER:
    ax.plot([], [], 's', color=SCOL[snm], ms=8, label=SLAB[snm])
ax.set_ylim(0, ymax)
ax.set_ylabel('d$\\sigma_0$ bottom - surface [kg m$^{-3}$]')
ax.set_title('Penn Cove stratification; dashed lines = that season\'s tercile edges '
             '(%s); strip on top = season'
             % ', '.join('%s %.2f/%.2f' % (k, *edges_s[k]) for k in SORDER), fontsize=9.5)
ax.grid(**GRID)
ax.legend(fontsize=7.5, loc='upper left', ncol=3)
ax = axs[1]
for k in ['weak', 'mid', 'strong']:
    m = stc_s == k
    ax.scatter(np.array(t0s)[m], efs[m], s=18, color=STC[k], edgecolor='k', lw=0.4, zorder=5,
               label='still inside, %s' % k)
    ax.scatter(np.array(t0s)[m], efn[m], s=14, marker='^', color=STC[k], edgecolor='0.4', lw=0.3,
               zorder=4)
ax.scatter([], [], s=14, marker='^', color='w', edgecolor='k', lw=0.6, label='never left (triangles)')
for a, b in zip(months[:-1], months[1:]):
    ax.axvspan(a, b, ymin=0.97, ymax=1, color=SCOL[SEASON[a.month]], lw=0)
ax.set_ylabel('1/e time of release [d]')
ax.set_title('each release, coloured by its within-season stratification tercile', fontsize=10)
ax.grid(**GRID)
ax.legend(fontsize=8, loc='upper left', ncol=4)
ax.set_xlim(yr0, yr1)
fig.suptitle('%s pcmap: stratification terciles within season, %d (%s)' % (args.gtx, args.year, grp),
             fontsize=12)
fig.tight_layout()
fn_out = out_dir / ('pcmap_retention_bulk_%s_%s_strat_season_series.png' % (tag, args.seasons))
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)

# ------------------------------------------------- generic classification ---
def classify_family(key, t_ser, v_ser, vrel, labs, cols, vname, vunit, ser_title, lp_hours=None):
    """Tercile classes of a per-release forcing value vrel (low / mid / high,
    named by labs), year-wide and within season; four figures, as for
    stratification. t_ser/v_ser is the forcing series drawn in the time
    series (smoothed by a running mean of lp_hours if given)."""
    lo_lab, hi_lab = labs
    order = [lo_lab, 'mid', hi_lab]
    efs_ = np.array([efold(c) for c in S]); efn_ = np.array([efold(c) for c in N])

    def classes(blocks):
        c = np.full(len(S), 'mid', dtype=object)
        edges = []
        for m in blocks:
            l_, h_ = np.percentile(vrel[m], [100 / 3, 200 / 3])
            c[m & (vrel <= l_)] = lo_lab
            c[m & (vrel >= h_)] = hi_lab
            edges.append((l_, h_))
        return c, edges

    cy, (ey,) = classes([np.ones(len(S), dtype=bool)])
    cs_, es_ = classes([seas == k for k in SORDER])
    edges_s_ = dict(zip(SORDER, es_))
    print('\n%s: %s over the first %g d; year-wide tercile edges %.2f / %.2f %s'
          % (key, vname, args.sn_days, ey[0], ey[1], vunit))
    for k in order:
        m = cy == k
        print('  %-10s n %2d  %s %+.2f  still %.2f  never %.2f  | by season: %s'
              % (k, m.sum(), vname, vrel[m].mean(), efold(S[m].mean(0)), efold(N[m].mean(0)),
                 '  '.join('%s %d' % (q, (m & (seas == q)).sum()) for q in SORDER)))
    print('  corr(%s, e-fold) all: still %+.2f never %+.2f | within season: %s'
          % (vname, np.corrcoef(vrel, efs_)[0, 1], np.corrcoef(vrel, efn_)[0, 1],
             '  '.join('%s %+.2f/%+.2f' % (q, np.corrcoef(vrel[seas == q], efs_[seas == q])[0, 1],
                                          np.corrcoef(vrel[seas == q], efn_[seas == q])[0, 1])
                       for q in SORDER)))
    print('  within-season terciles:')
    for q in SORDER:
        ms = seas == q
        print('    %-8s edges %+.2f / %+.2f | %s' % (q, *edges_s_[q], ' | '.join(
            '%s n %d still %.2f never %.2f' % (k, (ms & (cs_ == k)).sum(),
                                              efold(S[ms & (cs_ == k)].mean(0)),
                                              efold(N[ms & (cs_ == k)].mean(0)))
            for k in [lo_lab, hi_lab])))
    pd.DataFrame(dict(t0=t0s, set=sets, season=seas, value=vrel, class_year=cy, class_season=cs_,
                      efold_still=efs_, efold_never=efn_)
                 ).to_csv(out_dir / ('pcmap_retention_bulk_%s_%s_releases.csv' % (tag, key)), index=False)

    def curves(ax, A, ms, cls, lab, ttl, legend):
        for k in ['mid', lo_lab, hi_lab]:
            for cc in A[ms & (cls == k)]:
                ax.plot(days, cc, color=cols[k], lw=0.4, alpha=0.35 if k != 'mid' else 0.25)
        txt = []
        for k in [lo_lab, hi_lab]:
            m = ms & (cls == k)
            ax.plot(days, A[m].mean(0), color=cols[k], lw=3,
                    label='%s mean (n %d, %+.2f %s)' % (k, m.sum(), vrel[m].mean(), vunit))
            ax.plot(days, np.median(A[m], axis=0), color=cols[k], lw=1.6, ls='--', label='%s median' % k)
            txt.append('%s %.2f' % (k, efold(A[m].mean(0))))
        ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
        ax.set_title('%s, %s\n1/e of mean: %s d' % (ttl, lab, ', '.join(txt)), fontsize=10)
        ax.set_xlim(0, days[-1])
        ax.grid(**GRID)
        if legend:
            ax.plot([], [], color=cols['mid'], lw=1, label='mid tercile')
            ax.legend(fontsize=7.5, loc='upper right')

    # year-wide curves
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    allm = np.ones(len(S), dtype=bool)
    curves(axs[0], S, allm, cy, 'still inside the cove', 'all releases', True)
    curves(axs[1], N, allm, cy, 'never left the cove', 'all releases', False)
    for ax in axs:
        ax.set_xlabel('days from release')
    axs[0].set_ylim(0, 1.02); axs[0].set_ylabel('fraction of particles')
    fig.suptitle('%s pcmap retention by %s (terciles over the year, first %g d): %s'
                 % (args.gtx, vname, args.sn_days, grp), fontsize=12)
    fig.tight_layout()
    fn_out = out_dir / ('pcmap_retention_bulk_%s_%s.png' % (tag, key))
    fig.savefig(fn_out, dpi=200, transparent=True); plt.close(fig)
    print('wrote %s' % fn_out)

    # within-season curves
    fig, axs = plt.subplots(2, len(SEAS4), figsize=(4.2 * len(SEAS4), 8), sharex=True, sharey=True,
                            squeeze=False)
    for r, (A, lab) in enumerate([(S, 'still inside the cove'), (N, 'never left the cove')]):
        for c, q in enumerate(SEAS4):
            curves(axs[r, c], A, seas == q, cs_, lab, SLAB[q], r == 0)
            if r == 1:
                axs[r, c].set_xlabel('days from release')
            if c == 0:
                axs[r, c].set_ylabel('fraction of particles')
    axs[0, 0].set_ylim(0, 1.02)
    fig.suptitle('%s pcmap retention by %s, terciles within each season: %s' % (args.gtx, vname, grp),
                 fontsize=12)
    fig.tight_layout()
    fn_out = out_dir / ('pcmap_retention_bulk_%s_%s_%s_season_panels.png' % (tag, args.seasons, key))
    fig.savefig(fn_out, dpi=200, transparent=True); plt.close(fig)
    print('wrote %s' % fn_out)

    # time series, year-wide and within-season edges
    tt_ = pd.to_datetime(t_ser)
    vv = np.asarray(v_ser, dtype=float)
    if lp_hours:
        step_h = (tt_[1] - tt_[0]) / pd.Timedelta(hours=1)
        vv = pd.Series(vv).rolling(max(1, int(round(lp_hours / step_h))), center=True,
                                   min_periods=1).mean().values
    mt_ = (tt_ >= yr0) & (tt_ < yr1)
    lo_y = min(np.nanmin(vv[mt_]), vrel.min()); hi_y = max(np.nanmax(vv[mt_]), vrel.max())
    pad = 0.08 * (hi_y - lo_y); ylo, yhi = lo_y - pad, hi_y + pad
    months_ = pd.date_range(yr0, yr1, freq='MS')
    for mode, cls in [('year', cy), ('season', cs_)]:
        fig, axs = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True,
                                gridspec_kw=dict(height_ratios=[1.3, 1]))
        ax = axs[0]
        if mode == 'year':
            ax.axhspan(ylo, ey[0], color='#e4f0f7', lw=0, zorder=0)
            ax.axhspan(ey[1], yhi, color='#dfe5ef', lw=0, zorder=0)
            for v in ey:
                ax.axhline(v, color='0.3', lw=0.8, ls='--')
            xt = yr1 - pd.Timedelta(days=4)
            ax.text(xt, (ylo + ey[0]) / 2, lo_lab, ha='right', va='center', fontsize=10, color=cols[lo_lab])
            ax.text(xt, sum(ey) / 2, 'mid', ha='right', va='center', fontsize=10, color='0.35')
            ax.text(xt, (ey[1] + yhi) / 2, hi_lab, ha='right', va='center', fontsize=10, color=cols[hi_lab])
        else:
            for a, b in zip(months_[:-1], months_[1:]):
                q = SEASON[a.month]; l_, h_ = edges_s_[q]
                ax.fill_between([a, b], ylo, l_, color='#e4f0f7', lw=0, zorder=0)
                ax.fill_between([a, b], h_, yhi, color='#dfe5ef', lw=0, zorder=0)
                ax.plot([a, b], [l_, l_], color='0.3', lw=0.8, ls='--', zorder=2)
                ax.plot([a, b], [h_, h_], color='0.3', lw=0.8, ls='--', zorder=2)
                ax.fill_between([a, b], yhi - 0.03 * (yhi - ylo), yhi, color=SCOL[q], lw=0, zorder=4)
            for q in SORDER:
                ax.plot([], [], 's', color=SCOL[q], ms=8, label=SLAB[q])
        if ylo < 0 < yhi:
            ax.axhline(0, color='k', lw=0.6)
        ax.plot(tt_[mt_], vv[mt_], color='0.25', lw=1, zorder=3,
                label='%s%s' % (vname, ' (%g h running mean)' % lp_hours if lp_hours else ''))
        for k in order:
            m = cls == k
            ax.scatter(np.array(t0s)[m], vrel[m], s=18, color=cols[k], edgecolor='k', lw=0.4, zorder=5,
                       label='%s%s (n %d)' % (k, '' if mode == 'year' else ' for its season', m.sum()))
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel('%s [%s]' % (vname, vunit))
        ax.set_title('%s; markers = release at its first-%g-d mean; %s' % (
            ser_title, args.sn_days,
            'terciles %+.2f / %+.2f' % tuple(ey) if mode == 'year' else
            'dashed = season tercile edges, strip = season'), fontsize=9.5)
        ax.grid(**GRID)
        ax.legend(fontsize=7.5, loc='upper left', ncol=4)
        ax = axs[1]
        for k in order:
            m = cls == k
            ax.scatter(np.array(t0s)[m], efs_[m], s=18, color=cols[k], edgecolor='k', lw=0.4, zorder=5,
                       label='still inside, %s' % k)
            ax.scatter(np.array(t0s)[m], efn_[m], s=14, marker='^', color=cols[k], edgecolor='0.4',
                       lw=0.3, zorder=4)
        ax.scatter([], [], s=14, marker='^', color='w', edgecolor='k', lw=0.6, label='never left (triangles)')
        if mode == 'season':
            for a, b in zip(months_[:-1], months_[1:]):
                ax.axvspan(a, b, ymin=0.97, ymax=1, color=SCOL[SEASON[a.month]], lw=0)
        ax.set_ylabel('1/e time of release [d]')
        ax.set_title('each release, coloured by its %s tercile' % ('year-wide' if mode == 'year'
                                                                  else 'within-season'), fontsize=10)
        ax.grid(**GRID)
        ax.legend(fontsize=8, loc='upper left', ncol=4)
        ax.set_xlim(yr0, yr1)
        fig.suptitle('%s pcmap: %s classes through %d, terciles %s (%s)'
                     % (args.gtx, vname, args.year, 'over the year' if mode == 'year' else 'within season',
                        grp), fontsize=12)
        fig.tight_layout()
        fn_out = out_dir / (('pcmap_retention_bulk_%s_%s_series.png' % (tag, key)) if mode == 'year' else
                            ('pcmap_retention_bulk_%s_%s_%s_season_series.png' % (tag, args.seasons, key)))
        fig.savefig(fn_out, dpi=200, transparent=True); plt.close(fig)
        print('wrote %s' % fn_out)


# ------------------------------------------------------------------ wind ---
dw = __import__('xarray').open_dataset(tef2 / 'wind_2024.01.01_2025.12.31_wb1_pc1.nc')
tw = pd.to_datetime(dw.day.values) + pd.Timedelta(hours=12)        # daily means, centred at noon
wal = dw.w_along.values
dw.close()
okw = np.isfinite(wal)
wrel = np.interp(th_ns, tw[okw].values.astype('int64'), wal[okw]).mean(axis=1)
classify_family('wind', tw, wal, wrel, ('down-cove', 'up-cove'),
                {'down-cove': '#7b3294', 'up-cove': '#e66101', 'mid': '0.65'},
                'w_along', 'm/s', 'along-cove wind over Penn Cove (+ = into the cove, mouth -> head)',
                lp_hours=72)

# ----------------------------------------------------------- initial DO ---
do0 = np.array(do0)
if np.isfinite(do0).any():
    print('\ninitial DO (mean at release, %s): %.2f - %.2f mg/L; corr with e-fold: still %+.2f, never %+.2f'
          % (grp, np.nanmin(do0), np.nanmax(do0),
             np.corrcoef(do0, [efold(c) for c in S])[0, 1], np.corrcoef(do0, [efold(c) for c in N])[0, 1]))
    cmap_do = plt.get_cmap('RdYlBu')
    norm_do = plt.Normalize(np.floor(np.nanmin(do0)), np.ceil(np.nanmax(do0)))
    order_do = np.argsort(-do0)                      # high DO first, low DO drawn on top
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5), sharey=True)
    for ax, A, lab in zip(axs, [S, N], ['still inside the cove', 'never left the cove']):
        for k in order_do:
            ax.plot(days, A[k], color=cmap_do(norm_do(do0[k])), lw=0.7, alpha=0.8)
        ax.plot(days, A.mean(0), color='k', lw=2.5, label='mean of %d releases' % len(A))
        ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
        ax.set_title('all releases, %s' % lab, fontsize=10)
        ax.set_xlim(0, days[-1]); ax.set_xlabel('days from release')
        ax.grid(**GRID)
    axs[0].set_ylim(0, 1.02); axs[0].set_ylabel('fraction of particles')
    axs[0].legend(fontsize=8, loc='upper right')
    sm = plt.cm.ScalarMappable(norm=norm_do, cmap=cmap_do)
    fig.colorbar(sm, ax=axs, shrink=0.9, pad=0.02,
                 label='mean DO at release, %s [mg/L]' % ('cove volume' if args.group == 'cove' else args.group))
    fig.suptitle('%s pcmap retention coloured by initial DO: %s' % (args.gtx, grp), fontsize=12)
    fn_out = out_dir / ('pcmap_retention_bulk_%s_DO.png' % tag)
    fig.savefig(fn_out, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('wrote %s' % fn_out)

# ------------------------------------------- DO at the end of flushing ---
hyp_fn = Ldir['LOo'] / 'DM_outs' / '20260806_hypoxia' / ('hypoxia_series_%s_pc.csv' % args.gtx)
if hyp_fn.is_file():
    hyp = pd.read_csv(hyp_fn)
    th_ = pd.to_datetime(hyp.time_local, utc=True).dt.tz_localize(None)
    hv = hyp.do_vol_mean.values
    okh = np.isfinite(hv)
    hx = th_[okh].values.astype('datetime64[ns]').astype('int64')
    efs_all = np.array([efold(c) for c in S])
    t_end = np.array([t + pd.Timedelta(days=e) for t, e in zip(t0s, efs_all)])
    do_end = np.interp(t_end.astype('datetime64[ns]').astype('int64'), hx, hv[okh])
    do_t0 = np.interp(np.array(t0s).astype('datetime64[ns]').astype('int64'), hx, hv[okh])
    print('\nDO at end of flushing (pc do_vol_mean at t0 + still-inside 1/e): %.2f - %.2f mg/L; '
          'change over flushing time %+.2f +/- %.2f mg/L'
          % (do_end.min(), do_end.max(), (do_end - do_t0).mean(), (do_end - do_t0).std()))
    if args.group == 'cove':
        print('  consistency: series at t0 vs particle initial DO r = %.3f, mean diff %+.2f mg/L'
              % (np.corrcoef(do_t0, do0)[0, 1], (do_t0 - do0).mean()))
    print('  corr(DO_end, e-fold): still %+.2f, never %+.2f'
          % (np.corrcoef(do_end, efs_all)[0, 1], np.corrcoef(do_end, [efold(c) for c in N])[0, 1]))
    norm_e = plt.Normalize(np.floor(min(do_end.min(), np.nanmin(do0))), np.ceil(max(do_end.max(), np.nanmax(do0))))
    order_e = np.argsort(-do_end)
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5), sharey=True)
    for ax, A, lab in zip(axs, [S, N], ['still inside the cove', 'never left the cove']):
        for k in order_e:
            ax.plot(days, A[k], color=cmap_do(norm_e(do_end[k])), lw=0.7, alpha=0.8)
        ax.plot(days, A.mean(0), color='k', lw=2.5, label='mean of %d releases' % len(A))
        ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
        ax.set_title('all releases, %s' % lab, fontsize=10)
        ax.set_xlim(0, days[-1]); ax.set_xlabel('days from release')
        ax.grid(**GRID)
    axs[0].set_ylim(0, 1.02); axs[0].set_ylabel('fraction of particles')
    axs[0].legend(fontsize=8, loc='upper right')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_e, cmap=cmap_do), ax=axs, shrink=0.9, pad=0.02,
                 label='Penn Cove volume-mean DO at t0 + still-inside 1/e [mg/L]')
    fig.suptitle('%s pcmap retention coloured by DO at the end of the flushing time: %s' % (args.gtx, grp),
                 fontsize=12)
    fn_out = out_dir / ('pcmap_retention_bulk_%s_DO_end.png' % tag)
    fig.savefig(fn_out, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('wrote %s' % fn_out)
