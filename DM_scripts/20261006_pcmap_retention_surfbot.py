"""
Retention curves for the surface vs the bottom half of the Penn Cove water
column: particles grouped by where they STARTED in the vertical (surface if the
initial cs >= -0.5, as in 20261005_pcmap_reduce.py), and the curve is always
the fraction still inside the whole COVE (or never having left it).

Whole-cove surface and bottom curves are the particle-weighted sums of the four
quadrant-half curves stored by the reduce (inner-N-surf, ..., outer-S-bot).

  fig 1  one column per half, rows still inside / never left: every release
         thin grey, release mean thick black, release median dashed
  fig 2  surface and bottom overlaid: means solid, medians dashed
  fig 3  seasons within each half: one column per half, rows still inside /
         never left, releases thin, season means solid, medians dashed
  fig 4  halves within each season: one row per season, surface vs bottom,
         means solid, medians dashed
Seasons are the four-month blocks matched to the Penn Cove oxygen cycle:
Dec-Mar (winter), Apr-Jul (spring), Aug-Nov (low DO); Dec-Mar is Jan-Mar plus
Dec of the same year.

Each release has equal weight in a mean or median. Only the releases of the
every-3rd-lunar-day table are used (-every 0 for all reduced files).

Output: LO_output/DM_outs/20261006_pcmap_retention_surfbot/<gtx>/

run 20261006_pcmap_retention_surfbot.py
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
p.add_argument('-every', type=int, default=3, help='release table to keep; 0 = all files')
p.add_argument('-year', type=int, default=2025)
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
red_dir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_reduce' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261006_pcmap_retention_surfbot' / args.gtx
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
QUADS = ['inner-N', 'inner-S', 'outer-N', 'outer-S']
HALVES = ['surf', 'bot']
HLAB = {'surf': 'surface half', 'bot': 'bottom half'}
HCOL = {'surf': '#f0a04b', 'bot': '#3b0f70'}

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

C = {h: dict(still=[], never=[]) for h in HALVES}
npart = {h: [] for h in HALVES}
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
        ks = ['%s-%s' % (q, h) for q in QUADS if '%s-%s' % (q, h) in D['curves']]
        w = np.array([D['curves'][k]['n'] for k in ks], dtype=float)
        for kk in ['still', 'never']:
            C[h][kk].append(sum(wi * D['curves'][k][kk] for wi, k in zip(w, ks)) / w.sum())
        npart[h].append(w.sum())
if nrel == 0:
    raise SystemExit('no releases found in %s' % red_dir)
nf = min(len(c) for h in HALVES for c in C[h]['still'])
for h in HALVES:
    for kk in ['still', 'never']:
        C[h][kk] = np.array([c[:nf] for c in C[h][kk]])
days = np.arange(nf) / 24
seas = np.array(seas)


def efold(c):
    k = np.where(c < 1 / np.e)[0]
    return days[k[0]] if len(k) else np.nan


rows = []
for h in HALVES:
    r = dict(half=h, particles_per_release=int(np.median(npart[h])))
    for kk in ['still', 'never']:
        A = C[h][kk]
        ef = np.array([efold(c) for c in A])
        r['%s_efold_of_mean_d' % kk] = efold(A.mean(0))
        r['%s_efold_of_median_d' % kk] = efold(np.median(A, axis=0))
        r['%s_efold_median_d' % kk] = np.nanmedian(ef)
        r['%s_efold_p10_d' % kk] = np.nanpercentile(ef, 10)
        r['%s_efold_p90_d' % kk] = np.nanpercentile(ef, 90)
        r['%s_mean_at_end' % kk] = A.mean(0)[-1]
    rows.append(r)
T = pd.DataFrame(rows)
pd.set_option('display.width', 240)
print('%d releases, record %.1f d' % (nrel, days[-1]))
print(T.to_string(index=False, float_format=lambda v: '%.2f' % v))
T.to_csv(out_dir / 'pcmap_retention_surfbot.csv', index=False)

# ------------------------------------------------- fig 1: one per column ---
fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
for r, (kk, lab) in enumerate([('still', 'still inside the cove'), ('never', 'never left the cove')]):
    for c, h in enumerate(HALVES):
        ax = axs[r, c]
        A = C[h][kk]
        for cc in A:
            ax.plot(days, cc, color='0.6', lw=0.4, alpha=0.4)
        ax.plot(days, A.mean(0), color='k', lw=2.5, label='mean of %d releases' % len(A))
        ax.plot(days, np.median(A, axis=0), color='k', lw=1.6, ls='--', label='median')
        ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
        ax.set_title('started in the %s (~%d per release)\n%s: 1/e of mean %.2f, of median %.2f d'
                     % (HLAB[h], np.median(npart[h]), lab, efold(A.mean(0)), efold(np.median(A, axis=0))),
                     fontsize=10, color=HCOL[h])
        ax.set_xlim(0, days[-1])
        ax.grid(**GRID)
        if r == 1:
            ax.set_xlabel('days from release')
        if c == 0:
            ax.set_ylabel('fraction of particles')
axs[0, 0].set_ylim(0, 1.02)
axs[0, 0].legend(fontsize=8, loc='upper right')
fig.suptitle('%s pcmap retention, surface vs bottom half of the column (whole cove), %d releases'
             % (args.gtx, nrel), fontsize=12)
fig.tight_layout()
fn_out = out_dir / 'pcmap_retention_surfbot_panels.png'
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)

# ----------------------------------------------- fig 2: overlaid curves ---
fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, (kk, lab) in zip(axs, [('still', 'still inside the cove'), ('never', 'never left the cove')]):
    for h in HALVES:
        A = C[h][kk]
        ax.plot(days, A.mean(0), color=HCOL[h], lw=2.5,
                label='%s mean (1/e %.2f d)' % (HLAB[h], efold(A.mean(0))))
        ax.plot(days, np.median(A, axis=0), color=HCOL[h], lw=1.6, ls='--',
                label='%s median (1/e %.2f d)' % (HLAB[h], efold(np.median(A, axis=0))))
    ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
    ax.set_ylim(0, 1.02); ax.set_xlim(0, days[-1])
    ax.set_title('%s, across releases' % lab, fontsize=10)
    ax.set_xlabel('days from release')
    ax.grid(**GRID)
    ax.legend(fontsize=8, loc='upper right')
axs[0].set_ylabel('fraction of particles')
fig.suptitle('%s pcmap retention, surface vs bottom half of the column (whole cove), %d releases'
             % (args.gtx, nrel), fontsize=12)
fig.tight_layout()
fn_out = out_dir / 'pcmap_retention_surfbot_means.png'
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)

# ----------------------------------------------------- seasonal tables ---
rowsS = []
for h in HALVES:
    for sn in SORDER:
        m = seas == sn
        r = dict(half=h, season=sn, n=int(m.sum()))
        for kk in ['still', 'never']:
            r['%s_efold_of_mean_d' % kk] = efold(C[h][kk][m].mean(0))
            r['%s_efold_of_median_d' % kk] = efold(np.median(C[h][kk][m], axis=0))
        r['still_mean_at_end'] = C[h]['still'][m].mean(0)[-1]
        rowsS.append(r)
TS = pd.DataFrame(rowsS)
print('\nby season, 1/e of the release-mean (median) curve [d]:')
for kk, lab in [('still', 'still inside'), ('never', 'never left')]:
    print('  %s' % lab)
    for h in HALVES:
        q = TS[TS.half == h].set_index('season')
        print('    %-5s %s' % (h, '   '.join('%s %.2f (%.2f)' % (sn, q.loc[sn, '%s_efold_of_mean_d' % kk],
                                                                q.loc[sn, '%s_efold_of_median_d' % kk])
                                          for sn in SORDER)))
TS.to_csv(out_dir / 'pcmap_retention_surfbot_season.csv', index=False)

# ---------------------------------------- fig 3: seasons within each half ---
fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
for r, (kk, lab) in enumerate([('still', 'still inside the cove'), ('never', 'never left the cove')]):
    for c, h in enumerate(HALVES):
        ax = axs[r, c]
        A = C[h][kk]
        for cc, sn in zip(A, seas):
            ax.plot(days, cc, color=SCOL[sn], lw=0.3, alpha=0.2)
        txt = []
        for sn in SORDER:
            m = seas == sn
            ax.plot(days, A[m].mean(0), color=SCOL[sn], lw=2.8, label='%s mean (n %d)' % (SLAB[sn], m.sum()))
            ax.plot(days, np.median(A[m], axis=0), color=SCOL[sn], lw=1.6, ls='--', label='%s median' % sn)
            txt.append('%.2f' % efold(A[m].mean(0)))
        ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
        ax.set_title('started in the %s, %s\n1/e of season means: %s d' % (HLAB[h], lab, ' / '.join(txt)),
                     fontsize=9.5, color=HCOL[h])
        ax.set_xlim(0, days[-1])
        ax.grid(**GRID)
        if r == 1:
            ax.set_xlabel('days from release')
        if c == 0:
            ax.set_ylabel('fraction of particles')
axs[0, 0].set_ylim(0, 1.02)
axs[0, 0].legend(fontsize=7.5, loc='upper right')
fig.suptitle('%s pcmap retention, surface vs bottom half by season (1/e listed winter / spring / low DO)'
             % args.gtx, fontsize=12)
fig.tight_layout()
fn_out = out_dir / 'pcmap_retention_surfbot_by_season.png'
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)

# ---------------------------------------- fig 4: halves within each season ---
fig, axs = plt.subplots(len(SORDER), 2, figsize=(12, 3.6 * len(SORDER)), sharex=True, sharey=True)
for r, sn in enumerate(SORDER):
    m = seas == sn
    for c, (kk, lab) in enumerate([('still', 'still inside the cove'), ('never', 'never left the cove')]):
        ax = axs[r, c]
        for h in HALVES:
            A = C[h][kk][m]
            ax.plot(days, A.mean(0), color=HCOL[h], lw=2.5,
                    label='%s mean (1/e %.2f d)' % (HLAB[h], efold(A.mean(0))))
            ax.plot(days, np.median(A, axis=0), color=HCOL[h], lw=1.6, ls='--',
                    label='%s median (1/e %.2f d)' % (HLAB[h], efold(np.median(A, axis=0))))
        ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
        ax.set_title('%s (n %d): %s' % (SLAB[sn], m.sum(), lab), fontsize=10)
        ax.set_xlim(0, days[-1])
        ax.grid(**GRID)
        ax.legend(fontsize=7.5, loc='upper right')
        if r == len(SORDER) - 1:
            ax.set_xlabel('days from release')
        if c == 0:
            ax.set_ylabel('fraction of particles')
axs[0, 0].set_ylim(0, 1.02)
fig.suptitle('%s pcmap retention: surface vs bottom half within each season' % args.gtx, fontsize=12)
fig.tight_layout()
fn_out = out_dir / 'pcmap_retention_surfbot_season_means.png'
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)
