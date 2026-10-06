"""
Retention curves for the four quadrants of Penn Cove: particles grouped by the
quadrant they STARTED in (inner-N, inner-S, outer-N, outer-S, as defined in
20261005_pcmap_reduce.py), and the curve is always the fraction still inside
the whole COVE (or never having left it), not inside their own quadrant.

  fig 1  one column per quadrant, rows still inside / never left: every
         release thin grey, release mean thick black, release median dashed
  fig 2  a map of the quadrants, then the four quadrant mean curves overlaid
         (still inside, never left)
  fig 3  seasons within each quadrant: one column per quadrant, rows still
         inside / never left, releases thin and season means thick
  fig 4  quadrants within each season: one row per season, columns still
         inside / never left, the four quadrant means overlaid
Seasons are the four-month blocks matched to the Penn Cove oxygen cycle:
Dec-Mar (winter), Apr-Jul (spring), Aug-Nov (low DO); Dec-Mar is Jan-Mar plus
Dec of the same year.

Each release has equal weight in a mean. Only the releases of the
every-3rd-lunar-day table are used (-every 0 for all reduced files).

Output: LO_output/DM_outs/20261006_pcmap_retention_quadrants/<gtx>/

run 20261006_pcmap_retention_quadrants.py
"""
import argparse
import pickle
import re

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-glob', default='pcmap_3d*', help='reduced files to use')
p.add_argument('-every', type=int, default=3, help='release table to keep; 0 = all files')
p.add_argument('-year', type=int, default=2025)
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
red_dir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_reduce' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261006_pcmap_retention_quadrants' / args.gtx
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
QNAMES = ['inner-N', 'inner-S', 'outer-N', 'outer-S']
QCOL = dict(zip(QNAMES, ['#e8455e', '#f0a04b', '#4565e8', '#45b0a8']))

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

C = {q: dict(still=[], never=[]) for q in QNAMES}
npart = {q: [] for q in QNAMES}
seas = []
nrel = 0
for fn in sorted(red_dir.glob(args.glob + '.p')):
    D = pickle.load(open(fn, 'rb'))
    m = re.search(r'_([EF])_(\d{4}\.\d{2}\.\d{2})$', D['meta']['dir'])
    if keep_tags is not None and (not m or '%s_%s' % (m.group(1), m.group(2)) not in keep_tags):
        continue
    nrel += 1
    seas.append(SEASON[pd.Timestamp(D['meta']['t0']).month])
    for q in QNAMES:
        C[q]['still'].append(D['curves'][q]['still'])
        C[q]['never'].append(D['curves'][q]['never'])
        npart[q].append(D['curves'][q]['n'])
if nrel == 0:
    raise SystemExit('no releases found in %s' % red_dir)
nf = min(len(c) for q in QNAMES for c in C[q]['still'])
for q in QNAMES:
    for k in ['still', 'never']:
        C[q][k] = np.array([c[:nf] for c in C[q][k]])
days = np.arange(nf) / 24
seas = np.array(seas)


def efold(c):
    k = np.where(c < 1 / np.e)[0]
    return days[k[0]] if len(k) else np.nan


rows = []
for q in QNAMES:
    r = dict(quadrant=q, particles_per_release=int(np.median(npart[q])))
    for k in ['still', 'never']:
        A = C[q][k]
        ef = np.array([efold(c) for c in A])
        r['%s_efold_of_mean_d' % k] = efold(A.mean(0))
        r['%s_efold_median_d' % k] = np.nanmedian(ef)
        r['%s_efold_p10_d' % k] = np.nanpercentile(ef, 10)
        r['%s_efold_p90_d' % k] = np.nanpercentile(ef, 90)
        r['%s_mean_at_end' % k] = A.mean(0)[-1]
    rows.append(r)
T = pd.DataFrame(rows)
pd.set_option('display.width', 220)
print('%d releases, record %.1f d' % (nrel, days[-1]))
print(T.to_string(index=False, float_format=lambda v: '%.2f' % v))
T.to_csv(out_dir / 'pcmap_retention_quadrants.csv', index=False)

# ------------------------------------------------- fig 1: one per column ---
fig, axs = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)
for r, (k, lab) in enumerate([('still', 'still inside the cove'), ('never', 'never left the cove')]):
    for c, q in enumerate(QNAMES):
        ax = axs[r, c]
        A = C[q][k]
        for cc in A:
            ax.plot(days, cc, color='0.6', lw=0.4, alpha=0.4)
        ax.plot(days, A.mean(0), color='k', lw=2.5, label='mean of %d releases' % len(A))
        ax.plot(days, np.median(A, axis=0), color='k', lw=1.6, ls='--', label='median')
        ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
        ax.set_title('started in %s (~%d per release)\n%s: 1/e of mean %.2f d'
                     % (q, np.median(npart[q]), lab, efold(A.mean(0))), fontsize=10, color=QCOL[q])
        ax.set_xlim(0, days[-1])
        ax.grid(**GRID)
        if r == 1:
            ax.set_xlabel('days from release')
        if c == 0:
            ax.set_ylabel('fraction of particles')
axs[0, 0].set_ylim(0, 1.02)
axs[0, 0].legend(fontsize=8, loc='upper right')
fig.suptitle('%s pcmap retention by starting quadrant, %d releases' % (args.gtx, nrel), fontsize=12)
fig.tight_layout()
fn_out = out_dir / 'pcmap_retention_quadrants_panels.png'
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)

# ------------------------------------------- fig 2: map + overlaid means ---
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat, h, mask = g.lon_rho.values, g.lat_rho.values, g.h.values, g.mask_rho.values
g.close()
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
QUAD = np.full((NR, NC), np.nan)
QUAD[cove] = (2 * (~inner) + (~north))[cove]

fig = plt.figure(figsize=(17, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1])
ax = fig.add_subplot(gs[0])
ax.pcolormesh(lon, lat, np.ma.masked_where(mask == 0, h), cmap='Greys', vmin=0, vmax=150,
              shading='nearest', alpha=0.35)
cm = matplotlib.colors.ListedColormap([QCOL[q] for q in QNAMES])
ax.pcolormesh(lon, lat, np.ma.masked_invalid(QUAD), cmap=cm, vmin=-0.5, vmax=3.5, shading='nearest')
pfun.add_coast(ax, color='k', linewidth=0.6)
pfun.dar(ax)
ax.axis([lon[cove].min() - 0.01, lon[cove].max() + 0.02, lat[cove].min() - 0.01, lat[cove].max() + 0.01])
for q in QNAMES:
    ax.plot([], [], 's', color=QCOL[q], ms=9, label=q)
ax.legend(fontsize=8, loc='lower left', ncol=2)
ax.set_title('starting quadrants', fontsize=10)
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.locator_params(axis='x', nbins=4)
ax.ticklabel_format(useOffset=False)
for c, (k, lab) in enumerate([('still', 'still inside the cove'), ('never', 'never left the cove')]):
    ax = fig.add_subplot(gs[c + 1])
    for q in QNAMES:
        ax.plot(days, C[q][k].mean(0), color=QCOL[q], lw=2.5,
                label='%s (1/e %.2f d)' % (q, efold(C[q][k].mean(0))))
    ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
    ax.set_ylim(0, 1.02); ax.set_xlim(0, days[-1])
    ax.set_title('%s, release mean' % lab, fontsize=10)
    ax.set_xlabel('days from release')
    if c == 0:
        ax.set_ylabel('fraction of particles')
    ax.grid(**GRID)
    ax.legend(fontsize=8, loc='upper right')
fig.suptitle('%s pcmap retention by starting quadrant, %d releases' % (args.gtx, nrel), fontsize=12)
fig.tight_layout()
fn_out = out_dir / 'pcmap_retention_quadrants_means.png'
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)

# ----------------------------------------------------- seasonal tables ---
rowsS = []
for q in QNAMES:
    for sn in SORDER:
        m = seas == sn
        rowsS.append(dict(quadrant=q, season=sn, n=int(m.sum()),
                          still_efold_of_mean_d=efold(C[q]['still'][m].mean(0)),
                          never_efold_of_mean_d=efold(C[q]['never'][m].mean(0)),
                          still_mean_at_end=C[q]['still'][m].mean(0)[-1]))
TS = pd.DataFrame(rowsS)
print('\nby season, 1/e of the release-mean curve [d]:')
print(TS.pivot(index='quadrant', columns='season', values='still_efold_of_mean_d')[SORDER]
      .round(2).to_string().replace('season', 'still inside'))
print(TS.pivot(index='quadrant', columns='season', values='never_efold_of_mean_d')[SORDER]
      .round(2).to_string().replace('season', 'never left  '))
TS.to_csv(out_dir / 'pcmap_retention_quadrants_season.csv', index=False)

# ------------------------------------ fig 3: seasons within each quadrant ---
fig, axs = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)
for r, (k, lab) in enumerate([('still', 'still inside the cove'), ('never', 'never left the cove')]):
    for c, q in enumerate(QNAMES):
        ax = axs[r, c]
        A = C[q][k]
        for cc, sn in zip(A, seas):
            ax.plot(days, cc, color=SCOL[sn], lw=0.3, alpha=0.2)
        txt = []
        for sn in SORDER:
            m = seas == sn
            ax.plot(days, A[m].mean(0), color=SCOL[sn], lw=2.8,
                    label='%s (n %d)' % (SLAB[sn], m.sum()))
            txt.append('%.2f' % efold(A[m].mean(0)))
        ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
        ax.set_title('started in %s, %s\n1/e of season means: %s d'
                     % (q, lab, ' / '.join(txt)), fontsize=9.5, color=QCOL[q])
        ax.set_xlim(0, days[-1])
        ax.grid(**GRID)
        if r == 1:
            ax.set_xlabel('days from release')
        if c == 0:
            ax.set_ylabel('fraction of particles')
axs[0, 0].set_ylim(0, 1.02)
axs[0, 0].legend(fontsize=8, loc='upper right')
fig.suptitle('%s pcmap retention by starting quadrant and season (1/e listed winter / spring / low DO)'
             % args.gtx, fontsize=12)
fig.tight_layout()
fn_out = out_dir / 'pcmap_retention_quadrants_by_season.png'
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)

# ------------------------------------ fig 4: quadrants within each season ---
fig, axs = plt.subplots(len(SORDER), 2, figsize=(12, 3.6 * len(SORDER)), sharex=True, sharey=True)
for r, sn in enumerate(SORDER):
    m = seas == sn
    for c, (k, lab) in enumerate([('still', 'still inside the cove'), ('never', 'never left the cove')]):
        ax = axs[r, c]
        for q in QNAMES:
            ax.plot(days, C[q][k][m].mean(0), color=QCOL[q], lw=2.5,
                    label='%s (1/e %.2f d)' % (q, efold(C[q][k][m].mean(0))))
        ax.axhline(1 / np.e, color='0.4', lw=0.8, ls=':')
        ax.set_title('%s (n %d): %s, release mean' % (SLAB[sn], m.sum(), lab), fontsize=10)
        ax.set_xlim(0, days[-1])
        ax.grid(**GRID)
        ax.legend(fontsize=8, loc='upper right')
        if r == len(SORDER) - 1:
            ax.set_xlabel('days from release')
        if c == 0:
            ax.set_ylabel('fraction of particles')
axs[0, 0].set_ylim(0, 1.02)
fig.suptitle('%s pcmap retention: quadrants compared within each season' % args.gtx, fontsize=12)
fig.tight_layout()
fn_out = out_dir / 'pcmap_retention_quadrants_season_means.png'
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)
