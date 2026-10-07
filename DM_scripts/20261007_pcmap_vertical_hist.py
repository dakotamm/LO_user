"""
Histograms of particle height in the water column for the pcmap releases --
the full distribution, not just the cohort mean of 20261006_pcmap_vertical.py.
Reads the hist_all / hist_in counts of 20261006_pcmap_vertical_reduce.py.

Height is fractional, cs + 1 (0 = bed, 1 = surface). Each release's counts are
turned into a fraction per bin at each hour, then averaged over releases (each
release equal weight). A distribution spread evenly through the column is
flat at 1 / nbins.

  fig 1  snapshots: fraction per height bin at -snap days, one column per
         cohort (all particles, surface starters, bottom starters), rows inside
         the cove / everywhere. Height on the y axis, so it reads like a
         profile; dashed = even distribution.
  fig 2  time-height maps: the same fraction per bin through the first -days,
         same layout; colour = fraction relative to even (1 = even), so values
         above 1 mean particles concentrating at that height.

The starting distribution is not exactly flat in 10 bins: particles start at
sigma cell centres, about DZ = 2 m apart, so each column contributes a few
discrete heights; the t = 0 snapshot shows that reference.

Inside-only fractions of a release are dropped once fewer than -min_in of the
cohort are inside.

Output: LO_output/DM_outs/20261007_pcmap_vertical_hist/<gtx>/

run 20261007_pcmap_vertical_hist.py
run 20261007_pcmap_vertical_hist.py -glob 'pcret*' -every 0     (mac test)
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
from matplotlib.colors import TwoSlopeNorm

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-glob', default='pcmap_3d*')
p.add_argument('-every', type=int, default=3, help='release table to keep; 0 = all files')
p.add_argument('-year', type=int, default=2025)
p.add_argument('-snap', default='0,1,3,7,14', help='snapshot days')
p.add_argument('-days', type=float, default=14.0, help='length of the time-height maps')
p.add_argument('-min_in', type=int, default=20)
args = p.parse_args()
snaps = [float(x) for x in args.snap.split(',')]

Ldir = Lfun.Lstart(gridname='wb1')
red_dir = Ldir['LOo'] / 'DM_outs' / '20261006_pcmap_vertical_reduce' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261007_pcmap_vertical_hist' / args.gtx
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
COH = [('all', 'all particles'), ('surf', 'surface-half starters'), ('bot', 'bottom-half starters')]
WHERE = [('hist_in', 'inside the cove'), ('hist_all', 'everywhere')]

keep_tags = None
if args.every > 0:
    keep_tags = set(pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_release_times'
                                / ('pcmap_release_times_%d_every3.csv' % args.year)).sub_tag)

F = {(c, w): [] for c, _ in COH for w, _ in WHERE}
nrel = 0
nb = None
for fn in sorted(red_dir.glob(args.glob + '.p')):
    D = pickle.load(open(fn, 'rb'))
    m = re.search(r'_([EF])_(\d{4}\.\d{2}\.\d{2})$', D['meta']['dir'])
    if keep_tags is not None and (not m or '%s_%s' % (m.group(1), m.group(2)) not in keep_tags):
        continue
    if 'hist_all' not in D['groups'].get('all', {}):
        raise SystemExit('%s has no histograms -- rerun 20261006_pcmap_vertical_reduce.py -clobber' % fn.name)
    nrel += 1
    for c, _ in COH:
        for w, _ in WHERE:
            h = D['groups'][c][w].astype(float)
            tot = h.sum(axis=1, keepdims=True)
            frac = np.where(tot > 0, h / np.maximum(tot, 1), np.nan)
            if w == 'hist_in':
                frac[tot[:, 0] < args.min_in] = np.nan
            F[(c, w)].append(frac)
            nb = h.shape[1]
if nrel == 0:
    raise SystemExit('no files in %s' % red_dir)
nf = min(a.shape[0] for v in F.values() for a in v)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)
    M = {k: np.nanmean(np.array([a[:nf] for a in v]), axis=0) for k, v in F.items()}   # (nf, nb)
hrs = np.arange(nf)
edges = np.linspace(0, 1, nb + 1)
mid = 0.5 * (edges[:-1] + edges[1:])
even = 1 / nb
print('%d releases, %d height bins, record %.1f d' % (nrel, nb, (nf - 1) / 24))
for c, clab in COH:
    for w, wlab in WHERE:
        row = ['%gd: %s' % (d, ' '.join('%.2f' % x for x in M[(c, w)][int(round(d * 24))]))
               for d in snaps if int(round(d * 24)) < nf]
        print('%-22s %-15s bed->surface  %s' % (clab, wlab, ' | '.join(row)))

# ------------------------------------------------------ fig 1: snapshots ---
cmap = plt.get_cmap('viridis')
fig, axs = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
for r, (w, wlab) in enumerate(WHERE):
    for c_i, (c, clab) in enumerate(COH):
        ax = axs[r, c_i]
        for k, d in enumerate(snaps):
            it = int(round(d * 24))
            if it >= nf:
                continue
            col = '0.2' if d == 0 else cmap(k / max(len(snaps) - 1, 1))
            ax.plot(np.r_[M[(c, w)][it], M[(c, w)][it][-1]], edges, drawstyle='steps-post',
                    color=col, lw=2.2 if d == 0 else 1.8, ls=':' if d == 0 else '-',
                    label='day %g' % d)
        ax.axvline(even, color='k', lw=1, ls='--')
        ax.set_ylim(0, 1)
        ax.set_title('%s, %s' % (clab, wlab), fontsize=10)
        ax.grid(**GRID)
        if r == 1:
            ax.set_xlabel('fraction of particles in bin')
        if c_i == 0:
            ax.set_ylabel('height in column [0 bed, 1 surface]')
axs[0, 0].legend(fontsize=8, loc='upper right')
fig.suptitle('%s pcmap: distribution of particle height, mean of %d releases (dashed = even)'
             % (args.gtx, nrel), fontsize=12)
fig.tight_layout()
fn_out = out_dir / 'pcmap_vertical_hist_snapshots.png'
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)

# --------------------------------------------------- fig 2: time-height ---
nmap = min(nf, int(round(args.days * 24)) + 1)
fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2.5)
for r, (w, wlab) in enumerate(WHERE):
    for c_i, (c, clab) in enumerate(COH):
        ax = axs[r, c_i]
        pc = ax.pcolormesh(np.r_[hrs[:nmap], nmap] / 24, edges, (M[(c, w)][:nmap] / even).T,
                           cmap='RdBu_r', norm=norm, shading='flat')
        ax.set_title('%s, %s' % (clab, wlab), fontsize=10)
        if r == 1:
            ax.set_xlabel('days from release')
        if c_i == 0:
            ax.set_ylabel('height in column [0 bed, 1 surface]')
fig.colorbar(pc, ax=axs, shrink=0.85, pad=0.01, label='fraction in bin / even fraction (1 = even)')
fig.suptitle('%s pcmap: where in the column the particles are, mean of %d releases'
             % (args.gtx, nrel), fontsize=12)
fn_out = out_dir / 'pcmap_vertical_hist_timeheight.png'
fig.savefig(fn_out, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('wrote %s' % fn_out)
