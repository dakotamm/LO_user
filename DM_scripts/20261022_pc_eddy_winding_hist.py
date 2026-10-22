"""
Histograms of the winding number W about the fixed inner-cove eddy centre
(W_fix from 20261021_pc_eddy_winding.py): does the whole distribution shift
counterclockwise when the Eulerian eddy is present? W is continuous, so this
replaces the looper (W >= 1) / non-looper split.

Particles: eligible only (>= -hmin counted hours 200-1200 m from the centre,
before first exit), pcmap every-3rd-lunar-day set (218 releases).
  (a) all releases, grouped by the share of the first 3 d with the eddy present
      (pres3d): absent (0), partial (0-1), present (1)
  (b) the two example releases of 20261022_pc_eddy_loop_spaghetti.py
Density histograms (bin -bw turns), medians marked; W outside [-2, 3] is
piled into the end bins (shares printed).

Output: LO_output/DM_outs/20261022_pc_eddy_winding_hist/winding_hist_<band>.png (+ .csv of medians)

  python 20261022_pc_eddy_winding_hist.py
  python 20261022_pc_eddy_winding_hist.py -band z4-10
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-band', default='lower')
p.add_argument('-region', default='pc_cp')
p.add_argument('-hmin', type=float, default=24)
p.add_argument('-bw', type=float, default=0.1, help='bin width [turns]')
p.add_argument('-rels', default='pcmap_3d_sh15_F_2025.12.12,pcmap_3d_sh18_F_2025.10.02')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
win = Ldir['LOo'] / 'DM_outs' / '20261021_pc_eddy_winding' / args.region / args.band
out_dir = Ldir['LOo'] / 'DM_outs' / '20261022_pc_eddy_winding_hist'
Lfun.make_dir(out_dir)

A = pd.read_pickle(win / ('winding_particles_%s.p' % args.gtx))
keep = set(pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_release_times' /
                       'pcmap_release_times_2025_every3.csv').out_name)
A = A[A.release.str.split('__').str[0].isin(keep)]
E = A[A.h_fix >= args.hmin]
print('%d releases, %d eligible particles of %d' % (A.release.nunique(), len(E), len(A)))

LO, HI = -2.0, 3.0
bins = np.arange(LO, HI + args.bw / 2, args.bw)
BLUE, GREY, ORANGE = '#2a6fdb', '#8a8a8a', '#e8590c'


def hist(ax, w, color, ls, label):
    wc = np.clip(w, LO + 1e-9, HI - 1e-9)
    ax.hist(wc, bins=bins, density=True, histtype='step', color=color, ls=ls, lw=2, label=label)
    md = np.median(w)
    ax.axvline(md, color=color, ls=ls, lw=1.2)
    return dict(label=label.split('\n')[0], n=len(w), median=md, q25=np.percentile(w, 25), q75=np.percentile(w, 75),
                frac_ge1=(w >= 1).mean(), frac_le0=(w <= 0).mean(), below_lo=(w < LO).mean(), above_hi=(w > HI).mean())


rows = []
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), layout='constrained', sharey=True)
G = [('eddy absent all of first 3 d', E.pres3d <= 0.001, ORANGE, '-'),
     ('eddy present part of first 3 d', (E.pres3d > 0.001) & (E.pres3d < 0.999), GREY, '--'),
     ('eddy present all of first 3 d', E.pres3d >= 0.999, BLUE, '-')]
for lab, m, c, ls in G:
    w = E.W_fix[m].values
    nrel = E.release[m].nunique()
    rows.append(dict(panel='a', **hist(axs[0], w, c, ls, '%s\n%d releases, %d particles, median %+.2f'
                                        % (lab, nrel, len(w), np.median(w)))))
axs[0].set_title('(a) all releases, by eddy presence over the first 3 d', fontsize=11, loc='left')
for (r, lab, c) in zip(args.rels.split(','), ['eddy present', 'eddy absent'], [BLUE, ORANGE]):
    m = E.release.str.startswith(r + '__')
    w = E.W_fix[m].values
    rows.append(dict(panel='b', **hist(axs[1], w, c, '-', '%s (%s)\n%d particles, median %+.2f'
                                        % (r.split('_', 3)[-1], lab, len(w), np.median(w)))))
axs[1].set_title('(b) the two example releases', fontsize=11, loc='left')
for ax in axs:
    for x in [0, 1]:
        ax.axvline(x, color='0.25', lw=0.8, ls=':', zorder=0)
    ax.set_xlim(LO, HI)
    ax.set_xlabel('winding number W about the eddy centre [turns; + = counterclockwise]')
    ax.grid(color='0.88', lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc='upper left', frameon=False)
axs[0].set_ylabel('probability density')
fig.suptitle('Net turning of particles about the inner-cove eddy centre (%s band; particles with >= %g h within '
             '200-1200 m before first exit; vertical lines = medians, dotted = 0 and 1 turn)' % (args.band, args.hmin),
             fontsize=10)
fn = out_dir / ('winding_hist_%s.png' % args.band)
fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
S = pd.DataFrame(rows)
S.to_csv(out_dir / ('winding_hist_%s.csv' % args.band), index=False, float_format='%.4g')
print(S.round(3).to_string(index=False))
print('saved ' + str(fn))
