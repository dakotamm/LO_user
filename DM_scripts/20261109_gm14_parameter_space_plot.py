"""
Penn Cove on the Geyer & MacCready (2014) Fig. 6 estuarine parameter space, wb1_t0_xn11abbur00, 2024-2025.

Combines the three earlier steps; computes nothing new:
    20261109_gm14_fig6_digitize.py   GM14 Fig. 6 (regime shading, boundaries, estuary boxes), exact from the PDF
    20261109_gm14_M_axis.py          M at neap / median / spring for the section
    20261109_gm14_Frf_axis.py        monthly Fr_f, (a) local P - E and (b) effective from stratification

LAYOUT
    Axes are widened (M 0.01 - 3, Fr_f 1e-8 - 1) so Penn Cove fits. GM14's own panel is outlined and
    their regime shading and boundaries stay inside it: those boundaries are empirical, so they are
    not extrapolated. The mixing line M = alpha^(1/4) Fr_f^(1/6) is analytic and is continued
    (dashed) outside their panel; left of it the tidal boundary layer does not reach the surface
    within a tide.
    Penn Cove boxes: M from neap to spring (qprism-picked), Fr_f from the smallest to the largest
    positive monthly value, + at (median M, median Fr_f). Box (a)'s 10 net-evaporative months have
    Fr_f < 0 and are marked with a downward arrow from its lower edge.

INPUT  LO_output/DM_outs/20261109_gm14_fig6_digitize/gm14_fig6.p
       LO_output/DM_outs/20261109_gm14_M_axis/M_summary_[sect].csv
       LO_output/DM_outs/20261109_gm14_Frf_axis/Frf_monthly_[sect].csv
OUTPUT LO_output/DM_outs/20261109_gm14_parameter_space_plot/gm14_penn_cove_[sect][_mcol].png

run 20261109_gm14_parameter_space_plot.py -sect pc_lj          (-mcol: M from the fastest column)
"""
import argparse
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-sect', type=str, default='pc_lj')
parser.add_argument('-mcol', action='store_true', help='use M of the fastest column instead of the section mean')
args = parser.parse_args()

M_LIM, F_LIM = (0.01, 3.0), (1e-8, 1.0)
INK, MUTED = '#0b0b0b', '#52514e'

Ldir = Lfun.Lstart()
dm = Ldir['LOo'] / 'DM_outs'
G = pickle.load(open(dm / '20261109_gm14_fig6_digitize' / 'gm14_fig6.p', 'rb'))
Ms = pd.read_csv(dm / '20261109_gm14_M_axis' / ('M_summary_%s.csv' % args.sect), index_col=0)
Fm = pd.read_csv(dm / '20261109_gm14_Frf_axis' / ('Frf_monthly_%s.csv' % args.sect), index_col=0)
out_dir = dm / '20261109_gm14_parameter_space_plot'
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ Penn Cove boxes
mk = 'M_col' if args.mcol else 'M'
M_box = (Ms.loc['neap', mk], Ms.loc['median', mk], Ms.loc['spring', mk])
boxes = {}
for key, col, ls, lab in (('a', 'Fr_local', '--', '(a) local freshwater (net precip.)'),
                          ('b', 'Fr_eff', '-', '(b) effective, from stratification')):
    f = Fm[col]
    fp = f[f > 0]
    boxes[key] = dict(F=(fp.min(), f.median(), fp.max()), n_neg=int((f <= 0).sum()), n=len(f), ls=ls, lab=lab)
    print('%s  M %.3f-%.3f (median %.3f)   Fr_f %.1e-%.1e (median %.1e)   %d/%d months <= 0'
          % ((lab, M_box[0], M_box[2], M_box[1]) + boxes[key]['F'][::2] + (boxes[key]['F'][1], boxes[key]['n_neg'], boxes[key]['n'])))

# ------------------------------------------------------------------ GM14 panel, as digitized
fig, ax = plt.subplots(figsize=(8, 7.5))
for f in G['fills']:
    ax.fill(f['xy'][:, 0], f['xy'][:, 1], color=f['rgb'], lw=0, alpha=0.55)
for ln in G['lines']:
    ax.plot(ln['xy'][:, 0], ln['xy'][:, 1], color=ln['rgb'], lw=1.2, ls=(0, tuple(ln['dash'])) if ln['dash'] else '-')
(gm0, gm1), (gf0, gf1) = G['M_lim'], G['F_lim']
ax.add_patch(plt.Rectangle((gm0, gf0), gm1 - gm0, gf1 - gf0, fill=False, ec=MUTED, lw=0.8))
ax.text(gm1, gf0 / 1.3, 'GM14 Fig. 6 panel', fontsize=7, color=MUTED, ha='right', va='top')
for n, r in G['rects'].items():
    (m0, m1), (f0, f1) = r['M'], r['Fr']
    ax.add_patch(plt.Rectangle((m0, f0), m1 - m0, f1 - f0, fill=False, ec=r['rgb'], lw=0.9))
    ax.text(np.sqrt(m0 * m1), np.sqrt(f0 * f1), n.replace(' River', '').replace(' ', '\n', 1), ha='center',
            va='center', fontsize=5.5, color=MUTED)
for t, (m, f) in G['labels'].items():
    if t != 'Bu':
        ax.text(m, f, t, ha='center', va='center', fontsize=7, weight='bold', color='0.35')

# mixing line: solid inside the GM14 panel, dashed where it is continued
ff = np.logspace(np.log10(F_LIM[0]), 0, 200)
mm = G['mix_c'] * ff ** G['mix_p']
inside = ff >= gf0
ax.plot(mm[inside], ff[inside], color='#e34948', lw=1.5)
ax.plot(mm[~inside], ff[~inside], color='#e34948', lw=1.5, ls='--')
ax.text(G['mix_c'] * 1e-7 ** G['mix_p'] * 1.08, 1e-7, r'$M=\alpha^{1/4}Fr_f^{1/6}$' + '\n(boundary layer\nreaches surface)',
        fontsize=7, color='#c23b3a', va='center')

# ------------------------------------------------------------------ Penn Cove
for key, b in boxes.items():
    f0, fmed, f1 = b['F']
    ax.add_patch(plt.Rectangle((M_box[0], f0), M_box[2] - M_box[0], f1 - f0, fill=False, ec=INK, lw=2, ls=b['ls']))
    ax.plot(M_box[1], fmed, marker='+', ms=10, mew=2, color=INK)
    ax.text(M_box[2] * 1.12, np.sqrt(f0 * f1), 'Penn Cove %s\n%s' % (args.sect, b['lab']), fontsize=8, color=INK, va='center')
    if b['n_neg']:
        ax.annotate('', xy=(M_box[1], f0 / 8), xytext=(M_box[1], f0), arrowprops=dict(arrowstyle='->', color=INK, lw=1.2))
        ax.text(M_box[1] * 1.1, f0 / 5, '%d of %d months\nnet evaporation' % (b['n_neg'], b['n']), fontsize=7, color=MUTED, va='center')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(M_LIM)
ax.set_ylim(F_LIM)
ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2])
ax.set_xticklabels(['0.01', '0.02', '0.05', '0.1', '0.2', '0.5', '1', '2'])
ax.set_xlabel(r'$M = (C_D U_T^2 / \omega N_0 H^2)^{1/2}$')
ax.set_ylabel(r'$Fr_f = U_R / (\beta g s_{ocn} H)^{1/2}$')
ax.set_title('Penn Cove (%s, %s M) in the Geyer & MacCready (2014) parameter space'
             % (args.sect, 'fastest-column' if args.mcol else 'section-mean'), loc='left', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fn = out_dir / ('gm14_penn_cove_%s%s.png' % (args.sect, '_mcol' if args.mcol else ''))
fig.savefig(fn, dpi=200, transparent=True)
plt.close(fig)
print('saved ' + str(fn))
