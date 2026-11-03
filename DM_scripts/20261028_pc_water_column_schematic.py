"""
Conceptual diagram of the WHOLE water column at the Penn Cove mouth: the mean
exchange and the hourly two-layer divide.

Inputs:
  DM_outs/20261013_maccready2011_sec2_avg/pc_lp_sec2.nc   u1 (tidally averaged
      exchange velocity per element) at pc_lp, + = INTO the cove
  DM_outs/20261028_pc_mouth_interface/hourly_interface.csv  hourly layer count,
      divide depth below the surface, orientation and layer velocities
      (20261028_pc_mouth_interface.py; + = OUT there)

  a  the 2-year mean exchange is SIDE BY SIDE through the whole column: in on
     the north, out on the south; its divide (u1 = 0) is near-vertical
  b  the hourly flow is STACKED: in two-layer hours the divide sits ~10 m down
     (median and IQR shown), half the water depth
  c  through a tide: for each stage, how often the column is one / two / three
     layers, and for the two-layer hours the two arrangements (surface in over
     deep out, or surface out over deep in) with their divide depth and layer
     speeds
  d  the divide depth barely moves with spring/neap, tropic/equatorial or season

Output: LO_output/DM_outs/20261028_pc_water_column_schematic/pc_water_column_schematic.png

  python 20261028_pc_water_column_schematic.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import cmocean

from lo_tools import Lfun

Ldir = Lfun.Lstart(gridname='wb1')
S2 = xr.open_dataset(Ldir['LOo'] / 'DM_outs' / '20261013_maccready2011_sec2_avg' / 'pc_lp_sec2.nc')
R = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261028_pc_mouth_interface' / 'hourly_interface.csv', parse_dates=['time'])
out_dir = Ldir['LOo'] / 'DM_outs' / '20261028_pc_water_column_schematic'
Lfun.make_dir(out_dir)
C_IN, C_OUT = '#0072B2', '#D55E00'
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
STAGES = ['strong flood', 'weak flood', 'slack', 'weak ebb', 'strong ebb']

A = R[R.part == 'all']
T2 = A[A.NL == 2]
HSEC = A.Hsec.mean()
D50, D25, D75 = T2.d_int.median(), T2.d_int.quantile(0.25), T2.d_int.quantile(0.75)

fig = plt.figure(figsize=(17, 10.5), layout='constrained')
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05], height_ratios=[1, 1.05])

# ---------------------------------------------------- a: mean exchange (u1)
ok = S2.ok.values.astype(bool)
u1m = np.nanmean(S2.u1.values[ok], axis=0) # (k, p), + = in
z0, h = S2.z0_mean.values, S2.h.values
NK, NP = z0.shape
pp = np.broadcast_to(np.arange(NP)[None, :], (NK, NP))
ax = fig.add_subplot(gs[0, 0])
vmax = np.nanmax(np.abs(u1m)) * 100
cs = ax.pcolormesh(pp, z0, 100 * u1m, cmap=cmocean.cm.balance_r, vmin=-vmax, vmax=vmax, shading='nearest')
ax.contour(pp, z0, u1m, levels=[0], colors='k', linewidths=2)
ax.fill_between(np.arange(NP), -h, -30, color='0.85', zorder=3)
ax.text(1.3, -6, 'IN', color='white', fontsize=16, fontweight='bold', ha='center')
ax.text(9.0, -6, 'OUT', color='white', fontsize=16, fontweight='bold', ha='center')
ax.set_ylim(-28, 0.5)
ax.set_xlim(-0.5, NP - 0.5)
ax.set_xticks([0, NP - 1]); ax.set_xticklabels(['north', 'south'])
ax.set_ylabel('depth [m]')
fig.colorbar(cs, ax=ax, shrink=0.8, label='2-year mean exchange velocity u1 [cm/s]')
ax.set_title('a  On average: SIDE BY SIDE, the whole column\n(in on the north, out on the south; black = divide)', loc='left', fontsize=11, fontweight='bold')

# ---------------------------------------------------- b: the hourly stacked divide
ax = fig.add_subplot(gs[0, 1])
ax.add_patch(Rectangle((0, -HSEC), 1, HSEC, fc='#EEF4FA', ec='0.5'))
ax.axhspan(-D75, -D25, xmin=0.0, xmax=1.0, color='0.7', alpha=0.5)
ax.axhline(-D50, color='k', lw=2.5)
ax.text(1.03, -D50, 'divide %.1f m\n(half the depth)' % D50, va='center', fontsize=10, fontweight='bold')
ax.text(1.03, -D25 + 0.3, 'middle half of hours:\n%.1f-%.1f m' % (D25, D75), va='bottom', fontsize=8.5, color='0.3')
for y, lab in [(-D50 / 2, 'UPPER LAYER'), (-(D50 + HSEC) / 2, 'LOWER LAYER')]:
    ax.text(0.5, y, lab, ha='center', va='center', fontsize=11, fontweight='bold', color='0.35')
    ax.add_patch(FancyArrowPatch((0.25, y - 1.6), (0.75, y - 1.6), arrowstyle='<|-|>', mutation_scale=16, color='0.35', lw=2))
ax.text(0.5, -HSEC - 1.2, 'two-layer %.0f%%, one layer %.0f%%, three layers %.0f%% of hours'
        % (100 * (A.NL == 2).mean(), 100 * (A.NL == 1).mean(), 100 * (A.NL == 3).mean()), ha='center', va='top', fontsize=9)
ax.set_xlim(-0.05, 1.7)
ax.set_ylim(-HSEC - 3.5, 1)
ax.axis('off')
ax.set_title('b  Hour by hour: STACKED, the two layers\nflow opposite ways across a divide ~%.0f m down' % D50, loc='left', fontsize=11, fontweight='bold')

# ---------------------------------------------------- d: robustness of the divide
ax = fig.add_subplot(gs[0, 2])
groups = [('all', T2)] + [(b, T2[T2[b] == 1]) for b in ['neap', 'spring', 'equatorial', 'tropic']] + \
         [(s_, T2[T2.season == s_]) for s_ in ['Dec-Mar', 'Apr-Jul', 'Aug-Nov']] + [(s_, T2[T2.stage == s_]) for s_ in STAGES]
for n, (lab, g) in enumerate(groups):
    m, lo, hi = g.d_int.median(), g.d_int.quantile(0.25), g.d_int.quantile(0.75)
    col = '#CC79A7' if lab in STAGES else ('k' if lab == 'all' else '#009E73')
    ax.plot([n, n], [-lo, -hi], '-', color=col, lw=3, alpha=0.6)
    ax.plot(n, -m, 'o', color=col, ms=7)
ax.axhline(-D50, color='k', lw=0.8, ls=':')
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups], rotation=60, ha='right', fontsize=8)
ax.set_ylabel('divide depth [m] (median, middle half)')
ax.set_ylim(-20, 0)
ax.grid(axis='y', **GRID)
ax.set_title('d  The divide stays near %.0f m through spring/neap,\ntropic/equatorial and season; it deepens on strong floods' % D50, loc='left', fontsize=11, fontweight='bold')

# ---------------------------------------------------- c: through a tide
ax = fig.add_subplot(gs[1, :])
W = 0.38
SC = 0.022 # x units per cm/s
for n, stg in enumerate(STAGES):
    g = A[A.stage == stg]
    x0 = n * 1.0
    ax.text(x0 + 0.5, 3.0, stg, ha='center', fontsize=11, fontweight='bold')
    ax.text(x0 + 0.5, 2.55, 'one %.0f%% | two %.0f%% | three %.0f%%' % (100 * (g.NL == 1).mean(), 100 * (g.NL == 2).mean(), 100 * (g.NL == 3).mean()), ha='center', fontsize=8.5)
    g2 = g[g.NL == 2]
    for m, (orient, xs) in enumerate([('in over out', x0 + 0.06), ('out over in', x0 + 0.54)]):
        go = g2[g2.orient == orient]
        frac = len(go) / max(len(g2), 1)
        d = go.d_int.median() / HSEC # fraction of depth, drawn on a 0..2 box
        ax.add_patch(Rectangle((xs, 0), W, 2.0, fc='#EEF4FA', ec='0.5', lw=1, alpha=0.4 + 0.6 * frac))
        yd = 2.0 * (1 - d)
        ax.plot([xs, xs + W], [yd, yd], 'k-', lw=2)
        xc = xs + W / 2
        for y, val in [((2.0 + yd) / 2, -100 * go.U_up.mean()), (yd / 2, -100 * go.U_lo.mean())]: # + = IN here
            col = C_IN if val > 0 else C_OUT
            L = -val * SC # IN points left
            ax.add_patch(FancyArrowPatch((xc, y), (xc + L, y), arrowstyle='-|>', mutation_scale=13, color=col, lw=2.5, shrinkA=0, shrinkB=0))
            ax.text(xc, y + 0.16, '%.1f' % abs(val), ha='center', fontsize=7.5, color='0.25')
        ax.text(xc, -0.12, '%s\n%.0f%%, divide %.1f m' % (orient, 100 * frac, go.d_int.median()), ha='center', va='top', fontsize=7.5)
ax.set_xlim(-0.1, 5.1)
ax.set_ylim(-0.75, 3.3)
ax.axis('off')
ax.text(-0.08, -0.72, 'boxes = the water column (surface at top, bed at bottom; the black line is the median divide); arrows: blue = IN (left), red = OUT (right), '
        'mean layer speed in cm/s; box shading = how common that arrangement is at that stage. Low-water slack runs the same stages in reverse.',
        fontsize=8.5, color='0.3')
ax.set_title('c  Through a tide (peak flood -> slack -> peak ebb): a ~6-8 cm/s counterflow is always there; the layer moving WITH the tide is thick, '
             'and the opposing layer is squeezed against the surface or the bed', loc='left', fontsize=11, fontweight='bold')

fig.suptitle('The water column at the Penn Cove mouth: side-by-side on average, stacked hour by hour (model wb1_t0_xn11abbur00, 2024-2025)', fontsize=13)
fn = out_dir / 'pc_water_column_schematic.png'
fig.savefig(fn, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved ' + str(fn))
for stg in STAGES:
    g2 = T2[T2.stage == stg]
    print(stg, ', '.join('%s %.0f%%: divide %.1f m, upper %+.1f, lower %+.1f cm/s (+ out)' % (o, 100 * len(go) / len(g2), go.d_int.median(), 100 * go.U_up.mean(), 100 * go.U_lo.mean())
                         for o, go in g2.groupby('orient')))
