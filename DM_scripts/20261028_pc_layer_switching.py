"""
How the layer arrangement at the Penn Cove mouth switches through a tide, and how
the wind decides it at peak flood and peak ebb.

Input: DM_outs/20261028_pc_mouth_interface/hourly_interface.csv (hourly layer
count and orientation of the section-mean profile, 20261028_pc_mouth_interface.py)
and the hourly cove wind (20260806_wind_reduce.py).

  key  cartoons of the four structures, coloured as in the bars
  a    share of each structure in 8 phases of the tide (phase of the slab flow,
       hourly minus Godin, from its analytic signal), peak flood -> high-water
       slack -> peak ebb -> low-water slack -> peak flood
  b    peak flood (strong-flood stage) by the 6-h mean along-cove wind
  c    peak ebb, the same

Output: LO_output/DM_outs/20261028_pc_layer_switching/pc_layer_switching.png

  python 20261028_pc_layer_switching.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.signal import hilbert

from lo_tools import Lfun, zfun

Ldir = Lfun.Lstart(gridname='wb1')
R = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261028_pc_mouth_interface' / 'hourly_interface.csv', parse_dates=['time'])
out_dir = Ldir['LOo'] / 'DM_outs' / '20261028_pc_layer_switching'
Lfun.make_dir(out_dir)
C_IN, C_OUT = '#0072B2', '#D55E00'
CAT = ['in_over_out', 'out_over_in', 'one', 'three']
CC = {'in_over_out': '#009E73', 'out_over_in': '#CC79A7', 'one': '0.65', 'three': '#E69F00'}
LAB = {'in_over_out': 'surface IN over deep OUT', 'out_over_in': 'surface OUT over deep IN', 'one': 'one layer (all with the tide)', 'three': 'three layers'}
KLAB = {'in_over_out': 'surface IN\nover deep OUT', 'out_over_in': 'surface OUT\nover deep IN', 'one': 'one layer\n(all with the tide)', 'three': 'three\nlayers'}

A = R[R.part == 'all'].set_index('time').sort_index()
A['cat'] = np.select([A.NL == 1, (A.NL == 2) & (A.orient == 'in over out'), (A.NL == 2) & (A.orient == 'out over in'), A.NL == 3],
                     ['one', 'in_over_out', 'out_over_in', 'three'], 'none')
A = A[A.cat != 'none']
U = A.Ubar.values.astype(float)
ut = U - zfun.lowpass(U, f='godin')
ok = np.isfinite(ut)
ph = np.full(len(U), np.nan)
ph[ok] = np.rad2deg(np.angle(hilbert(ut[ok]))) # 0 = peak ebb (max out), +-180 = peak flood; increases with time
A['ph'] = ph
A['pbin'] = (np.round(A.ph / 45) % 8).astype('Int64') # 0 peak ebb, 1 late ebb, 2 LW slack, 3 early flood, 4 peak flood, 5 late flood, 6 HW slack, 7 early ebb
ORDER = [4, 5, 6, 7, 0, 1, 2, 3]
PLAB = {4: 'peak\nflood', 5: 'late\nflood', 6: 'high-water\nslack', 7: 'early\nebb', 0: 'peak\nebb', 1: 'late\nebb', 2: 'low-water\nslack', 3: 'early\nflood'}
W = pd.read_pickle(Ldir['LOo'] / 'DM_outs' / '20260806_wind' / 'wind_hourly_atm00_2024.01.01_2025.12.31.p')['W']
wa = (W.u_pc * -0.9789 + W.v_pc * -0.2043).rolling(6, min_periods=4).mean() # + = into the cove
A = A.join(wa.rename('w'), how='left')
WB = [-20, -3, -1, 1, 3, 20]
WLAB = ['out of the\ncove > 3', 'out\n1-3', 'calm\n< 1', 'into the\ncove 1-3', 'into the\ncove > 3']
A['wbin'] = pd.cut(A.w, WB, labels=range(5))

def shares(g):
    return pd.Series({c: (g.cat == c).mean() for c in CAT})

fig = plt.figure(figsize=(16, 10.5), layout='constrained')
gs = fig.add_gridspec(2, 3, height_ratios=[0.42, 1], width_ratios=[1.5, 1, 1])

# ------------------------------------------------------------------ key
ax = fig.add_subplot(gs[0, :])
def column(ax, x0, cat, w=0.55, H=1.0):
    ax.add_patch(Rectangle((x0, 0), w, H, fc='#EEF4FA', ec=CC[cat], lw=4))
    xc = x0 + w / 2
    def arr(y, inward, L=0.17):
        ax.add_patch(FancyArrowPatch((xc + (L if inward else -L) / 2, y), (xc - (L if inward else -L) / 2, y), arrowstyle='-|>', mutation_scale=14,
                                     color=C_IN if inward else C_OUT, lw=2.5))
    if cat == 'one':
        for y in [0.78, 0.5, 0.22]:
            arr(y, True)
    elif cat == 'in_over_out':
        ax.plot([x0, x0 + w], [0.3, 0.3], 'k-', lw=1.5)
        arr(0.75, True, 0.24); arr(0.52, True, 0.24); arr(0.15, False, 0.12)
    elif cat == 'out_over_in':
        ax.plot([x0, x0 + w], [0.7, 0.7], 'k-', lw=1.5)
        arr(0.85, False, 0.12); arr(0.48, True, 0.24); arr(0.25, True, 0.24)
    else:
        ax.plot([x0, x0 + w], [0.72, 0.72], 'k-', lw=1.5); ax.plot([x0, x0 + w], [0.33, 0.33], 'k-', lw=1.5)
        arr(0.86, True, 0.14); arr(0.52, False, 0.14); arr(0.17, True, 0.14)
    ax.text(xc, -0.06, KLAB[cat], ha='center', va='top', fontsize=10, fontweight='bold', color=CC[cat] if cat != 'one' else '0.35')
for n, c in enumerate(['one', 'in_over_out', 'out_over_in', 'three']):
    column(ax, 0.3 + n * 1.25, c)
ax.text(5.15, 0.75, 'each box is the water column at the mouth (surface on top, bed at the bottom)\n'
        'blue arrow = flowing IN, red = flowing OUT; black line = the divide\n'
        'cartoons drawn for a FLOOD (net inflow); on an ebb every arrow reverses', fontsize=9.5, va='top', color='0.25')
ax.set_xlim(0, 9.5)
ax.set_ylim(-0.45, 1.05)
ax.axis('off')
ax.set_title('Key: the four ways the water column at the mouth can be layered', loc='left', fontsize=11, fontweight='bold')

def stacked(ax, T, labels, title):
    x = np.arange(len(T))
    bot = np.zeros(len(T))
    for c in CAT:
        v = 100 * T[c].values
        ax.bar(x, v, 0.75, bottom=bot, color=CC[c], edgecolor='white', lw=1, label=LAB[c])
        for xi, vi, bi in zip(x, v, bot):
            if vi >= 8:
                ax.text(xi, bi + vi / 2, '%.0f' % vi, ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        bot += v
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel('% of hours')
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold')

ax = fig.add_subplot(gs[1, 0])
T = A.groupby('pbin').apply(shares).reindex(ORDER)
stacked(ax, T, [PLAB[k] for k in ORDER], 'a  Through one tide: the arrangement swaps because the surface\nturns before the deep water (high-water slack: surface already ebbing,\ndeep still flooding; low-water slack: the reverse)')
ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)

for k, stg, title in [(1, 'strong flood', 'b  At PEAK FLOOD, by the wind\n(previous 6-h mean along-cove wind, m/s)\nwind out of the cove holds the surface back'), (2, 'strong ebb', 'c  At PEAK EBB, by the wind\n(the same; cartoon arrows reversed)\nwind into the cove holds the surface back')]:
    ax = fig.add_subplot(gs[1, k])
    g = A[A.stage == stg]
    T = g.groupby('wbin', observed=True).apply(shares)
    nn = g.wbin.value_counts().sort_index()
    stacked(ax, T, ['%s\n(n %d)' % (WLAB[i], nn.get(i, 0)) for i in T.index], title)

fig.suptitle('Which layer is on top at the Penn Cove mouth, through a tide and under different winds (wb1_t0_xn11abbur00, 2024-2025)', fontsize=13)
fn = out_dir / 'pc_layer_switching.png'
fig.savefig(fn, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved ' + str(fn))
for stg in ['strong flood', 'strong ebb']:
    g = A[A.stage == stg]
    print(stg); print((100 * g.groupby('wbin', observed=True).apply(shares)).round(0).to_string())
