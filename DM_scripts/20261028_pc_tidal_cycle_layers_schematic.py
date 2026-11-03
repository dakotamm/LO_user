"""
Simplest picture: the layering of the water column at the Penn Cove mouth through
a typical tidal cycle.

Eight phases of the slab flow (hourly minus Godin, phase from the analytic
signal), peak flood -> high-water slack -> peak ebb -> low-water slack. For each
phase the MOST COMMON layer arrangement among the hours at that phase is drawn
(one layer, surface in over deep out, surface out over deep in), with its median
divide depth and mean layer speeds; the share of hours it covers is printed.
Sea level is drawn as the standing wave Penn Cove is (peak flow at mid-tide,
slack at high and low water).

Input: DM_outs/20261028_pc_mouth_interface/hourly_interface.csv
Output: LO_output/DM_outs/20261028_pc_tidal_cycle_layers_schematic/pc_tidal_cycle_layers.png

  python 20261028_pc_tidal_cycle_layers_schematic.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.signal import hilbert

from lo_tools import Lfun, zfun

import textwrap

def wrapcap(s, fig, cpi=13.5):
    """wrap a caption to the figure width (~cpi characters per inch at fontsize 8.5)"""
    return '\n'.join(textwrap.fill(p, int(fig.get_figwidth() * cpi)) for p in s.split('\n'))


Ldir = Lfun.Lstart(gridname='wb1')
R = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261028_pc_mouth_interface' / 'hourly_interface.csv', parse_dates=['time'])
out_dir = Ldir['LOo'] / 'DM_outs' / '20261028_pc_tidal_cycle_layers_schematic'
Lfun.make_dir(out_dir)
C_IN, C_OUT = '#0072B2', '#D55E00'

A = R[R.part == 'all'].set_index('time').sort_index()
A['cat'] = np.select([A.NL == 1, (A.NL == 2) & (A.orient == 'in over out'), (A.NL == 2) & (A.orient == 'out over in'), A.NL == 3],
                     ['one', 'in_over_out', 'out_over_in', 'three'], 'none')
U = A.Ubar.values.astype(float)
ut = U - zfun.lowpass(U, f='godin')
ok = np.isfinite(ut)
ph = np.full(len(U), np.nan)
ph[ok] = np.rad2deg(np.angle(hilbert(ut[ok]))) # 0 = peak ebb, +-180 = peak flood, increasing in time
A['pbin'] = (np.round(ph / 45) % 8)
ORDER = [4, 5, 6, 7, 0, 1, 2, 3]
NAME = {4: 'peak flood', 5: 'late flood', 6: 'high-water\nslack', 7: 'early ebb', 0: 'peak ebb', 1: 'late ebb', 2: 'low-water\nslack', 3: 'early flood'}
H = A.Hsec.mean()

fig, ax = plt.subplots(figsize=(17, 6.6), layout='constrained')
W, GAP = 0.78, 1.0
TOP = 0.0 # water surface at y = 0, bed at y = -H (metres)
for n, b in enumerate(ORDER):
    g = A[A.pbin == b]
    cats = 100 * g.cat.value_counts(normalize=True).reindex(['one', 'in_over_out', 'out_over_in']).fillna(0).sort_values(ascending=False)
    cat, share = cats.index[0], cats.iloc[0]
    tie = (cats.iloc[0] - cats.iloc[1]) < 3
    x0 = n * GAP
    xc = x0 + W / 2
    ax.add_patch(Rectangle((x0, -H), W, H, fc='#EEF4FA', ec='0.4', lw=1.2))
    sel = g[g.cat == cat]
    if cat == 'one':
        v = -100 * sel.Ubar.mean() # + = IN
        layers = [(-H / 2, v)]
    else:
        d = sel.d_int.median()
        ax.plot([x0, x0 + W], [-d, -d], 'k-', lw=2.5)
        layers = [(-d / 2, -100 * sel.U_up.mean()), (-(d + H) / 2, -100 * sel.U_lo.mean())]
        ax.text(x0 + W + 0.02, -d, '%.0f m' % d, va='center', fontsize=8, color='0.3')
    for y, v in layers:
        L = -v * 0.045 # IN (v > 0) points left
        ax.add_patch(FancyArrowPatch((xc - L / 2, y), (xc + L / 2, y), arrowstyle='simple,head_width=9,head_length=9,tail_width=3.5',
                                     color=C_IN if v > 0 else C_OUT, zorder=3))
        ax.text(xc, y + 1.4, '%s %.0f' % ('IN' if v > 0 else 'OUT', abs(v)), ha='center', fontsize=8.5, color=C_IN if v > 0 else C_OUT, fontweight='bold')
    ax.text(xc, -H - 1.0, NAME[b], ha='center', va='top', fontsize=10.5, fontweight='bold')
    TXT = {'one': 'one layer', 'in_over_out': 'surface in / deep out', 'out_over_in': 'surface out / deep in'}
    ax.text(xc, -H - 4.2, '%.0f%% of hours' % share + ('\n(tied with %s, %.0f%%)' % (TXT[cats.index[1]], cats.iloc[1]) if tie else ''),
            ha='center', va='top', fontsize=8, color='0.35')

# sea level (standing wave): mid-tide at peak flood, high water at HW slack, low water at LW slack
xs = np.linspace(-0.2, len(ORDER) * GAP, 300)
xm = np.arange(len(ORDER)) * GAP + W / 2
eta = lambda x: np.sin((x - W / 2) / GAP * np.pi / 4)
ax.plot(xs, 7.0 + 3.0 * eta(xs), '-', color='0.3', lw=2)
ax.plot(xm, 7.0 + 3.0 * eta(xm), 'o', color='0.3', ms=6)
ax.text(-0.25, 7.0, 'sea\nlevel', ha='right', va='center', fontsize=9, color='0.3')
ax.axhline(0, color='#56B4E9', lw=1, xmin=0.02, xmax=0.98)
ax.set_xlim(-0.6, len(ORDER) * GAP + 0.3)
ax.set_ylim(-H - 10.5, 11.5)
ax.axis('off')
fig.text(0.01, 0.0, wrapcap('Each box is the water column at the mouth (about %.0f m deep), showing the most common layering at that point in the tide; arrows: blue = flowing IN, '
        'red = flowing OUT, cm/s; black line = the divide.\nThe surface turns before the deep water, so near each slack the two layers flow opposite ways; the layer moving with the tide is the thick one. '
        'Wind against the surface flow can flip the order.\nThrough each half-cycle the divide starts shallow just after the surface turns and sinks as the layer moving with the tide thickens.' % H, fig), fontsize=9, color='0.25', va='top', ha='left')
fig.suptitle('Layering of the water column at the Penn Cove mouth through a typical tidal cycle (model wb1_t0_xn11abbur00, 2024-2025)', fontsize=13)
fn = out_dir / 'pc_tidal_cycle_layers.png'
fig.savefig(fn, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved ' + str(fn))
