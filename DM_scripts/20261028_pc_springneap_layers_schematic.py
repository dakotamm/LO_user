"""
The layering of the water column at the Penn Cove mouth through an average tidal
cycle, drawn twice: on SPRING and NEAP days (-pair springneap, default), or on
TROPIC and EQUATORIAL days (-pair tropeq), or on days of HIGH and LOW
stratification at the mouth (-pair strat: top / bottom third of the daily
section-mean bottom-minus-top salinity, DM_outs/20261027_pc_twolayer_tides/daily.csv;
raw values, so the groups are partly seasonal -- their months are printed).

Same construction as 20261028_pc_tidal_cycle_layers_schematic.py: eight phases of
the slab flow (hourly minus Godin, phase from the analytic signal); for each phase
the most common layer arrangement among that subset's hours, with its median
divide depth and mean layer speeds (total velocity, cm/s), and the share of
hours it covers. Spring / neap days = top / bottom third of the semidiurnal (D2)
index from the band fits (20261027_tidal_ellipses_windowed.py); tropic /
equatorial the same on the diurnal (D1) index. Arrows and depths use one scale in
both rows; the sea-level curves are scaled by each subset's mean box-mean zeta
amplitude in that band (D2 or D1).

Input: DM_outs/20261028_pc_mouth_interface/hourly_interface.csv,
       tidal_ellipses/pc_cove_*_bands.nc
Output: LO_output/DM_outs/20261028_pc_springneap_layers_schematic/pc_<pair>_layers.png

  python 20261028_pc_springneap_layers_schematic.py
  python 20261028_pc_springneap_layers_schematic.py -pair tropeq
  python 20261028_pc_springneap_layers_schematic.py -pair strat
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
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


p = argparse.ArgumentParser()
p.add_argument('-pair', default='springneap', choices=['springneap', 'tropeq', 'strat'])
args = p.parse_args()
PAIR = {'springneap': (['spring', 'neap'], 'D2', 'semidiurnal'), 'tropeq': (['tropic', 'equatorial'], 'D1', 'diurnal'),
        'strat': (['high_strat', 'low_strat'], 'D2', 'semidiurnal')}[args.pair]

GTX, JOB, DS0, DS1 = 'wb1_t0_xn11abbur00', 'pc_cove', '2024.01.01', '2025.12.31'
Ldir = Lfun.Lstart(gridname='wb1')
R = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261028_pc_mouth_interface' / 'hourly_interface.csv', parse_dates=['time'])
BF = xr.open_dataset(Ldir['LOo'] / 'extract' / GTX / 'tidal_ellipses' / ('%s_%s_%s_bands.nc' % (JOB, DS0, DS1)))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261028_pc_springneap_layers_schematic'
Lfun.make_dir(out_dir)
C_IN, C_OUT = '#0072B2', '#D55E00'
ORDER = [4, 5, 6, 7, 0, 1, 2, 3]
NAME = {4: 'peak flood', 5: 'late flood', 6: 'high-water\nslack', 7: 'early ebb', 0: 'peak ebb', 1: 'late ebb', 2: 'low-water\nslack', 3: 'early flood'}
TXT = {'one': 'one layer', 'in_over_out': 'surface in / deep out', 'out_over_in': 'surface out / deep in'}
W, GAP, ASC = 0.78, 1.0, 0.045 # box width, spacing, arrow length per cm/s

A = R[R.part == 'all'].set_index('time').sort_index()
A['cat'] = np.select([A.NL == 1, (A.NL == 2) & (A.orient == 'in over out'), (A.NL == 2) & (A.orient == 'out over in'), A.NL == 3],
                     ['one', 'in_over_out', 'out_over_in', 'three'], 'none')
U = A.Ubar.values.astype(float)
ut = U - zfun.lowpass(U, f='godin')
ok = np.isfinite(ut)
ph = np.full(len(U), np.nan)
ph[ok] = np.rad2deg(np.angle(hilbert(ut[ok])))
A['pbin'] = np.round(ph / 45) % 8
H = A.Hsec.mean()
zB = pd.Series(BF['zidx_' + PAIR[1]].values, index=pd.DatetimeIndex(BF.time.values).normalize())
EXTRA = {}
if args.pair == 'strat':
    DS = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261027_pc_twolayer_tides' / 'daily.csv', index_col=0, parse_dates=True)
    q1, q2 = DS.strat.quantile([1 / 3, 2 / 3])
    days = {'high_strat': DS.index[DS.strat >= q2], 'low_strat': DS.index[DS.strat <= q1]}
    for b in PAIR[0]:
        A[b] = A.index.normalize().isin(days[b]).astype(int)
        EXTRA[b] = 'bottom minus top salinity %.1f g/kg; ' % DS.strat[DS.index.isin(days[b])].mean()
        print(b, 'days by month:', pd.Series(days[b].month).value_counts().sort_index().to_dict())
    AMP = {b: zB.reindex(days[b]).mean() for b in PAIR[0]}
else:
    days = {b: pd.DatetimeIndex(BF.time.values).normalize()[BF['is_' + b].values == 1] for b in PAIR[0]}
    AMP = {b: zB.reindex(days[b]).mean() for b in PAIR[0]}
# both tidal bands for every group, so a tropic day reads as 'semidiurnal X + diurnal Y'
zD = {k: pd.Series(BF['zidx_' + k].values, index=pd.DatetimeIndex(BF.time.values).normalize()) for k in ['D2', 'D1']}
BANDS = {b: 'semidiurnal %.2f m + diurnal %.2f m' % (zD['D2'].reindex(days[b]).mean(), zD['D1'].reindex(days[b]).mean()) for b in PAIR[0]}

def draw_row(ax, S, label, amp):
    rows = []
    for n, b in enumerate(ORDER):
        g = S[S.pbin == b]
        cats = 100 * g.cat.value_counts(normalize=True).reindex(['one', 'in_over_out', 'out_over_in']).fillna(0).sort_values(ascending=False)
        cat, share = cats.index[0], cats.iloc[0]
        tie = (cats.iloc[0] - cats.iloc[1]) < 3
        x0 = n * GAP
        xc = x0 + W / 2
        ax.add_patch(Rectangle((x0, -H), W, H, fc='#EEF4FA', ec='0.4', lw=1.2))
        sel = g[g.cat == cat]
        if cat == 'one':
            layers = [(-H / 2, -100 * sel.Ubar.mean())]
            d = np.nan
        else:
            d = sel.d_int.median()
            ax.plot([x0, x0 + W], [-d, -d], 'k-', lw=2.5)
            ax.text(x0 + W + 0.02, -d, '%.0f m' % d, va='center', fontsize=8, color='0.3')
            layers = [(-d / 2, -100 * sel.U_up.mean()), (-(d + H) / 2, -100 * sel.U_lo.mean())]
        for y, v in layers: # v + = IN
            L = -v * ASC
            ax.add_patch(FancyArrowPatch((xc - L / 2, y), (xc + L / 2, y), arrowstyle='simple,head_width=9,head_length=9,tail_width=3.5',
                                         color=C_IN if v > 0 else C_OUT, zorder=3))
            ax.text(xc, y + 1.4, '%s %.0f' % ('IN' if v > 0 else 'OUT', abs(v)), ha='center', fontsize=8.5, color=C_IN if v > 0 else C_OUT, fontweight='bold')
        ax.text(xc, -H - 1.0, NAME[b], ha='center', va='top', fontsize=10, fontweight='bold')
        ax.text(xc, -H - 4.0, '%.0f%% of hours' % share + ('\n(tied with %s, %.0f%%)' % (TXT[cats.index[1]], cats.iloc[1]) if tie else ''),
                ha='center', va='top', fontsize=8, color='0.35')
        rows.append(dict(phase=NAME[b].replace('\n', ' '), cat=cat, share=share, divide=d, layers=layers, tie=tie))
    xs = np.linspace(-0.2, len(ORDER) * GAP, 300)
    xm = np.arange(len(ORDER)) * GAP + W / 2
    eta = lambda x: np.sin((x - W / 2) / GAP * np.pi / 4)
    a = 3.0 * amp / AMP[PAIR[0][0]] # same vertical scale for both rows
    ax.plot(xs, 7.0 + a * eta(xs), '-', color='0.3', lw=2)
    ax.plot(xm, 7.0 + a * eta(xm), 'o', color='0.3', ms=6)
    ax.text(-0.25, 7.0, 'sea\nlevel', ha='right', va='center', fontsize=9, color='0.3')
    ax.axhline(0, color='#56B4E9', lw=1, xmin=0.02, xmax=0.98)
    ax.set_xlim(-0.6, len(ORDER) * GAP + 0.3)
    ax.set_ylim(-H - 8.5, 11.5)
    ax.axis('off')
    ax.set_title(label, loc='left', fontsize=12, fontweight='bold')
    return rows

fig, axs = plt.subplots(2, 1, figsize=(17, 12.5), layout='constrained')
out = {}
for ax, b, letter in zip(axs, PAIR[0], 'ab'):
    S = A[A[b] == 1]
    nmb = b.replace('_strat', ' stratification').upper()
    lab = '%s  %s (%stide: %s; %d days)' % (letter, nmb if args.pair == 'strat' else b.upper() + ' tides', EXTRA.get(b, ''), BANDS[b], S.index.normalize().nunique())
    out[b] = draw_row(ax, S, lab, AMP[b])
fig.text(0.01, 0.0, wrapcap('Each box is the water column at the mouth (about %.0f m deep), showing the most common layering at that point in the tide on those days; arrows: blue = flowing IN, red = flowing OUT, '
         'mean speed in cm/s (same scale in both rows); black line = the divide. The tide is mixed (semidiurnal + diurnal parts, box-mean amplitudes in the titles); the sea-level sketch is one flood-ebb oscillation, scaled to each set\'s %s part.' % (H, PAIR[2]), fig), fontsize=9, color='0.25', va='top', ha='left')
fig.suptitle('Layering of the water column at the Penn Cove mouth through an average tidal cycle: %s vs %s (model %s, 2024-2025)' % (PAIR[0][0].replace('_', ' '), PAIR[0][1].replace('_', ' '), GTX), fontsize=13)
fn = out_dir / ('pc_%s_layers.png' % args.pair)
fig.savefig(fn, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved ' + str(fn))
for b in out:
    print(b)
    for r in out[b]:
        print('   %-18s %-12s %3.0f%%%s divide %s | %s' % (r['phase'], r['cat'], r['share'], ' (tie)' if r['tie'] else '', '%.1f m' % r['divide'] if np.isfinite(r['divide']) else '  -  ',
                                                    ', '.join('%s %.1f' % ('IN' if v > 0 else 'OUT', abs(v)) for _, v in r['layers'])))
