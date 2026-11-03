"""
Layering of the water column at the Penn Cove mouth through a TYPICAL TIDAL DAY
of the mixed tide: two high and two low waters (higher high HHW, lower low LLW,
lower high LHW, higher low HLW), instead of one generic oscillation.

Turning points: local maxima / minima of the hourly box-mean sea level (pc_cove
box), >= 6 h apart and >= 0.15 m prominent, forced to alternate. A high water is
HIGHER if it stands above the mean of its two neighbouring highs, otherwise
LOWER; low waters the same way. The most common order through the day sets the
typical sequence; only hours on segments that belong to it are used (share
printed). Each flood / ebb between turning points gives a PEAK stage (middle
third) and a LATE stage (last third); each turning point gives a SLACK stage
(within 1 h) -> 12 stages per tidal day.

For each stage the most common layer arrangement among its hours is drawn
(as in 20261028_pc_tidal_cycle_layers_schematic.py: total velocity, section-mean
profile on depth below the surface, median divide, mean layer speeds in cm/s),
with the share of hours it covers. The sea-level curve is the MEAN sea level of
the hours in each stage, so the diurnal inequality is the model's own.

-pair  all (one row, default) | springneap | tropeq | strat (as in
       20261028_pc_springneap_layers_schematic.py) | wind: HOURS grouped by the
       previous 6-h mean along-cove wind, out of the cove (<= -2 m/s) vs into the
       cove (>= +2 m/s), from 20260806_wind_reduce.py (u_pc, v_pc)

Input: DM_outs/20261028_pc_mouth_interface/hourly_interface.csv, the pc_cove box
(zeta), band-fit flags, DM_outs/20261027_pc_twolayer_tides/daily.csv (strat)
Output: LO_output/DM_outs/20261028_pc_tidalday_layers_schematic/pc_tidalday_layers_<pair>.png

  python 20261028_pc_tidalday_layers_schematic.py
  python 20261028_pc_tidalday_layers_schematic.py -pair springneap
  python 20261028_pc_tidalday_layers_schematic.py -pair wind
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline

from lo_tools import Lfun

import textwrap

def wrapcap(s, fig, cpi=13.5):
    """wrap a caption to the figure width (~cpi characters per inch at fontsize 8.5)"""
    return '\n'.join(textwrap.fill(p, int(fig.get_figwidth() * cpi)) for p in s.split('\n'))


p = argparse.ArgumentParser()
p.add_argument('-pair', default='all', choices=['all', 'springneap', 'tropeq', 'strat', 'wind'])
p.add_argument('-wthr', type=float, default=2.0, help='wind threshold [m/s] for -pair wind')
args = p.parse_args()
GTX, JOB, DS0, DS1 = 'wb1_t0_xn11abbur00', 'pc_cove', '2024.01.01', '2025.12.31'
Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261028_pc_tidalday_layers_schematic'
Lfun.make_dir(out_dir)
C_IN, C_OUT = '#0072B2', '#D55E00'
TXT = {'one': 'one layer', 'in_over_out': 'surface in / deep out', 'out_over_in': 'surface out / deep in'}
W, GAP, ASC = 0.78, 1.0, 0.045

R = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261028_pc_mouth_interface' / 'hourly_interface.csv', parse_dates=['time'])
A = R[R.part == 'all'].set_index('time').sort_index()
A['cat'] = np.select([A.NL == 1, (A.NL == 2) & (A.orient == 'in over out'), (A.NL == 2) & (A.orient == 'out over in'), A.NL == 3],
                     ['one', 'in_over_out', 'out_over_in', 'three'], 'none')
H = A.Hsec.mean()

# ---------------------------------------------------------------- turning points
box = xr.open_dataset(Ldir['LOo'] / 'extract' / GTX / 'box' / ('%s_%s_%s.nc' % (JOB, DS0, DS1)))
mask = box.mask_rho.values == 1
Z = pd.Series(np.nanmean(box.zeta.values[:, mask], axis=1), index=pd.DatetimeIndex(box.ocean_time.values))
box.close()
z = Z.values
ih, _ = find_peaks(z, distance=6, prominence=0.15)
il, _ = find_peaks(-z, distance=6, prominence=0.15)
E = sorted([(i, 'H') for i in ih] + [(i, 'L') for i in il])
alt = []
for i, k in E: # force alternation, keeping the more extreme of a repeated pair
    if alt and alt[-1][1] == k:
        if (k == 'H' and z[i] > z[alt[-1][0]]) or (k == 'L' and z[i] < z[alt[-1][0]]):
            alt[-1] = (i, k)
    else:
        alt.append((i, k))
EV = pd.DataFrame(alt, columns=['i', 'k'])
EV['z'] = z[EV.i]
EV['t'] = Z.index[EV.i]
lab = []
for n, r in EV.iterrows():
    same = EV[EV.k == r.k]
    j = same.index.get_loc(n)
    nb = [same.z.iloc[m] for m in (j - 1, j + 1) if 0 <= m < len(same)]
    hi = r.z > np.mean(nb)
    lab.append({('H', True): 'HHW', ('H', False): 'LHW', ('L', True): 'HLW', ('L', False): 'LLW'}[(r.k, hi)])
EV['lab'] = lab
# typical order: what follows HHW
nxt = EV.lab.shift(-1)
f_llw = (nxt[EV.lab == 'HHW'] == 'LLW').mean()
SEQ = ['HHW', 'LLW', 'LHW', 'HLW'] if f_llw >= 0.5 else ['HHW', 'HLW', 'LHW', 'LLW']
print('turning points: %d; after HHW comes LLW on %.0f%% of days -> typical order %s' % (len(EV), 100 * f_llw, ' -> '.join(SEQ)))
OKSEG = {(SEQ[m], SEQ[(m + 1) % 4]) for m in range(4)}

# ---------------------------------------------------------------- stage of every hour
stage = pd.Series(index=A.index, dtype=object)
tt = A.index.values
for n in range(len(EV) - 1):
    a, b = EV.iloc[n], EV.iloc[n + 1]
    seg = (a.lab, b.lab)
    if seg not in OKSEG:
        continue
    sel = (tt > a.t.to_datetime64()) & (tt < b.t.to_datetime64())
    f = (tt[sel] - a.t.to_datetime64()) / (b.t - a.t).to_timedelta64()
    kind = 'ebb' if a.k == 'H' else 'flood'
    s = np.where(f >= 2 / 3, 'late %s to %s' % (kind, b.lab), np.where(f >= 1 / 3, 'peak %s to %s' % (kind, b.lab), ''))
    stage.iloc[np.flatnonzero(sel)] = s
for _, r in EV.iterrows(): # slack within 1 h of each turning point
    near = np.abs(tt - r.t.to_datetime64()) <= np.timedelta64(1, 'h')
    stage.iloc[np.flatnonzero(near)] = '%s slack' % r.lab
A['stage12'] = stage.replace('', np.nan)
ORDER = []
for m in range(4):
    a, b = SEQ[m], SEQ[(m + 1) % 4]
    kind = 'ebb' if a in ('HHW', 'LHW') else 'flood'
    ORDER += ['%s slack' % a, 'peak %s to %s' % (kind, b), 'late %s to %s' % (kind, b)]
print('hours assigned to a stage of the typical tidal day: %.0f%%' % (100 * A.stage12.isin(ORDER).mean()))
A = A.join(Z.rename('zeta'), how='left')

# ---------------------------------------------------------------- groups
BF = xr.open_dataset(Ldir['LOo'] / 'extract' / GTX / 'tidal_ellipses' / ('%s_%s_%s_bands.nc' % (JOB, DS0, DS1)))
bday = pd.DatetimeIndex(BF.time.values).normalize()
zD = {k: pd.Series(BF['zidx_' + k].values, index=bday) for k in ['D2', 'D1']}
if args.pair == 'all':
    GROUPS = {'all days': bday}
elif args.pair == 'strat':
    DS = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261027_pc_twolayer_tides' / 'daily.csv', index_col=0, parse_dates=True)
    q1, q2 = DS.strat.quantile([1 / 3, 2 / 3])
    GROUPS = {'high stratification (%.1f g/kg)' % DS.strat[DS.strat >= q2].mean(): DS.index[DS.strat >= q2],
              'low stratification (%.1f g/kg)' % DS.strat[DS.strat <= q1].mean(): DS.index[DS.strat <= q1]}
elif args.pair == 'wind':
    Wd = pd.read_pickle(Ldir['LOo'] / 'DM_outs' / '20260806_wind' / 'wind_hourly_atm00_2024.01.01_2025.12.31.p')['W']
    wa = (Wd.u_pc * -0.9789 + Wd.v_pc * -0.2043).rolling(6, min_periods=4).mean().reindex(A.index) # + = into the cove
    mo, mi = (wa <= -args.wthr).values, (wa >= args.wthr).values
    GROUPS = {'wind OUT of the cove (6-h mean <= -%g m/s, mean %.1f m/s; %d h)' % (args.wthr, -wa[mo].mean(), mo.sum()): mo,
              'wind INTO the cove (6-h mean >= +%g m/s, mean %.1f m/s; %d h)' % (args.wthr, wa[mi].mean(), mi.sum()): mi}
else:
    pr = {'springneap': ['spring', 'neap'], 'tropeq': ['tropic', 'equatorial']}[args.pair]
    GROUPS = {b + ' tides': bday[BF['is_' + b].values == 1] for b in pr}

def draw_row(ax, S, title):
    xs, zs = [], []
    for n, stg in enumerate(ORDER):
        g = S[S.stage12 == stg]
        cats = 100 * g.cat.value_counts(normalize=True).reindex(['one', 'in_over_out', 'out_over_in']).fillna(0).sort_values(ascending=False)
        cat, share = cats.index[0], cats.iloc[0]
        tie = (cats.iloc[0] - cats.iloc[1]) < 3
        x0 = n * GAP
        xc = x0 + W / 2
        ax.add_patch(Rectangle((x0, -H), W, H, fc='#EEF4FA' if 'slack' not in stg else '#E2ECF5', ec='0.4', lw=1.2))
        sel = g[g.cat == cat]
        if cat == 'one':
            layers = [(-H / 2, -100 * sel.Ubar.mean())]
        else:
            d = sel.d_int.median()
            ax.plot([x0, x0 + W], [-d, -d], 'k-', lw=2.5)
            ax.text(x0 + W + 0.02, -d, '%.0f' % d, va='center', fontsize=7.5, color='0.3')
            layers = [(-d / 2, -100 * sel.U_up.mean()), (-(d + H) / 2, -100 * sel.U_lo.mean())]
        for y, v in layers:
            L = -v * ASC
            ax.add_patch(FancyArrowPatch((xc - L / 2, y), (xc + L / 2, y), arrowstyle='simple,head_width=8,head_length=8,tail_width=3',
                                         color=C_IN if v > 0 else C_OUT, zorder=3))
            ax.text(xc, y + 1.4, '%s %.0f' % ('IN' if v > 0 else 'OUT', abs(v)), ha='center', fontsize=8, color=C_IN if v > 0 else C_OUT, fontweight='bold')
        nm = stg.replace(' to ', '\nto ').replace(' slack', '\nslack')
        ax.text(xc, -H - 0.8, nm, ha='center', va='top', fontsize=8.5, fontweight='bold')
        ax.text(xc, -H - 4.6, '%.0f%%' % share + ('\n(tie: %s %.0f%%)' % (TXT[cats.index[1]], cats.iloc[1]) if tie else ''),
                ha='center', va='top', fontsize=7.5, color='0.35')
        xs.append(xc)
        zs.append(g.zeta.mean())
    # sea level: mean of each stage's hours, closed periodic spline
    xs, zs = np.array(xs), np.array(zs)
    span = len(ORDER) * GAP
    cs = CubicSpline(np.r_[xs, xs[0] + span], np.r_[zs, zs[0]], bc_type='periodic')
    xx = np.linspace(xs[0] - 0.4, xs[-1] + 0.4, 400)
    sc, z0 = 1.6, 7.5
    ax.plot(xx, z0 + sc * cs(xx), '-', color='0.3', lw=2)
    ax.plot(xs, z0 + sc * zs, 'o', color='0.3', ms=5)
    for x_, z_, stg in zip(xs, zs, ORDER):
        if 'slack' in stg:
            ax.text(x_, z0 + sc * z_ + (0.8 if z_ > 0 else -0.8), stg.split()[0], ha='center', va='bottom' if z_ > 0 else 'top', fontsize=8.5, fontweight='bold', color='0.25')
    ax.text(-0.3, z0, 'sea\nlevel', ha='right', va='center', fontsize=9, color='0.3')
    ax.axhline(0, color='#56B4E9', lw=1, xmin=0.02, xmax=0.98)
    ax.set_xlim(-0.7, span + 0.3)
    ax.set_ylim(-H - 9, z0 + 4.5)
    ax.axis('off')
    ax.set_title(title, loc='left', fontsize=11.5, fontweight='bold')

nrow = len(GROUPS)
fig, axs = plt.subplots(nrow, 1, figsize=(19, 6.6 * nrow + 0.6), layout='constrained', squeeze=False)
for ax, (gname, days), letter in zip(axs[:, 0], GROUPS.items(), 'abcd'):
    if args.pair == 'wind':
        S = A[days]
        days = S.index.normalize().unique()
    else:
        S = A[A.index.normalize().isin(days)]
    gl = gname.split(' (')
    gt = 'All days' if args.pair == 'all' else gl[0].upper() + (' (' + gl[1] if len(gl) > 1 else '')
    if args.pair == 'wind':
        title = '%s  %s' % (letter, gl[0].replace('wind OUT', 'WIND OUT').replace('wind INTO', 'WIND INTO') + ' (' + gl[1])
    else:
        title = '%s  %s: semidiurnal %.2f m + diurnal %.2f m; %d days' % (letter, gt, zD['D2'].reindex(days).mean(),
                                                                       zD['D1'].reindex(days).mean(), S.index.normalize().nunique())
    draw_row(ax, S, title)
fig.text(0.01, 0.0, wrapcap('Each box is the water column at the mouth (about %.0f m deep): the most common layering at that stage (percent of hours below; shaded boxes = slack within 1 h of a turning point). '
         'Arrows: blue = IN, red = OUT, mean speed in cm/s; black line = the divide (m below the surface).\nTidal day of the mixed tide in its most common order (%s); sea level = the mean of each stage\'s hours. '
         'Peak = middle third, late = last third of each flood or ebb.' % (H, ' -> '.join(SEQ)), fig), fontsize=8.5, color='0.25', va='top', ha='left')
fig.suptitle('Layering of the water column at the Penn Cove mouth through a typical tidal day (two highs, two lows), model %s 2024-2025' % GTX, fontsize=13)
fn = out_dir / ('pc_tidalday_layers_%s.png' % args.pair)
fig.savefig(fn, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved ' + str(fn))
