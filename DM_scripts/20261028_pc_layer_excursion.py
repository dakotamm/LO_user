"""
How far the DEEP and the SURFACE water at the Penn Cove mouth are carried during
each flood and ebb of the typical tidal day, for each grouping used in the
tidal-day schematics (all, spring/neap, tropic/equatorial, high/low
stratification, wind out of / into the cove).

Layers follow the two-layer picture, not a fixed bottom slab: the LOWER layer
is everything deeper than the typical divide (10.6 m below the moving surface,
20261028_pc_mouth_interface.py), the UPPER layer everything above it. Hourly
total velocity (+ = OUT) in the mouth section of the pc_cove box is interpolated
to depth below the surface, width-averaged, then thickness-averaged over each
layer. The displacement of a layer over one flood or ebb is the time integral of
its section-mean velocity between turning points (km, + = out): how far that
water would go if it moved with the local mean flow at the mouth. It includes
the residual, so the four segments sum to the net drift over a tidal day.

Turning points and the typical order (HHW -> HLW -> LHW -> LLW) as in
20261028_pc_tidalday_layers_schematic.py. Groups: segment start day in the
band-fit / stratification terciles; wind = mean along-cove wind over the segment
(<= -2 out of, >= +2 m/s into the cove).

Output: LO_output/DM_outs/20261028_pc_layer_excursion/pc_layer_excursion.png,
        segments.csv, report.txt

  python 20261028_pc_layer_excursion.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.signal import find_peaks

from lo_tools import Lfun

import textwrap

def wrapcap(s, fig, cpi=13.5):
    """wrap a caption to the figure width (~cpi characters per inch at fontsize 8.5)"""
    return '\n'.join(textwrap.fill(p, int(fig.get_figwidth() * cpi)) for p in s.split('\n'))


GTX, JOB, DS0, DS1 = 'wb1_t0_xn11abbur00', 'pc_cove', '2024.01.01', '2025.12.31'
Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261028_pc_layer_excursion'
Lfun.make_dir(out_dir)
C_IN, C_OUT = '#0072B2', '#D55E00'
OUT = np.array([0.9789, 0.2043])
DIV = float(pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261028_pc_mouth_interface' / 'hourly_interface.csv',
                        usecols=['part', 'NL', 'd_int']).query("part == 'all' and NL == 2").d_int.median())
DG = np.arange(0.25, 26.0, 0.5)

# ---------------------------------------------------------- hourly layer velocities
ds = xr.open_dataset(Ldir['LOo'] / 'extract' / GTX / 'box' / ('%s_%s_%s.nc' % (JOB, DS0, DS1)))
tt = pd.DatetimeIndex(ds.ocean_time.values)
lon = ds.lon_rho.values
mask = ds.mask_rho.values == 1
LON_LP = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc_lp.p').x.mean()
i = int(np.flatnonzero(lon[0] < LON_LP)[-1])
jw = np.array([j for j in np.flatnonzero(mask[:, i]) if 0 < j < mask.shape[0] - 1])
u = np.nan_to_num(ds.u[:, :, :, i - 1:i + 1].values).mean(axis=-1)[:, :, jw]
v = np.nan_to_num(ds.v[:, :, :, i].values)
v = 0.5 * (v[:, :, jw - 1] + v[:, :, jw])
zr = ds.z_rho[:, :, :, i].values[:, :, jw]
zw = ds.z_w[:, :, :, i].values[:, :, jw]
Zb = pd.Series(np.nanmean(ds.zeta.values[:, mask], axis=1), index=tt)
ds.close()
uo = u * OUT[0] + v * OUT[1]
NT, NZ, NJ = uo.shape
dep = zw[:, -1:, :] - zr
UG = np.full((NT, len(DG), NJ), np.nan)
for j in range(NJ):
    hcol = zw[:, -1, j] - zw[:, 0, j]
    for t in range(NT):
        g = DG <= hcol[t] - 0.25
        UG[t, g, j] = np.interp(DG[g], dep[t, ::-1, j], uo[t, ::-1, j])
P = np.nanmean(UG, axis=2) # (T, depth) width-mean profile
lo_, up_ = DG > DIV, DG <= DIV
Ulo = np.nanmean(np.where(lo_[None], P, np.nan), axis=1)
Uup = np.nanmean(np.where(up_[None], P, np.nan), axis=1)
dz = np.diff(zw, axis=1)
Ubar = (uo * dz).sum((1, 2)) / dz.sum((1, 2))
HV = pd.DataFrame({'lo': Ulo, 'up': Uup, 'slab': Ubar}, index=tt)
print('layer velocities done; divide %.1f m' % DIV)

# ---------------------------------------------------------- turning points (as the schematic)
z = Zb.values
ih, _ = find_peaks(z, distance=6, prominence=0.15)
il, _ = find_peaks(-z, distance=6, prominence=0.15)
alt = []
for k_, kind in sorted([(n, 'H') for n in ih] + [(n, 'L') for n in il]):
    if alt and alt[-1][1] == kind:
        if (kind == 'H' and z[k_] > z[alt[-1][0]]) or (kind == 'L' and z[k_] < z[alt[-1][0]]):
            alt[-1] = (k_, kind)
    else:
        alt.append((k_, kind))
EV = pd.DataFrame(alt, columns=['i', 'k'])
EV['z'] = z[EV.i]
EV['t'] = tt[EV.i]
lab = []
for n, r in EV.iterrows():
    same = EV[EV.k == r.k]
    j = same.index.get_loc(n)
    nb = [same.z.iloc[m] for m in (j - 1, j + 1) if 0 <= m < len(same)]
    lab.append({('H', True): 'HHW', ('H', False): 'LHW', ('L', True): 'HLW', ('L', False): 'LLW'}[(r.k, r.z > np.mean(nb))])
EV['lab'] = lab
SEQ = ['HHW', 'HLW', 'LHW', 'LLW']
SEGS = [('HHW', 'HLW', 'small ebb\nHHW -> HLW'), ('HLW', 'LHW', 'small flood\nHLW -> LHW'), ('LHW', 'LLW', 'big ebb\nLHW -> LLW'), ('LLW', 'HHW', 'big flood\nLLW -> HHW')]
SEGN = {(a, b): nm for a, b, nm in SEGS}

Wd = pd.read_pickle(Ldir['LOo'] / 'DM_outs' / '20260806_wind' / 'wind_hourly_atm00_2024.01.01_2025.12.31.p')['W']
wa = (Wd.u_pc * -0.9789 + Wd.v_pc * -0.2043).reindex(tt) # + = into the cove
rows = []
for n in range(len(EV) - 1):
    a, b = EV.iloc[n], EV.iloc[n + 1]
    if (a.lab, b.lab) not in SEGN:
        continue
    seg = HV.loc[a.t:b.t].iloc[:-1] # hours from a up to (not including) b
    if len(seg) < 3 or seg.isna().any().any():
        continue
    rows.append(dict(t0=a.t, seg=SEGN[(a.lab, b.lab)], hours=len(seg), X_lo=seg.lo.sum() * 3.6, X_up=seg.up.sum() * 3.6, X_slab=seg.slab.sum() * 3.6,
                     wind=wa.loc[a.t:b.t].mean()))
S = pd.DataFrame(rows)
S['day'] = S.t0.dt.normalize()
BF = xr.open_dataset(Ldir['LOo'] / 'extract' / GTX / 'tidal_ellipses' / ('%s_%s_%s_bands.nc' % (JOB, DS0, DS1)))
F = pd.DataFrame({nm: BF['is_' + nm].values for nm in ['spring', 'neap', 'tropic', 'equatorial']}, index=pd.DatetimeIndex(BF.time.values).normalize())
S = S.join(F, on='day')
DS = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261027_pc_twolayer_tides' / 'daily.csv', index_col=0, parse_dates=True)
q1, q2 = DS.strat.quantile([1 / 3, 2 / 3])
S['high_strat'] = S.day.map(DS.strat >= q2).fillna(False).astype(int)
S['low_strat'] = S.day.map(DS.strat <= q1).fillna(False).astype(int)
S['wind_out'] = (S.wind <= -2).astype(int)
S['wind_in'] = (S.wind >= 2).astype(int)
S.to_csv(out_dir / 'segments.csv', index=False, float_format='%.4f')

GROUPS = [('all days', None), ('spring', 'spring'), ('neap', 'neap'), ('tropic', 'tropic'), ('equatorial', 'equatorial'),
          ('high stratification', 'high_strat'), ('low stratification', 'low_strat'), ('wind out of the cove', 'wind_out'), ('wind into the cove', 'wind_in')]
segnames = [s[2] for s in SEGS]
TAB = {}
rep = ['displacement over each segment, km (+ = OUT); lower layer = deeper than %.1f m; mean over segments' % DIV]
for gname, col in GROUPS:
    g = S if col is None else S[S[col] == 1]
    T = g.groupby('seg')[['X_lo', 'X_up', 'X_slab', 'hours']].mean().reindex(segnames)
    nn = g.groupby('seg').size().reindex(segnames)
    TAB[gname] = (T, nn)
    rep.append('%-22s ' % gname + ' | '.join('%s: deep %+.2f, surf %+.2f (n %d)' % (s.split('\n')[0], T.loc[s, 'X_lo'], T.loc[s, 'X_up'], nn[s]) for s in segnames)
               + ' | NET deep %+.2f, surf %+.2f km/tidal day' % (T.X_lo.sum(), T.X_up.sum()))

# ---------------------------------------------------------- figure
NG = len(GROUPS)
fig, ax = plt.subplots(figsize=(17, 1.05 * NG + 2.2), layout='constrained')
SCX = 0.21 # axis units per km
colx = np.arange(5) * 1.0
for r_, (gname, col) in enumerate(GROUPS):
    y0 = NG - 1 - r_
    T, nn = TAB[gname]
    ax.text(-0.62, y0, gname, ha='right', va='center', fontsize=10.5, fontweight='bold')
    for c, sname in enumerate(segnames + ['NET over the tidal day']):
        if c < 4:
            xlo, xup = T.loc[sname, 'X_lo'], T.loc[sname, 'X_up']
        else:
            xlo, xup = T.X_lo.sum(), T.X_up.sum()
        for dy, X, alpha, lw, lab in [(0.17, xup, 0.45, 2.0, 'surface'), (-0.17, xlo, 1.0, 3.5, 'deep')]:
            L = X * SCX
            colr = C_OUT if X > 0 else C_IN
            if abs(L) > 0.01:
                ax.add_patch(FancyArrowPatch((colx[c], y0 + dy), (colx[c] + L, y0 + dy), arrowstyle='-|>', mutation_scale=12, color=colr, alpha=alpha, lw=lw, shrinkA=0, shrinkB=0))
            ax.text(colx[c] + 0.47, y0 + dy, '%+.1f' % X, ha='left', va='center', fontsize=8.5, color=colr, alpha=max(alpha, 0.7),
                    fontweight='bold' if lab == 'deep' else 'normal')
    ax.axhline(y0 - 0.5, color='0.9', lw=0.8)
for c, sname in enumerate(segnames + ['NET over the\ntidal day']):
    ax.text(colx[c], NG - 0.35, sname, ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.plot([colx[c], colx[c]], [-0.5, NG - 0.4], color='0.85', lw=0.8, zorder=0)
ax.set_xlim(-2.4, 4.75)
ax.set_ylim(-0.7, NG + 0.35)
ax.axis('off')
fig.text(0.01, 0.0, wrapcap('Arrows start at each column\'s grey line: RIGHT/red = carried OUT of the cove, LEFT/blue = carried IN, km per flood or ebb (number at right). '
        'Bold = DEEP water (below the typical divide, %.1f m); faint = SURFACE water (above it).\nSection-mean velocity at the mouth integrated over each flood/ebb of the typical tidal day (HHW -> HLW -> LHW -> LLW); includes the residual, so the four '
        'segments add up to the net drift (last column).' % DIV, fig, cpi=8.5), fontsize=8.5, color='0.25', va='top', ha='left')
fig.suptitle('How far deep and surface water at the Penn Cove mouth are carried during each flood and ebb (model %s, 2024-2025)' % GTX, fontsize=13)
fig.savefig(out_dir / 'pc_layer_excursion.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
(out_dir / 'report.txt').write_text('\n'.join(rep) + '\n')
print('\n'.join(rep))
