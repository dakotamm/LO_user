"""
When is the flow at the Penn Cove mouth ONE layer (all in or all out), TWO
layers (in over out, or out over in) or THREE+ layers? Does any tidal
combination switch the usual layered flow to one layer?

Hourly, pc_cove box, mouth section (last rho column west of pc_lp), TOTAL
velocity (tide + residual + wind), along-cove component (+ = OUT):
  U_k(t)   section-mean velocity on each sigma level (dz-weighted across the
           section) -> one profile per hour
  layers   contiguous runs of one sign, counting only levels with |U| >= 1 cm/s
           and runs >= 2 m thick; adjacent runs of the same sign merge.
           0 = everything weaker than 1 cm/s
  Ubar     depth- and width-mean velocity (filling/draining, the "slab")
  stage    Ubar / that UTC day's max |Ubar|: strong flood < -0.5 <= weak flood
           < -0.15 <= slack <= 0.15 < weak ebb <= 0.5 < strong ebb
Grouped by stage, by the 2 x 2 spring/neap x tropic/equatorial combinations
(band-fit tercile flags, 20261027_tidal_ellipses_windowed.py), by month, and
against |Ubar| relative to that day's two-layer strength.

Output: LO_output/DM_outs/20261027_pc_mouth_layers/pc_mouth_layers.png, report.txt,
        hourly_layers.csv

  python 20261027_pc_mouth_layers.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

GTX, JOB, DS0, DS1 = 'wb1_t0_xn11abbur00', 'pc_cove', '2024.01.01', '2025.12.31'
Ldir = Lfun.Lstart(gridname='wb1')
box_fn = Ldir['LOo'] / 'extract' / GTX / 'box' / ('%s_%s_%s.nc' % (JOB, DS0, DS1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261027_pc_mouth_layers'
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
OUT = np.array([0.9789, 0.2043])
UMIN, HMIN = 0.01, 2.0 # m/s, m
CL = {1: '#0072B2', 2: '#E69F00', 3: '#D55E00', 0: '0.75'}
LAB = {1: 'one layer', 2: 'two layers', 3: 'three or more', 0: 'all < 1 cm/s'}

ds = xr.open_dataset(box_fn)
tt = pd.DatetimeIndex(ds.ocean_time.values)
lon = ds.lon_rho.values
mask = ds.mask_rho.values == 1
LON_LP = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc_lp.p').x.mean()
i = int(np.flatnonzero(lon[0] < LON_LP)[-1])
jw = np.array([j for j in np.flatnonzero(mask[:, i]) if 0 < j < mask.shape[0] - 1])
u = np.nan_to_num(ds.u[:, :, :, i - 1:i + 1].values).mean(axis=-1)[:, :, jw]
v = np.nan_to_num(ds.v[:, :, :, i].values)
v = 0.5 * (v[:, :, jw - 1] + v[:, :, jw])
zw = ds.z_w[:, :, :, i].values[:, :, jw]
ds.close()
uo = u * OUT[0] + v * OUT[1]
dz = np.diff(zw, axis=1) # (T, NZ, nj)
Uk = (uo * dz).sum(2) / dz.sum(2) # (T, NZ) section-mean profile on sigma levels
Hk = dz.mean(2) # mean level thickness
Ubar = (uo * dz).sum((1, 2)) / dz.sum((1, 2))

def n_layers(U, H):
    sg = np.where(np.abs(U) >= UMIN, np.sign(U), 0)
    runs = []
    for s_, h in zip(sg, H):
        if s_ == 0:
            continue
        if runs and runs[-1][0] == s_:
            runs[-1][1] += h
        else:
            runs.append([s_, h])
    runs = [r for r in runs if r[1] >= HMIN]
    merged = []
    for r in runs:
        if merged and merged[-1][0] == r[0]:
            merged[-1][1] += r[1]
        else:
            merged.append(list(r))
    return min(len(merged), 3)

NL = np.array([n_layers(Uk[t], Hk[t]) for t in range(len(tt))])
H = pd.DataFrame({'NL': NL, 'Ubar': Ubar}, index=tt)
dmax = H.Ubar.abs().groupby(H.index.normalize()).transform('max')
x = H.Ubar / dmax
H['stage'] = pd.cut(x, [-1.01, -0.5, -0.15, 0.15, 0.5, 1.01], labels=['strong flood', 'weak flood', 'slack', 'weak ebb', 'strong ebb'])
BC = np.sqrt(((Uk - Ubar[:, None]) ** 2 * Hk).sum(1) / Hk.sum(1)) # two-layer strength this hour
H['BCd'] = pd.Series(BC, index=tt).groupby(tt.normalize()).transform(lambda s: np.sqrt((s ** 2).mean()))
H['ratio'] = H.Ubar.abs() / H.BCd

BF = xr.open_dataset(Ldir['LOo'] / 'extract' / GTX / 'tidal_ellipses' / ('%s_%s_%s_bands.nc' % (JOB, DS0, DS1)))
F = pd.DataFrame({nm: BF['is_' + nm].values for nm in ['spring', 'neap', 'tropic', 'equatorial']}, index=pd.DatetimeIndex(BF.time.values).normalize())
H = H.join(F, on=H.index.normalize())
combo = np.full(len(H), '', dtype=object)
for a in ['spring', 'neap']:
    for b in ['tropic', 'equatorial']:
        combo[(H[a] == 1).values & (H[b] == 1).values] = '%s +\n%s' % (a, b)
H['combo'] = combo

def frac(g):
    return pd.Series({k: (g.NL == k).mean() for k in [1, 2, 3, 0]})

rep = ['mouth section-mean profiles, total velocity, %d hours; layer = |U| >= %g cm/s and >= %g m thick' % (len(H), 100 * UMIN, HMIN)]
rep.append('all hours: ' + ', '.join('%s %.0f%%' % (LAB[k], 100 * (H.NL == k).mean()) for k in [1, 2, 3, 0]))
FS = H.groupby('stage', observed=True).apply(frac)
FC = H[H.combo != ''].groupby('combo').apply(frac)
FM = H.groupby(H.index.month).apply(frac)
for nm, Fx in [('by stage', FS), ('by combo', FC), ('by month', FM)]:
    rep.append(nm + ':')
    for idx, r_ in Fx.iterrows():
        rep.append('   %-24s one %.0f%%, two %.0f%%, three+ %.0f%%' % (str(idx).replace('\n', ' '), 100 * r_[1], 100 * r_[2], 100 * r_[3]))
cmb = H[H.combo != '']
for c in sorted(cmb.combo.unique()):
    g = cmb[(cmb.combo == c) & (cmb.stage.isin(['strong flood', 'strong ebb']))]
    rep.append('strong flood/ebb hours, %-24s one layer %.0f%% (n %d)' % (c.replace('\n', ' '), 100 * (g.NL == 1).mean(), len(g)))
rb = pd.cut(H.ratio, [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 10])
PR = H.groupby(rb, observed=True).apply(frac)

fig, axd = plt.subplot_mosaic([['a', 'b'], ['c', 'd']], layout='constrained', figsize=(14, 10))
def stacked(ax, Fx, labels, title, rot=0):
    bottom = np.zeros(len(Fx))
    xx = np.arange(len(Fx))
    for k in [1, 2, 3, 0]:
        ax.bar(xx, 100 * Fx[k].values, 0.7, bottom=bottom, color=CL[k], label=LAB[k], edgecolor='white', linewidth=1)
        bottom += 100 * Fx[k].values
    for xi, val in zip(xx, Fx[1].values):
        ax.text(xi, 100 * val / 2, '%.0f%%' % (100 * val), ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    ax.set_xticks(xx)
    ax.set_xticklabels(labels, fontsize=8, rotation=rot)
    ax.set_ylim(0, 100)
    ax.set_ylabel('% of hours')
    ax.set_title(title, fontsize=10, loc='left')
    ax.grid(axis='y', **GRID)
stacked(axd['a'], FS, [str(s) for s in FS.index], 'by tidal stage (Ubar relative to that day\'s peak)')
axd['a'].legend(fontsize=8, loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.08))
stacked(axd['b'], FC, list(FC.index), 'by tidal combination (all stages)')
stacked(axd['c'], FM, ['JFMAMJJASOND'[m - 1] for m in FM.index], 'by month (2024-2025)')
stacked(axd['d'], PR, ['%.2g-%.2g' % (iv.left, iv.right) for iv in PR.index], 'by |slab flow| / that day\'s two-layer strength')
axd['d'].set_xlabel('|Ubar| / daily rms of the departure from the depth mean')
for k in 'abcd':
    axd[k].text(0.01, 0.98, k, transform=axd[k].transAxes, fontsize=13, fontweight='bold', va='top')
fig.suptitle('How many layers does the flow at the Penn Cove mouth have? (hourly section-mean profiles, total velocity, %s 2024-2025)' % GTX, fontsize=11)
fig.savefig(out_dir / 'pc_mouth_layers.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
H[['NL', 'Ubar', 'stage', 'BCd', 'ratio']].to_csv(out_dir / 'hourly_layers.csv', float_format='%.4f')
(out_dir / 'report.txt').write_text('\n'.join(rep) + '\n')
print('\n'.join(rep))
