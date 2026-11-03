"""
Conceptual diagram: which way BOTTOM water goes at the Penn Cove mouth.
Numbers from 20261028_pc_bottom_direction.py (bottom 3 m / top 3 m of the mouth
section, hourly total velocity, 2024-2025); background arrows in panel a are
the 10 m-to-bed residual from the cached Godin band velocities
(20261027_pc_lobe_cavity.py).

  a  plan view: bottom water comes IN along the north side, goes OUT along the
     south side (2-year mean); the net is a small inflow
  b  the mouth section looking into the cove: the same side-by-side exchange
  c  through a tide (peak flood -> slack -> peak ebb): top, slab and bottom
     velocity; the bottom moves against the surface most of the time and
     comes in around slack, leaves on strong ebbs
  d  by season: share of days the (tidally filtered) bottom flow is inward

Output: LO_output/DM_outs/20261028_pc_bottom_water_schematic/pc_bottom_water_schematic.png

  python 20261028_pc_bottom_water_schematic.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as MplPath

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

GTX, JOB, DS0, DS1 = 'wb1_t0_xn11abbur00', 'pc_cove', '2024.01.01', '2025.12.31'
Ldir = Lfun.Lstart(gridname='wb1')
src = Ldir['LOo'] / 'DM_outs' / '20261028_pc_bottom_direction'
M = pd.read_csv(src / 'mean.csv', index_col=0).iloc[:, 0]
ST = pd.read_csv(src / 'stage.csv', index_col=0)
SE = pd.read_csv(src / 'season.csv')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261028_pc_bottom_water_schematic'
Lfun.make_dir(out_dir)
sect_dir = Ldir['LOo'] / 'section_lines'
C_IN, C_OUT, C_SLAB = '#0072B2', '#D55E00', '0.45'

box_dir = Ldir['LOo'] / 'extract' / GTX / 'box'
ds = xr.open_dataset(box_dir / ('%s_%s_%s.nc' % (JOB, DS0, DS1)))
lon, lat, h = ds.lon_rho.values, ds.lat_rho.values, ds.h.values
mask = ds.mask_rho.values == 1
Hm = h + ds.zeta.mean('ocean_time').values
ds.close()
Z = np.load(box_dir / ('%s_godin_bands_%s_%s.npz' % (JOB, DS0, DS1)))
NR, NC = mask.shape
MB = mask & (Hm >= 12.0)
U = np.nanmean(Z['U_z10-bed'], axis=0) * (MB[:, :-1] & MB[:, 1:])
V = np.nanmean(Z['V_z10-bed'], axis=0) * (MB[:-1, :] & MB[1:, :])
ur = np.full((NR, NC), np.nan); vr = np.full((NR, NC), np.nan)
ur[:, 1:-1] = 0.5 * (U[:, :-1] + U[:, 1:]); vr[1:-1, :] = 0.5 * (V[:-1, :] + V[1:, :])
wb = pd.read_pickle(sect_dir / 'wb.p')
pth = MplPath(np.column_stack([wb.x.values, wb.y.values]))
show = mask & pth.contains_points(np.column_stack([lon.ravel(), lat.ravel()])).reshape(lon.shape)
plon, plat = pfun.get_plon_plat(lon, lat)
aa = [plon.min(), plon.max(), plat.min(), plat.max()]
LON_LP = pd.read_pickle(sect_dir / 'pc_lp.p').x.mean()
im = int(np.flatnonzero(lon[0] < LON_LP)[-1])

fig, axd = plt.subplot_mosaic([['a', 'a', 'b'], ['c', 'c', 'd']], layout='constrained', figsize=(16, 10.5),
                              gridspec_kw={'height_ratios': [1, 0.95]})

# ---------------------------------------------------------------- a: plan view
ax = axd['a']
ax.pcolormesh(plon, plat, np.where(show, 1.0, np.nan), cmap='Greys', vmin=0, vmax=12, shading='flat', zorder=0)
pfun.add_coast(ax, color='0.4', linewidth=0.8)
jj, ii = np.meshgrid(np.arange(0, NR, 2), np.arange(0, NC, 2), indexing='ij')
s = show[jj, ii] & np.isfinite(ur[jj, ii]) & MB[jj, ii]
ax.quiver(lon[jj, ii][s], lat[jj, ii][s], ur[jj, ii][s], vr[jj, ii][s], color='0.6', scale=0.03 / 0.3, scale_units='inches', width=0.003, zorder=2)
ax.add_patch(FancyArrowPatch((-122.652, 48.2425), (-122.706, 48.2325), connectionstyle='arc3,rad=0.12', arrowstyle='simple,head_width=14,head_length=14,tail_width=6',
                             color=C_IN, alpha=0.85, zorder=4))
ax.add_patch(FancyArrowPatch((-122.700, 48.2245), (-122.652, 48.2290), connectionstyle='arc3,rad=0.12', arrowstyle='simple,head_width=12,head_length=12,tail_width=4.5',
                             color=C_OUT, alpha=0.85, zorder=4))
ax.text(-122.672, 48.2405, 'IN along the north side\n%.1f cm/s, inflowing %.0f%% of hours' % (-M.bottom_north, M.bottom_north_in_pct),
        color=C_IN, fontsize=10, fontweight='bold', ha='center', va='bottom', bbox=dict(fc='white', ec='none', alpha=0.8), zorder=5)
ax.text(-122.676, 48.2232, 'OUT along the south side\n%.1f cm/s, outflowing %.0f%% of hours' % (M.bottom_south, 100 - M.bottom_south_in_pct),
        color=C_OUT, fontsize=10, fontweight='bold', ha='center', va='top', bbox=dict(fc='white', ec='none', alpha=0.8), zorder=5)
ax.axvline(lon[0, im], color='k', ls=':', lw=1)
ax.text(lon[0, im], aa[3], 'mouth ', ha='right', va='top', fontsize=9)
ax.axis(aa)
pfun.dar(ax)
ax.set_xticks([]); ax.set_yticks([])
ax.text(0.01, 0.03, 'net through the mouth: bottom %.1f cm/s (slightly IN), surface %+.1f cm/s (slightly OUT)\nthe side-by-side exchange is 5-7x the net; grey = mean flow 10 m to the bed'
        % (M.bottom, M.top), transform=ax.transAxes, fontsize=9, va='bottom', bbox=dict(fc='white', ec='0.7', alpha=0.9))
ax.set_title('a  On average, bottom water circulates through the mouth sideways', loc='left', fontsize=11, fontweight='bold')

# ---------------------------------------------------------------- b: section
ax = axd['b']
jw = np.flatnonzero(show[:, im])
yy, hh = lat[jw, im], h[jw, im]
ax.fill_between(yy, -hh, 0, color='#DCEBF5', zorder=0)
ax.fill_between(yy, -hh, -hh + 3, color='#B9D3E8', zorder=1)
ax.plot(yy, -hh, color='0.3', lw=1.5)
ax.fill_between(yy, -hh, -30, color='0.85', zorder=0)
ymid = np.median(yy)
north, south = yy > ymid, yy <= ymid
def mark(x, y, inward, col, size=420):
    ax.scatter([x], [y], s=size, facecolor='white', edgecolor=col, lw=2.5, zorder=4)
    if inward:
        ax.scatter([x], [y], s=size * 0.45, marker='x', color=col, lw=2.5, zorder=5)
    else:
        ax.scatter([x], [y], s=size * 0.08, color=col, zorder=5)
xn, xs = np.mean(yy[north]), np.mean(yy[south])
mark(xn, -np.interp(xn, yy, hh) + 2.2, True, C_IN)
mark(xs, -np.interp(xs, yy, hh) + 2.2, False, C_OUT)
ax.text(xn, -np.interp(xn, yy, hh) + 5.0, 'IN', color=C_IN, ha='center', fontsize=11, fontweight='bold')
ax.text(xs, -np.interp(xs, yy, hh) + 5.0, 'OUT', color=C_OUT, ha='center', fontsize=11, fontweight='bold')
ax.text(ymid, -1.2, 'top 3 m: net slightly OUT', ha='center', va='top', fontsize=9, color=C_OUT)
ax.text(yy[np.argmax(hh)] + 0.002, -np.max(hh) + 3.3, 'bottom 3 m (shaded)', ha='left', va='bottom', fontsize=8, color='0.3')
ax.set_xlim(yy.min(), yy.max())
ax.set_ylim(-28, 1)
ax.set_xlabel('south  <-  across the mouth  ->  north')
ax.set_ylabel('depth [m]')
ax.set_xticks([])
ax.set_title('b  The mouth, looking into the cove\n(x = into the page, dot = out of it)', loc='left', fontsize=11, fontweight='bold')

# ---------------------------------------------------------------- c: through a tide
ax = axd['c']
stages = ['strong flood', 'weak flood', 'slack', 'weak ebb', 'strong ebb']
SC = 0.055 # axis units per cm/s
for n, stg in enumerate(stages):
    r = ST.loc[stg]
    x0 = n * 1.0
    ax.add_patch(Rectangle((x0 + 0.08, 0), 0.84, 3.0, fc='#EEF4FA', ec='0.6', lw=1, zorder=0))
    ax.add_patch(Rectangle((x0 + 0.08, 0), 0.84, 0.75, fc='#D3E3F1', ec='none', zorder=0))
    xc = x0 + 0.5
    for y, val, lab, ls, lw in [(2.6, r['top mean cm/s'], 'top', '-', 3.0), (1.55, r['slab mean cm/s'], 'slab', '--', 2.0), (0.38, r['bottom mean cm/s'], 'bottom', '-', 3.0)]:
        col = C_SLAB if lab == 'slab' else (C_IN if val < 0 else C_OUT)
        L = val * SC
        if abs(L) < 0.02:
            ax.plot([xc], [y], 'o', color=col, ms=5)
        else:
            ax.add_patch(FancyArrowPatch((xc, y), (xc + L, y), arrowstyle='-|>', mutation_scale=16, color=col, lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=3))
        ax.text(xc + (0.08 if val >= 0 else -0.08), y + 0.17, '%+.1f' % val, ha='left' if val >= 0 else 'right', fontsize=8.5, color='0.2')
    ax.text(xc, 3.15, stg, ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(xc, -0.12, 'bottom against the top\n%.0f%% of hours' % r['bottom opposite to top %'], ha='center', va='top', fontsize=8.5)
for y, lab in [(2.6, 'top 3 m'), (1.55, 'slab'), (0.38, 'bottom 3 m')]:
    ax.text(-0.05, y, lab, ha='right', va='center', fontsize=9)
ax.set_xlim(-0.75, 5.0)
ax.set_ylim(-0.8, 3.6)
ax.axis('off')
ax.text(-0.7, -0.75, '<- IN (blue)      OUT (red) ->      arrows in cm/s; mean over all hours of each stage, 2024-2025. Low-water slack runs the same stages in reverse (slack bar includes both slacks).',
        fontsize=8.5, color='0.3')
ax.set_title('c  Through a tide: bottom water creeps IN around slack under an outflowing surface, and leaves mainly on strong ebbs',
             loc='left', fontsize=11, fontweight='bold')

# ---------------------------------------------------------------- d: seasons
ax = axd['d']
x = np.arange(len(SE))
ax.bar(x, SE.bottom_in_days_pct, 0.6, color=C_IN)
for xi, r in zip(x, SE.itertuples()):
    ax.text(xi, r.bottom_in_days_pct + 1.5, '%.0f%% of days\nnet %+.2f cm/s' % (r.bottom_in_days_pct, r.bottom), ha='center', va='bottom', fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels(SE.season)
ax.set_xlim(-0.6, len(SE) - 0.4)
ax.set_ylim(0, 65)
ax.set_ylabel('days with bottom water flowing IN [%]')
ax.grid(axis='y', color='lightgray', linestyle='--', alpha=0.5)
ax.set_title('d  By season (tidally filtered): always a\nsmall net inflow, but in winter it comes in bursts', loc='left', fontsize=11, fontweight='bold')

fig.suptitle('Where bottom water goes at the Penn Cove mouth (model %s, 2024-2025)' % GTX, fontsize=13)
fn = out_dir / 'pc_bottom_water_schematic.png'
fig.savefig(fn, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved ' + str(fn))
