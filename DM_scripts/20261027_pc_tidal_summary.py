"""
One-figure summary of the tidal flow in Penn Cove: where it is, how it varies
with depth, and how it varies over the spring/neap (semidiurnal, D2) and
tropic/equatorial (diurnal, D1) cycles.

Input: the band-fit composites from 20261027_tidal_ellipses_windowed.py
(<job>_<ds0>_<ds1>_bands.nc: 73 h fits every 24 h, binned by terciles of the
box-mean zeta band amplitude).

  a, b  D2 depth-averaged ellipses at spring / neap
  d, e  D1 depth-averaged ellipses at tropic / equatorial
        (one color scale and one ellipse scale for all four maps)
  c     section-mean semi-major axis vs depth at the mouth (last rho column
        west of pc_lp) for the four cases
  f     current ratio (high/low phase of each cycle) next to the zeta ratio,
        inner cove (west of pc_cp) and outer cove (pc_cp to pc_lp): equal bars
        mean the current scales with the tide that drives it

Output: LO_output/DM_outs/20261027_pc_tidal_summary/pc_tidal_summary.png

  python 20261027_pc_tidal_summary.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator
import cmocean

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

GTX, JOB, DS0, DS1 = 'wb1_t0_xn11abbur00', 'pc_cove', '2024.01.01', '2025.12.31'
Ldir = Lfun.Lstart(gridname='wb1')
B = xr.open_dataset(Ldir['LOo'] / 'extract' / GTX / 'tidal_ellipses' / ('%s_%s_%s_bands.nc' % (JOB, DS0, DS1)))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261027_pc_tidal_summary'
Lfun.make_dir(out_dir)
sect_dir = Ldir['LOo'] / 'section_lines'
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
CASES = [('D2', 'spring', '#0072B2', '-'), ('D2', 'neap', '#0072B2', '--'), ('D1', 'tropic', '#D55E00', '-'), ('D1', 'equatorial', '#D55E00', '--')]
NAME = {'D2': 'semidiurnal', 'D1': 'diurnal'}

lon, lat = B.lon_rho.values, B.lat_rho.values
plon, plat = pfun.get_plon_plat(lon, lat)
dx = np.nanmedian(np.diff(lon[0]))
dy = np.nanmedian(np.diff(lat[:, 0]))
wb = pd.read_pickle(sect_dir / 'wb.p')
pth = MplPath(np.column_stack([wb.x.values, wb.y.values]))
show = (B.mask_rho.values == 1) & pth.contains_points(np.column_stack([lon.ravel(), lat.ravel()])).reshape(lon.shape)
LON_CP = pd.read_pickle(sect_dir / 'pc_cp.p').x.mean()
LON_LP = pd.read_pickle(sect_dir / 'pc_lp.p').x.mean()
inner = show & (lon < LON_CP)
outer = show & (lon >= LON_CP) & (lon < LON_LP)
imouth = np.flatnonzero(lon[0] < LON_LP)[-1]
jm = np.flatnonzero(show[:, imouth])
z0m = np.nanmean(B.z0_rho.values[:, jm, imouth], axis=1)
aa = [plon.min(), plon.max(), plat.min(), plat.max()]
lat0 = 0.5 * (aa[2] + aa[3])
asp = (aa[3] - aa[2]) / ((aa[1] - aa[0]) * np.cos(np.deg2rad(lat0)))
m_per_deg = 111.32e3
step = 2

L = {(b, s): B.bar_Lsmaj.sel(band=b, bin=s).values for b, s, _, _ in CASES}
Ln = {(b, s): B.bar_Lsmin.sel(band=b, bin=s).values for b, s, _, _ in CASES}
TH = {(b, s): B.bar_theta.sel(band=b, bin=s).values for b, s, _, _ in CASES}
ZA = {(b, s): B.zeta_A.sel(band=b, bin=s).values for b, s, _, _ in CASES}
vmax = np.nanpercentile(np.concatenate([100 * L[k][show] for k in L]), 99)
big = vmax / 100

def glyphs(ax, a_, b_, th_):
    jj, ii = np.meshgrid(np.arange(0, lon.shape[0], step), np.arange(0, lon.shape[1], step), indexing='ij')
    sel = show[jj, ii] & np.isfinite(a_[jj, ii])
    sc = 0.5 * step * dy * m_per_deg / big
    ph = np.linspace(0, 2 * np.pi, 41)
    segs = []
    for j, i in zip(jj[sel], ii[sel]):
        x, y = a_[j, i] * np.cos(ph), b_[j, i] * np.sin(ph)
        t = np.deg2rad(th_[j, i])
        X = (x * np.cos(t) - y * np.sin(t)) * sc / (m_per_deg * np.cos(np.deg2rad(lat[j, i])))
        Y = (x * np.sin(t) + y * np.cos(t)) * sc / m_per_deg
        segs.append(np.column_stack([lon[j, i] + X, lat[j, i] + Y]))
    ax.add_collection(LineCollection(segs, colors='k', linewidths=0.6, zorder=3))
    return sc

pw = 5.2
fig, axd = plt.subplot_mosaic([['a', 'b', 'c'], ['d', 'e', 'f']], layout='constrained',
                              figsize=(3 * pw, 2 * (pw * asp + 1.3)), gridspec_kw={'width_ratios': [1, 1, 0.8]})
for k, (b, s, col, ls) in zip('abde', CASES):
    ax = axd[k]
    cs = ax.pcolormesh(plon, plat, np.where(show, 100 * L[(b, s)], np.nan), cmap=cmocean.cm.speed, vmin=0, vmax=vmax, shading='flat')
    sc = glyphs(ax, L[(b, s)], Ln[(b, s)], TH[(b, s)])
    pfun.add_coast(ax, color='gray', linewidth=0.5)
    ax.axis(aa)
    pfun.dar(ax)
    for x0 in [LON_CP]:
        ax.axvline(x0, color='k', ls=':', lw=1)
    ax.axvline(lon[0, imouth], color='k', ls='-', lw=1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=8)
    ax.grid(**GRID)
    ax.set_title('%s, %s: zeta %.2f m, current %.1f cm/s (cove mean)' % (NAME[b], s, np.nanmean(ZA[(b, s)][show]), 100 * np.nanmean(L[(b, s)][show])), fontsize=9)
    if k == 'a':
        ref = 5.0 if vmax >= 5 else 2.0
        x0, y0 = aa[0] + 0.05 * (aa[1] - aa[0]), aa[2] + 0.06 * (aa[3] - aa[2])
        Lr = ref / 100 * sc / (m_per_deg * np.cos(np.deg2rad(lat0)))
        ax.plot([x0, x0 + Lr], [y0, y0], 'k-', lw=2)
        ax.text(x0 + Lr / 2, y0 + 0.02 * (aa[3] - aa[2]), '%g cm/s' % ref, ha='center', va='bottom', fontsize=8)
        ax.text(LON_CP, aa[3] - 0.02 * (aa[3] - aa[2]), ' pc_cp', fontsize=8, va='top')
        ax.text(lon[0, imouth], aa[3] - 0.02 * (aa[3] - aa[2]), 'mouth ', fontsize=8, va='top', ha='right')
fig.colorbar(cs, ax=[axd['a'], axd['b'], axd['d'], axd['e']], shrink=0.7, label='semi-major axis (peak tidal current), depth-averaged [cm/s]')

ax = axd['c']
for b, s, col, ls in CASES:
    prof = np.nanmean(B['3d_Lsmaj'].sel(band=b, bin=s).values[:, jm, imouth], axis=1)
    ax.plot(100 * prof, z0m, ls=ls, color=col, lw=2, label='%s %s' % (NAME[b], s))
ax.set_xlabel('semi-major axis at the mouth [cm/s]\n(section mean)')
ax.set_ylabel('depth at mean sea level [m]')
ax.set_title('surface-intensified; shape set by depth, size by the cycle', fontsize=9)
ax.legend(fontsize=8, loc='lower right')
ax.grid(**GRID)

ax = axd['f']
rows = []
for reg, msk in [('inner', inner), ('outer', outer)]:
    for b, hi, lo in [('D2', 'spring', 'neap'), ('D1', 'tropic', 'equatorial')]:
        rc = np.nanmean(L[(b, hi)][msk]) / np.nanmean(L[(b, lo)][msk])
        rz = np.nanmean(ZA[(b, hi)][msk]) / np.nanmean(ZA[(b, lo)][msk])
        rows.append((reg, b, hi, lo, rc, rz))
x = np.arange(len(rows))
ax.bar(x - 0.18, [r[5] for r in rows], 0.34, color='0.75', label='zeta (the forcing)')
ax.bar(x + 0.18, [r[4] for r in rows], 0.34, color=['#0072B2' if r[1] == 'D2' else '#D55E00' for r in rows], label='current')
for xi, r in zip(x, rows):
    ax.text(xi + 0.18, r[4] + 0.03, '%.2f' % r[4], ha='center', fontsize=8)
    ax.text(xi - 0.18, r[5] + 0.03, '%.2f' % r[5], ha='center', fontsize=8, color='0.35')
ax.set_xticks(x)
ax.set_xticklabels(['%s cove\n%s/%s' % (r[0], r[2], r[3]) for r in rows], fontsize=8)
ax.axhline(1, color='0.3', lw=0.8)
ax.set_ylabel('ratio, high / low phase of the cycle')
ax.set_title('does the current follow the forcing?', fontsize=9)
ax.set_ylim(0, 3.1)
ax.legend(fontsize=8, loc='upper center', ncol=2)
ax.grid(axis='y', **GRID)
for k in 'abcdef':
    axd[k].text(0.02, 0.97, k, transform=axd[k].transAxes, fontsize=13, fontweight='bold', va='top')
fig.suptitle('Tidal currents in Penn Cove: semidiurnal (spring vs neap) and diurnal (tropic vs equatorial), %s, %s to %s' % (GTX, DS0, DS1), fontsize=11)
fn = out_dir / 'pc_tidal_summary.png'
fig.savefig(fn, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved ' + str(fn))
for r in rows:
    print('%s %s %s/%s: current %.2f, zeta %.2f' % r)
