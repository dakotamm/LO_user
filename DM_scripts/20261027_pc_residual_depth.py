"""
Depth structure of the residual (2-year mean) flow in Penn Cove.

  a-c  plan view of the mean velocity in three depth bands below the moving
       surface (0-4, 4-10, 10-bed m), from the Godin-filtered band velocities
       cached by 20261027_pc_lobe_cavity.py (pc_cove_godin_bands_*.npz).
       A band exists in a cell only where the mean column is >= band top + 2 m.
  d-e  the mean east-west velocity on two N-S sections, pc_cp (mid-cove) and the
       mouth (last rho column west of pc_lp), vs depth (sigma layers placed at
       their zeta = 0 depths), from the 3D band-fit means
       (20261027_tidal_ellipses_windowed.py, bin 'all'). Red = eastward = OUT.

Output: LO_output/DM_outs/20261027_pc_residual_depth/pc_residual_depth.png

  python 20261027_pc_residual_depth.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator
import cmocean

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

GTX, JOB, DS0, DS1 = 'wb1_t0_xn11abbur00', 'pc_cove', '2024.01.01', '2025.12.31'
Ldir = Lfun.Lstart(gridname='wb1')
box_dir = Ldir['LOo'] / 'extract' / GTX / 'box'
Z = np.load(box_dir / ('%s_godin_bands_%s_%s.npz' % (JOB, DS0, DS1)))
B = xr.open_dataset(Ldir['LOo'] / 'extract' / GTX / 'tidal_ellipses' / ('%s_%s_%s_bands.nc' % (JOB, DS0, DS1)))
ds = xr.open_dataset(box_dir / ('%s_%s_%s.nc' % (JOB, DS0, DS1)))
Hm = ds.h.values + ds.zeta.mean('ocean_time').values
ds.close()
out_dir = Ldir['LOo'] / 'DM_outs' / '20261027_pc_residual_depth'
Lfun.make_dir(out_dir)
sect_dir = Ldir['LOo'] / 'section_lines'
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)

lon, lat = B.lon_rho.values, B.lat_rho.values
mask = B.mask_rho.values == 1
plon, plat = pfun.get_plon_plat(lon, lat)
wb = pd.read_pickle(sect_dir / 'wb.p')
pth = MplPath(np.column_stack([wb.x.values, wb.y.values]))
show = mask & pth.contains_points(np.column_stack([lon.ravel(), lat.ravel()])).reshape(lon.shape)
NR, NC = mask.shape
aa = [plon.min(), plon.max(), plat.min(), plat.max()]
lat0 = 0.5 * (aa[2] + aa[3])
asp = (aa[3] - aa[2]) / ((aa[1] - aa[0]) * np.cos(np.deg2rad(lat0)))

BANDS = [('z0-4', 0, '0-4 m below the surface'), ('z4-10', 4, '4-10 m'), ('z10-bed', 10, '10 m to the bed')]
UV = {}
for b, top, _ in BANDS:
    MB = mask & (Hm >= top + 2.0)
    U = np.nanmean(Z['U_' + b], axis=0) * (MB[:, :-1] & MB[:, 1:])
    V = np.nanmean(Z['V_' + b], axis=0) * (MB[:-1, :] & MB[1:, :])
    ur = np.full((NR, NC), np.nan)
    vr = np.full((NR, NC), np.nan)
    ur[:, 1:-1] = 0.5 * (U[:, :-1] + U[:, 1:])
    vr[1:-1, :] = 0.5 * (V[:-1, :] + V[1:, :])
    keep = MB & show
    UV[b] = (np.where(keep, ur, np.nan), np.where(keep, vr, np.nan))
spd = {b: np.hypot(*UV[b]) for b in UV}
big = np.nanpercentile(np.concatenate([100 * spd[b][np.isfinite(spd[b])] for b in spd]), 98)

LON_CP = pd.read_pickle(sect_dir / 'pc_cp.p').x.mean()
LON_LP = pd.read_pickle(sect_dir / 'pc_lp.p').x.mean()
SECTS = [('pc_cp (mid-cove)', int(np.argmin(np.abs(lon[0] - LON_CP)))), ('mouth', np.flatnonzero(lon[0] < LON_LP)[-1])]
u3 = B['3d_umean'].sel(bin='all').values # (s_rho, eta, xi)
z0 = B.z0_rho.values
umax = max(np.nanmax(np.abs(100 * u3[:, show[:, i], i])) for _, i in SECTS)

fig, axd = plt.subplot_mosaic([['a', 'a', 'b', 'b', 'c', 'c'], ['d', 'd', 'd', 'e', 'e', 'e']], layout='constrained', figsize=(16, 2 * (5.3 * asp + 1.4)))
step = 2
jj, ii = np.meshgrid(np.arange(0, NR, step), np.arange(0, NC, step), indexing='ij')
for k, (b, _, lab) in zip('abc', BANDS):
    ax = axd[k]
    u, v = UV[b]
    cs = ax.pcolormesh(plon, plat, 100 * spd[b], cmap=cmocean.cm.speed, vmin=0, vmax=big, shading='flat')
    s = np.isfinite(u[jj, ii])
    q = ax.quiver(lon[jj, ii][s], lat[jj, ii][s], u[jj, ii][s], v[jj, ii][s], scale=(big / 100) / 0.3, scale_units='inches', width=0.004, color='k', zorder=3)
    pfun.add_coast(ax, color='gray', linewidth=0.5)
    ax.axis(aa)
    pfun.dar(ax)
    for x0, nm in [(lon[0, SECTS[0][1]], 'pc_cp'), (lon[0, SECTS[1][1]], 'mouth')]:
        ax.axvline(x0, color='k', ls=':', lw=1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=8)
    ax.grid(**GRID)
    ax.set_title('%s (mean speed %.2f cm/s over %d cells)' % (lab, np.nanmean(100 * spd[b]), np.isfinite(spd[b]).sum()), fontsize=10)
    if k == 'a':
        ax.quiverkey(q, 0.15, 0.07, 0.02, '2 cm/s', labelpos='N', coordinates='axes', fontproperties={'size': 8})
fig.colorbar(cs, ax=[axd[k] for k in 'abc'], shrink=0.8, label='mean speed [cm/s]')

for k, (nm, i) in zip(['d', 'e'], SECTS):
    ax = axd[k]
    jw = np.flatnonzero(show[:, i])
    yy = np.broadcast_to(lat[jw, i], (z0.shape[0], len(jw)))
    cs2 = ax.pcolormesh(yy, z0[:, jw, i], 100 * u3[:, jw, i], cmap=cmocean.cm.balance, vmin=-umax, vmax=umax, shading='nearest')
    ax.contour(yy, z0[:, jw, i], u3[:, jw, i], levels=[0], colors='k', linewidths=1)
    ax.set_xlabel('latitude (south <-  -> north)')
    ax.set_ylabel('depth at mean sea level [m]')
    ax.set_title('%s: mean east-west velocity (red = OUT, blue = IN; black = 0)' % nm, fontsize=10)
    ax.grid(**GRID)
fig.colorbar(cs2, ax=[axd['d'], axd['e']], shrink=0.8, label='mean eastward velocity [cm/s]')
for k in ['a', 'b', 'c', 'd', 'e']:
    axd[k].text(0.02, 0.97, k, transform=axd[k].transAxes, fontsize=13, fontweight='bold', va='top')
fig.suptitle('Residual (2-year mean) flow in Penn Cove by depth: %s, %s to %s' % (GTX, DS0, DS1), fontsize=11)
fn = out_dir / 'pc_residual_depth.png'
fig.savefig(fn, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved ' + str(fn))
for nm, i in SECTS:
    jw = np.flatnonzero(show[:, i])
    print(nm, 'u mean by layer (bottom->top, cm/s, section mean):', np.round(100 * np.nanmean(u3[:, jw, i], axis=1)[::5], 2))
