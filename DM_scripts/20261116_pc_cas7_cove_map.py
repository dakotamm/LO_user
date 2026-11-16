"""
Map of the cas7 Penn Cove cells used by 20261116_pc_cas7_bottom_DO_cove.py (same mask rule).
cas7 h on wet rho cells with cell edges; each cove cell outlined in red; mouth line LON_MOUTH dashed; seed star;
wb1 coastline (black) for reference.
OUT    LO_output/DM_outs/20261116_pc_cas7_cove_map/
"""

import pickle
import numpy as np
import xarray as xr
from scipy.ndimage import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

LON_MOUTH = -122.652                       # wb1 pc_lp line
SEED = (-122.70878, 48.22710)              # single-point cell
BOX_LAT = (48.20, 48.26)
AX = [-122.76, -122.60, 48.20, 48.26]

Ldir = Lfun.Lstart(gridname='cas7', tag='t1', ex_name='x11ab')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261116_pc_cas7_cove_map'
Lfun.make_dir(out_dir)

g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
lonp, latp = g.lon_psi.values, g.lat_psi.values
mask, h = g.mask_rho.values, g.h.values
pm, pn = g.pm.values, g.pn.values
g.close()

wet = (mask == 1) & (lon < LON_MOUTH) & (lat > BOX_LAT[0]) & (lat < BOX_LAT[1])
lab, _ = label(wet)
js, is_ = np.unravel_index(np.argmin((lon - SEED[0])**2 + (lat - SEED[1])**2), lon.shape)
cove = lab == lab[js, is_]
J, I = np.where(cove)
area = (1 / (pm * pn))[cove].sum()
print(f'cove: {cove.sum()} cells  j {J.min()}-{J.max()}  i {I.min()}-{I.max()}  h {h[cove].min():.1f}-{h[cove].max():.1f} m  '
      f'area {area / 1e6:.2f} km2')

d = pickle.load(open(Ldir['LOo'] / 'DM_outs' / '20260806_hypoxia' /
                     'hypoxia_lowpass_wb1_t0_xn11abbur00_2024.01.01_2025.12.31.p', 'rb'))

fig, ax = plt.subplots(figsize=(9, 5))
cs = ax.pcolormesh(lonp, latp, np.where(mask == 1, h, np.nan)[1:-1, 1:-1], cmap='Blues', vmin=0, vmax=60,
                   edgecolors='0.7', linewidth=0.2)
fig.colorbar(cs, ax=ax, label='h [m]')
for j, i in zip(J, I):   # outline each cove cell on its psi corners
    ax.plot([lonp[j-1, i-1], lonp[j-1, i], lonp[j, i], lonp[j, i-1], lonp[j-1, i-1]],
            [latp[j-1, i-1], latp[j-1, i], latp[j, i], latp[j, i-1], latp[j-1, i-1]], '-', color='tab:red', lw=1.2)
ax.contour(d['lon'], d['lat'], d['water'].astype(float), [0.5], colors='k', linewidths=0.6)
ax.axvline(LON_MOUTH, color='tab:red', ls='--', lw=1)
ax.plot(*SEED, '*', color='gold', mec='k', ms=12, zorder=5)
ax.axis(AX)
ax.set_aspect(1 / np.cos(np.deg2rad(48.23)))
ax.set_xticks([-122.75, -122.70, -122.65, -122.60])
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title(f'cas7 Penn Cove: {cove.sum()} cells, {area / 1e6:.1f} km$^2$, h {h[cove].min():.1f}-{h[cove].max():.1f} m '
             f'(red); wb1 coast (black)', fontsize=10)
fig.tight_layout()
fig_fn = out_dir / 'pc_cas7_cove_map.png'
fig.savefig(fig_fn, dpi=200, transparent=True)
plt.close(fig)
print(f'saved: {fig_fn}')
