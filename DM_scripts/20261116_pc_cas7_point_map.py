"""
Map of a cas7 extraction point: the 20261116_pc_cas7_bottom_DO Penn Cove point (default), or any -lon/-lat, e.g.
the 20261117_pc_saratoga_strat_DO Saratoga Passage station (M5):
    python 20261116_pc_cas7_point_map.py -lon ' -122.582462' -lat 48.245981 -sn M5
LEFT   wb1_t0_xn11abbur00 fraction of 2024-25 days with bottom DO < 2 mg/L (20260806_hypoxia RUN['fhyp_bot_2']).
RIGHT  cas7 grid: h on wet rho cells, cell edges drawn; the cell the point snaps to (nearest wet rho, same rule as
       the extraction script) outlined in red.
Star = requested point.
OUT    LO_output/DM_outs/20261116_pc_cas7_point_map/<sn>_cas7_point_map.png
"""

import argparse
import pickle
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-lon', type=float, default=-122.70878)
parser.add_argument('-lat', type=float, default=48.22710)
parser.add_argument('-sn', type=str, default='pc_west')
args = parser.parse_args()
LON0, LAT0 = args.lon, args.lat
AX = [-122.76, -122.56, 48.19, 48.27]

Ldir = Lfun.Lstart(gridname='cas7', tag='t1', ex_name='x11ab')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261116_pc_cas7_point_map'
Lfun.make_dir(out_dir)

# ---- wb1 hypoxic-day fraction ----
d = pickle.load(open(Ldir['LOo'] / 'DM_outs' / '20260806_hypoxia' /
                     'hypoxia_lowpass_wb1_t0_xn11abbur00_2024.01.01_2025.12.31.p', 'rb'))
fh = np.where(d['water'], d['RUN']['fhyp_bot_2'], np.nan)

# ---- cas7 grid and snapped cell ----
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
lonp, latp = g.lon_psi.values, g.lat_psi.values
mask, h = g.mask_rho.values, g.h.values
g.close()
dist2 = ((lon - LON0) * np.cos(np.deg2rad(LAT0)))**2 + (lat - LAT0)**2
dist2[mask != 1] = np.inf
j, i = np.unravel_index(np.argmin(dist2), dist2.shape)
print(f'cas7 cell: j={j} i={i}  lon {lon[j, i]:.5f}  lat {lat[j, i]:.5f}  h {h[j, i]:.1f} m  '
      f'({np.sqrt(dist2[j, i]) * 111:.2f} km from the requested point)')
hw = np.where(mask == 1, h, np.nan)

fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

ax = axs[0]
cs = ax.pcolormesh(d['lon'], d['lat'], fh, cmap='Reds', vmin=0, vmax=0.3, shading='nearest')
fig.colorbar(cs, ax=ax, label='fraction of days, bottom DO < 2 mg/L')
ax.contour(d['lon'], d['lat'], d['water'].astype(float), [0.5], colors='k', linewidths=0.6)
ax.set_title('wb1_t0_xn11abbur00 (2024-25)')

ax = axs[1]
cs = ax.pcolormesh(lonp, latp, hw[1:-1, 1:-1], cmap='Blues', vmin=0, vmax=60, edgecolors='0.6', linewidth=0.2)
fig.colorbar(cs, ax=ax, label='h [m]')
ax.plot([lonp[j-1, i-1], lonp[j-1, i], lonp[j, i], lonp[j, i-1], lonp[j-1, i-1]],
        [latp[j-1, i-1], latp[j-1, i], latp[j, i], latp[j, i-1], latp[j-1, i-1]], '-', color='tab:red', lw=2)
ax.set_title(f'cas7 {args.sn}: snapped cell j={j} i={i}, h = {h[j, i]:.1f} m')

for ax in axs:
    ax.plot(LON0, LAT0, '*', color='gold', mec='k', ms=14, zorder=5)
    ax.axis(AX)
    ax.set_aspect(1 / np.cos(np.deg2rad(LAT0)))
    ax.set_xticks([-122.75, -122.70, -122.65, -122.60])
    ax.set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')
fig.tight_layout()
fig_fn = out_dir / (args.sn + '_cas7_point_map.png')
fig.savefig(fig_fn, dpi=200, transparent=True)
plt.close(fig)
print(f'saved: {fig_fn}')
