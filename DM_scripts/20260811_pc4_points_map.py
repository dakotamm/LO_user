"""
Where the three pc4 moorings are.

The stacks in 20260811_pc_forcing_stack.py are panels of series at named points,
and a reader has no way to check that "inner cove", "mouth" and "outside" mean
what they sound like, how far apart they are, or how deep. This is that check.

The points are NOT re-derived here: they are read from
LO_user/extract/moor/job_lists.py via get_sta_dict('pc4'), the same dict the
extraction used, so the map cannot drift from the data.

  cp_mid  midpoint of the pc_cp section (Coupeville line), inner cove
  lp_mid  midpoint of the pc_lp section (Long Point line), cove mouth
  M5      Saratoga Passage, outside; same point as pc0's M5

SECTION LINES are drawn from their DEFINITIONS in
LO_output/extract/tef2/sections_wb1_pc1/*.p -- the full shore-to-shore line as
specified, not the subset of u-faces that landed on wet cells. Drawing the wet
faces instead makes each section look shorter than it is and makes the midpoint
stations look off-centre, since a face is dropped wherever either neighbouring
rho cell is land.

Styling follows 20260807_grid_bathy_ppt.py so this sits beside the grid maps in
a deck: cmocean deep bathymetry, filled land, transparent background, large
fonts, and no text on the map itself. Point colors are sampled from cmcrameri
lajolla at 0.05 / 0.50 / 0.97, matching the three markers on the hypoxic-days
map, so a reader moving between figures keeps the same inner/mouth/outside
color sense.

run 20260811_pc4_points_map.py
"""
import argparse
import pickle
import sys

import cmocean
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cmcrameri import cm as cmc

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-gctag', default='wb1_pc1', type=str)
p.add_argument('-job', default='pc4', type=str)
args = p.parse_args()

Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
sys.path.append(str(Ldir['LOu'] / 'extract' / 'moor'))
from job_lists import get_sta_dict

sect_dir = Ldir['LOo'] / 'extract' / 'tef2' / ('sections_%s' % args.gctag)
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc4_points_map'
Lfun.make_dir(out_dir)

# ---- styling, lifted from 20260807_grid_bathy_ppt.py ----------------------
TEXT_COLOR = 'k'                 # 'k' for light slides, 'w' for dark slides
CMAP = cmocean.cm.deep
# land is filled rather than left transparent -- a see-through landmask reads as
# whatever the slide background is, which kills the coastline
LAND_COLOR = '#e8e4dc'

mpl.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'axes.edgecolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'savefig.transparent': True,
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
})
SAVE_KW = dict(dpi=300, bbox_inches='tight', transparent=True, facecolor='none')

# inner / mouth / outside, sampled off lajolla as on the hypoxic-days map
PC = {'cp_mid': mcolors.to_hex(cmc.lajolla(0.05)),
      'lp_mid': mcolors.to_hex(cmc.lajolla(0.50)),
      'M5': mcolors.to_hex(cmc.lajolla(0.97))}
SECT_C = {'pc_cp': PC['cp_mid'], 'pc_lp': PC['lp_mid']}
AA = [-122.745, -122.545, 48.200, 48.262]

sta_dict = get_sta_dict(args.job)
print('job %s: %s' % (args.job, ', '.join(sta_dict)))

# ---- grid ----------------------------------------------------------------
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
h, mask = g.h.values, g.mask_rho.values
g.close()
plon, plat = pfun.get_plon_plat(lon, lat)
Lon, Lat = lon[0, :], lat[:, 0]
hw = np.ma.masked_where(mask != 1, h)
land = np.ma.masked_where(mask == 1, mask)

# ---- figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5.6))

ax.pcolormesh(plon, plat, land, shading='flat', zorder=0, rasterized=True,
              cmap=mpl.colors.ListedColormap([LAND_COLOR]))
# Depth range is taken from the cells IN VIEW, not from the whole grid. The
# grid-map script scales to the domain max because it shows the domain; here
# that max (~280 m, out in the main basin) flattens Penn Cove's 15-30 m into one
# pale tone and the bathymetry stops carrying any information.
inview = ((lon >= AA[0]) & (lon <= AA[1]) & (lat >= AA[2]) & (lat <= AA[3])
          & (mask == 1))
vmax = np.ceil(np.nanmax(h[inview]) / 10) * 10
print('depth range in view: 0 to %.0f m' % vmax)
cs = ax.pcolormesh(plon, plat, hw, cmap=CMAP, shading='flat', zorder=1,
                   vmin=0, vmax=vmax, rasterized=True)

# sections at full specified length, shore to shore
for sn, c in SECT_C.items():
    sdf = pickle.load(open(sect_dir / (sn + '.p'), 'rb'))
    ax.plot(sdf.x.values, sdf.y.values, '-', color='w', lw=7.0, zorder=3,
            solid_capstyle='round')
    ax.plot(sdf.x.values, sdf.y.values, '-', color=c, lw=4.0, zorder=4,
            solid_capstyle='round')

for sn, (slon, slat) in sta_dict.items():
    i = int(np.argmin(np.abs(Lon - slon)))
    j = int(np.argmin(np.abs(Lat - slat)))
    print('  %-7s (%.6f, %.6f) -> i=%d j=%d, h = %.1f m'
          % (sn, slon, slat, i, j, h[j, i]))
    ax.plot(slon, slat, 'o', mfc=PC.get(sn, 'w'), mec='w', mew=2.6, ms=19,
            zorder=5)
    ax.plot(slon, slat, 'o', mfc='none', mec=TEXT_COLOR, mew=1.2, ms=19,
            zorder=6)

pfun.add_coast(ax, color=TEXT_COLOR, linewidth=0.8)
# set_xticks/set_yticks re-autoscale, and a rounded tick outside the grid then
# drags the view past the domain edge -- so pin the limits afterwards
ax.set_xticks(np.linspace(AA[0], AA[1], 5).round(2))
ax.set_yticks(np.linspace(AA[2], AA[3], 4).round(2))
ax.axis(AA)
ax.set_autoscale_on(False)
pfun.dar(ax)
ax.tick_params(length=6, labelrotation=0)
ax.set_xlabel('Longitude [$^{\\circ}$E]')
ax.set_ylabel('Latitude [$^{\\circ}$N]')
for s in ax.spines.values():
    s.set_visible(True)

cb = fig.colorbar(cs, ax=ax, shrink=0.85, pad=0.02, extend='max')
# depth increases downward, so the bar reads the way the water column does
cb.ax.invert_yaxis()
cb.set_label('Depth [m]', color=TEXT_COLOR)
cb.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
cb.outline.set_edgecolor(TEXT_COLOR)

fig.tight_layout()
fn_out = out_dir / ('pc4_points_map_%s.png' % args.job)
fig.savefig(fn_out, **SAVE_KW)
plt.close(fig)
print('\nwrote ' + str(fn_out))
