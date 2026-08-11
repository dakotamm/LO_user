"""
Where the three pc4 moorings are.

The stack in 20260811_pc_forcing_stack.py is four panels of series at named
points, and a reader has no way to check that "inner cove", "mouth" and
"outside" mean what they sound like, how far apart they are, or how deep. This
is that check.

The points are NOT re-derived here: they are read from
LO_user/extract/moor/job_lists.py via get_sta_dict('pc4'), the same dict the
extraction used, so the map cannot drift from the data. Depths are read from
grid.nc at the cell each point snaps to, which is what extract_moor.py does.

  cp_mid  midpoint of the pc_cp section (Coupeville line), inner cove
  lp_mid  midpoint of the pc_lp section (Long Point line), cove mouth
  M5      Saratoga Passage, outside; same point as pc0's M5

Both TEF sections are drawn face by face from sect_df_wb1_pc1.p, so the two
"midpoint" stations are visibly the middle of the line they belong to rather
than a claim in a caption.

Point colors are sampled from cmcrameri lajolla at 0.05 / 0.50 / 0.97, matching
the three markers on the hypoxic-days map, so a reader moving between the two
figures keeps the same inner / mouth / outside color sense. Bathymetry is
grayscale underneath so the lajolla markers stay legible on it.

run 20260811_pc4_points_map.py
run 20260811_pc4_points_map.py -job pc4
"""
import argparse
import pickle
import sys

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

out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc4_points_map'
Lfun.make_dir(out_dir)

# inner / mouth / outside, sampled off lajolla as on the hypoxic-days map
PC = {'cp_mid': mcolors.to_hex(cmc.lajolla(0.05)),
      'lp_mid': mcolors.to_hex(cmc.lajolla(0.50)),
      'M5': mcolors.to_hex(cmc.lajolla(0.97))}
SECT_C = {'pc_cp': PC['cp_mid'], 'pc_lp': PC['lp_mid']}

sta_dict = get_sta_dict(args.job)
print('job %s: %s' % (args.job, ', '.join(sta_dict)))

# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
h, mask = g.h.values, g.mask_rho.values
plon, plat = pfun.get_plon_plat(lon, lat)
Lon, Lat = lon[0, :], lat[:, 0]

hw = np.where(mask == 1, h, np.nan)          # water depth, land blanked
land = np.where(mask == 0, 1.0, np.nan)      # land, for a flat gray fill

sect_df = pickle.load(open(Ldir['LOo'] / 'extract' / 'tef2'
                           / ('sect_df_%s.p' % args.gctag), 'rb'))

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.2))

# Water on a blue ramp and land a flat gray: a single gray ramp for depth put
# 90 m of Saratoga Passage at the same tone as the land, which made the two
# unreadable against each other on the first pass.
ax.pcolormesh(plon, plat, hw, cmap='Blues', vmin=-30, vmax=110, zorder=-5)
ax.pcolormesh(plon, plat, land, cmap=mcolors.ListedColormap(['#b0b0b0']),
              zorder=-4)
cs = ax.contour(lon, lat, np.where(mask == 1, h, np.nan), levels=[20, 40, 80],
                colors='#4a4a4a', linewidths=0.6, alpha=0.55, zorder=-3)
ax.clabel(cs, fmt='%.0f m', fontsize=12)

# the two sections, face by face, with a white casing so the dark pc_cp color
# still reads where it crosses the near-black cp_mid marker
for sn, c in SECT_C.items():
    s = sect_df[sect_df.sn == sn]
    if len(s) == 0:
        continue
    # a face lies between the two rho cells it separates
    fx = 0.5 * (lon[s.jrp.values, s.irp.values] + lon[s.jrm.values, s.irm.values])
    fy = 0.5 * (lat[s.jrp.values, s.irp.values] + lat[s.jrm.values, s.irm.values])
    ax.plot(fx, fy, '-', color='w', lw=6.0, alpha=0.9, solid_capstyle='butt',
            zorder=3)
    ax.plot(fx, fy, '-', color=c, lw=3.5, solid_capstyle='butt',
            zorder=4, label='%s section' % sn)

for sn, (slon, slat) in sta_dict.items():
    i = int(np.argmin(np.abs(Lon - slon)))
    j = int(np.argmin(np.abs(Lat - slat)))
    print('  %-7s (%.6f, %.6f) -> i=%d j=%d, h = %.1f m'
          % (sn, slon, slat, i, j, h[j, i]))
    ax.plot(slon, slat, 'o', mfc=PC.get(sn, 'w'), mec='w', mew=2.6, ms=17,
            zorder=5)
    ax.plot(slon, slat, 'o', mfc='none', mec='k', mew=1.1, ms=17, zorder=6)

pfun.add_coast(ax)
pfun.dar(ax)
ax.set_xlim(-122.745, -122.545)
ax.set_ylim(48.200, 48.262)
# no title, no station labels, no legend: this is a figure panel, captioned
# elsewhere. Sparse ticks in the style of OSM_fig_2.py.
ax.set_xticks([-122.72, -122.68, -122.64, -122.60, -122.56])
ax.set_yticks([48.21, 48.23, 48.25])
ax.tick_params(labelsize=18)

fig.tight_layout()
fn_out = out_dir / ('pc4_points_map_%s.png' % args.job)
fig.savefig(fn_out, dpi=500, bbox_inches='tight', transparent=True)
print('\nwrote ' + str(fn_out))
