"""
Map of the Penn Cove TEF sections (pc_lp, pc_cp, pc_lj) from the wb1_pc1
collection: where each section sits on the grid and which way positive
transport points.

Section faces are drawn as the actual u/v grid faces, not a smoothed line,
since that is what the extraction integrates over.

run 20260921_pc_sections_map.py -gctag wb1_pc1
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

parser = argparse.ArgumentParser()
parser.add_argument('-gctag', default='wb1_pc1', type=str)
args = parser.parse_args()

gridname = args.gctag.split('_')[0]
Ldir = Lfun.Lstart(gridname=gridname)

tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'
sect_df = pd.read_pickle(tef2_dir / ('sect_df_' + args.gctag + '.p'))

dsg = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon = dsg.lon_rho.values
lat = dsg.lat_rho.values
lon_u, lat_u = dsg.lon_u.values, dsg.lat_u.values
lon_v, lat_v = dsg.lon_v.values, dsg.lat_v.values
mask = dsg.mask_rho.values
h = dsg.h.values
dsg.close()

sns = ['pc_lp', 'pc_cp', 'pc_lj']
sect_color = dict(zip(sns, ['magenta', 'tab:red', 'tab:orange']))

hm = np.where(mask == 1, h, np.nan)


def face_xy(d):
    du, dv = d[d.uv == 'u'], d[d.uv == 'v']
    xs = np.concatenate([lon_u[du.j.values, du.i.values],
                         lon_v[dv.j.values, dv.i.values]])
    ys = np.concatenate([lat_u[du.j.values, du.i.values],
                         lat_v[dv.j.values, dv.i.values]])
    return xs, ys


plt.close('all')
fig, ax = plt.subplots(figsize=(11, 9))

ax.pcolormesh(lon, lat, np.ma.masked_invalid(hm), cmap='Blues',
              vmin=0, vmax=60, shading='nearest', zorder=1)
ax.pcolormesh(lon, lat, np.ma.masked_invalid(np.where(mask == 1, np.nan, 1.)),
              cmap='Greys', vmin=0, vmax=3, shading='nearest', zorder=2)
pfun.add_coast(ax, color='k', linewidth=0.6)

for sn in sns:
    d = sect_df[sect_df.sn == sn]
    c = sect_color[sn]
    xs, ys = face_xy(d)
    ax.plot(xs, ys, 's', color=c, markersize=7, zorder=10)
    # positive transport goes from the minus rho cell to the plus rho cell
    r = d.iloc[len(d) // 2]
    x0, y0 = lon[r.jrm, r.irm], lat[r.jrm, r.irm]
    x1, y1 = lon[r.jrp, r.irp], lat[r.jrp, r.irp]
    ax.annotate('', xy=(x0 + 3 * (x1 - x0), y0 + 3 * (y1 - y0)),
                xytext=(x0, y0), zorder=13,
                arrowprops=dict(color=c, width=1.5, headwidth=9))
    ax.text(xs.mean(), ys.mean(), '%s (%d faces)' % (sn, len(d)), color=c,
            fontsize=11, fontweight='bold', ha='center', va='bottom', zorder=14,
            bbox=dict(fc='w', ec='none', alpha=0.7, pad=1))

# window on the whole cove plus a bit of Saratoga Passage, not just the faces
pfun.dar(ax)
ax.axis([-122.755, -122.615, 48.205, 48.275])
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Penn Cove TEF sections (%s)\narrow = positive transport direction'
             % args.gctag)
ax.legend(handles=[Line2D([], [], color=sect_color[sn], marker='s', ls='',
                          label=sn) for sn in sns],
          loc='lower right', fontsize=10, framealpha=0.9)

fig.tight_layout()
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)
fn_out = out_dir / ('20260921_pc_sections_map_' + args.gctag + '.png')
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close('all')
print('saved ' + str(fn_out))

for sn in sns:
    d = sect_df[sect_df.sn == sn]
    xs, ys = face_xy(d)
    print('%-8s %2d faces (%d u, %d v)  center %.4f, %.4f'
          % (sn, len(d), (d.uv == 'u').sum(), (d.uv == 'v').sum(),
             xs.mean(), ys.mean()))
