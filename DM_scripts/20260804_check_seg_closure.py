"""
Check that a tef2 section collection actually seals the region of interest,
before handing it to create_seg_info_dict.py.

create_seg_info_dict.py floods the rho grid outward from each section and
stops when it hits another section. If the sections do not form a complete
barrier -- together with land -- the fill escapes, walks to the edge of the
grid and dies with an IndexError. That is what a leak looks like, and it is
easier to find here than in the traceback.

This does the same fill, but:
  - blocks on u/v FACES the way the model does, not on rho cells;
  - guards the array bounds, so a leak is reported instead of crashing;
  - seeds from a point you choose (default: middle of Penn Cove);
  - tells you which grid edges the fill reached, and draws it.

The barrier convention matches create_sect_df.py:
  a u-face at (j,i) separates rho cells (j,i) and (j,i+1)
  a v-face at (j,i) separates rho cells (j,i) and (j+1,i)

run 20260804_check_seg_closure.py -gctag wb1_pc1
run 20260804_check_seg_closure.py -gctag wb1_pc1 -seedlon -122.35 -seedlat 47.95
"""
import argparse
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

parser = argparse.ArgumentParser()
parser.add_argument('-gctag', default='wb1_pc1', type=str)
# seed point for the fill -- default is the middle of Penn Cove
parser.add_argument('-seedlon', default=-122.68, type=float)
parser.add_argument('-seedlat', default=48.233, type=float)
args = parser.parse_args()

gridname = args.gctag.split('_')[0]
Ldir = Lfun.Lstart(gridname=gridname)

tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'
sect_df = pd.read_pickle(tef2_dir / ('sect_df_' + args.gctag + '.p'))
coll_dir = tef2_dir / ('sections_' + args.gctag)

dsg = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon = dsg.lon_rho.values
lat = dsg.lat_rho.values
mask = dsg.mask_rho.values
dsg.close()
NR, NC = mask.shape

# ---------------------------------------------------------------- barriers ---
# blocked_u[j,i] True  -> cannot step between (j,i) and (j,i+1)
# blocked_v[j,i] True  -> cannot step between (j,i) and (j+1,i)
blocked_u = np.zeros((NR, NC), dtype=bool)
blocked_v = np.zeros((NR, NC), dtype=bool)
u = sect_df[sect_df.uv == 'u']
v = sect_df[sect_df.uv == 'v']
blocked_u[u.j.values, u.i.values] = True
blocked_v[v.j.values, v.i.values] = True
print('barrier faces: %d u, %d v' % (blocked_u.sum(), blocked_v.sum()))

# ------------------------------------------------------------------- fill ---
j0 = zfun.find_nearest_ind(lat[:, 0], args.seedlat)
i0 = zfun.find_nearest_ind(lon[0, :], args.seedlon)
if mask[j0, i0] != 1:
    raise SystemExit('seed (%.4f, %.4f) -> (j=%d, i=%d) is on land'
                     % (args.seedlon, args.seedlat, j0, i0))
print('seed (%.4f, %.4f) -> j=%d, i=%d' % (args.seedlon, args.seedlat, j0, i0))

filled = np.zeros((NR, NC), dtype=bool)
filled[j0, i0] = True
q = deque([(j0, i0)])
edge_hits = set()

while q:
    j, i = q.popleft()
    # (dj, di, whether the step is blocked)
    steps = ((1, 0, blocked_v[j, i]),
             (-1, 0, blocked_v[j - 1, i] if j > 0 else True),
             (0, 1, blocked_u[j, i]),
             (0, -1, blocked_u[j, i - 1] if i > 0 else True))
    for dj, di, blocked in steps:
        if blocked:
            continue
        jj, ii = j + dj, i + di
        if jj < 0 or jj >= NR or ii < 0 or ii >= NC:
            edge_hits.add(('S' if jj < 0 else 'N') if dj else
                          ('W' if ii < 0 else 'E'))
            continue
        if mask[jj, ii] == 1 and not filled[jj, ii]:
            filled[jj, ii] = True
            q.append((jj, ii))

nwet = int((mask == 1).sum())
print('\nfilled %d cells of %d wet in the grid (%.1f%%)'
      % (filled.sum(), nwet, 100 * filled.sum() / nwet))

# a fill that reaches the outer ring of the grid has escaped
ring = np.zeros((NR, NC), dtype=bool)
ring[0, :] = ring[-1, :] = True
ring[:, 0] = ring[:, -1] = True
reached_ring = filled & ring

if len(edge_hits) == 0 and not reached_ring.any():
    print('SEALED -- the fill never reached the edge of the grid')
else:
    print('LEAK -- the fill reached the grid edge')
    if edge_hits:
        print('  stepped off edges: ' + ', '.join(sorted(edge_hits)))
    if reached_ring.any():
        jj, ii = np.where(reached_ring)
        print('  %d boundary cells reached, e.g.:' % len(jj))
        for k in np.linspace(0, len(jj) - 1, min(8, len(jj))).astype(int):
            print('    j=%3d i=%3d  lon %.4f  lat %.4f'
                  % (jj[k], ii[k], lon[jj[k], ii[k]], lat[jj[k], ii[k]]))

# ------------------------------------------------------------------- plot ---
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)

plt.close('all')
fig, ax = plt.subplots(figsize=(9, 11))
shade = np.full((NR, NC), np.nan)
shade[mask == 1] = 0
shade[filled] = 1
ax.pcolormesh(lon, lat, np.ma.masked_invalid(shade), cmap='Blues',
              vmin=0, vmax=1.6, shading='nearest')
pfun.add_coast(ax, color='gray', linewidth=0.5)
for fn in sorted(coll_dir.glob('*.p')):
    d = pd.read_pickle(fn)
    ax.plot(d.x, d.y, '-', color='magenta', lw=1.5)
    ax.text(d.x.mean(), d.y.mean(), '  ' + fn.stem, color='magenta', fontsize=8)
ax.plot(lon[j0, i0], lat[j0, i0], '*', color='yellow', markersize=18,
        markeredgecolor='k')
if reached_ring.any():
    jj, ii = np.where(reached_ring)
    ax.plot(lon[jj, ii], lat[jj, ii], '.', color='red', markersize=2)
pfun.dar(ax)
ax.axis([lon.min(), lon.max(), lat.min(), lat.max()])
ax.set_title('%s: fill from seed (dark = reached)\n%s'
             % (args.gctag,
                'SEALED' if not (edge_hits or reached_ring.any()) else
                'LEAK -- red = grid-edge cells reached'))
fig.tight_layout()
fn_out = out_dir / ('20260804_seg_closure_' + args.gctag + '.png')
fig.savefig(fn_out, dpi=200)
print('\nsaved ' + str(fn_out))
