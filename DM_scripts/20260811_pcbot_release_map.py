"""
Where the pcbot particles start.

The pcbot release is defined by a rule rather than by a list of stations
(ic_from_tef2_segs_bottom() in LO_user/tracker2/experiments.py: every wet cell
of the pc_cp_m tef2 segment, particles at 0.25-4.75 m above the bed in 0.5 m
steps, capped at half the local depth), and a rule is exactly the kind of thing
that can be described correctly in a docstring and still do something else in
the water. This is the check.

The positions are NOT re-derived from the experiment code. They are read from
time index 0 of the tracker output itself, so what is drawn is what the run
actually released. Every pcbot run of both experiments -- the two matched-week
pairs and the spring-neap pair -- must release the same particles in the same
places or their retention curves are not comparable, so the script asserts that
across every run it finds rather than assuming it.

WHAT THE COLOUR CARRIES
Each cell is coloured by how many particles it holds, 7 to 10. That is not
decoration: it is the only part of the release rule that varies from cell to
cell, and it is where the half-depth cap shows up. A level is dropped whenever
it would start above half the local depth, so the shallow cells around the head
and the south shore get fewer levels than the deep channel. Without that cap a
7 m cell would be seeded to 4.75 m -- 68 per cent of the way up the column --
and shallow cells would quietly feed surface water into a bottom-water cohort.
The count reduces to floor(h + 0.5), capped at 10.

The vertical structure of the release is NOT shown here. In plan view the 1573
particles collapse onto 165 dots and the bottom-hugging stack is invisible; if
that is what you want, see the -section flag of an earlier version or read the
hab column of the printout below.

EXTENT is fitted to the release plus all three pc sections, so the figure shows
the whole system the cohort moves through: pc_cp (which it starts behind),
pc_lj, and pc_lp at the mouth.

Colours follow the recent Penn Cove maps: land on BED, sections in black,
cmcrameri ramps, DAR aspect, half a cell of margin on the limits.

run 20260811_pcbot_release_map.py
run 20260811_pcbot_release_map.py -run pcbot_3d_loDO
"""
import argparse
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cmcrameri import cm as cmc
from matplotlib.colors import BoundaryNorm, ListedColormap

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-gctag', default='wb1_pc1')
p.add_argument('-run', default='pcbot_3d_sh14_hiDO_2025',
               help='tracks2 run directory to read the release from')
p.add_argument('-seg', default='pc_cp_m', help='tef2 segment the release fills')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
trk = Ldir['LOo'] / 'tracks2' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pcbot_release_map'
Lfun.make_dir(out_dir)

FS = 20                  # tick labels land at 18, as on the pc4 points map
LAND = '#b0b0b0'         # flat gray, so the warm markers own the warm end
# The window is fitted to all three pc sections but only pc_cp is drawn. pc_cp
# is the one the release is defined against -- the cohort starts landward of
# it -- while pc_lj and pc_lp are just places the water later passes, and
# drawing them puts two heavy black bars in the panel that mark nothing about
# the release. They still set the extent, so the figure covers the whole
# system without claiming three boundaries matter here.
SECTS_EXTENT = ['pc_cp', 'pc_lj', 'pc_lp']
SECTS_DRAW = ['pc_cp']

# --------------------------------------------------------------- release ---
fns = sorted((trk / args.run).glob('release_*.nc'))
if len(fns) == 0:
    sys.exit('no release file in %s' % (trk / args.run))
d = xr.open_dataset(fns[0])
plon = d.lon.values[0, :]
plat = d.lat.values[0, :]
pcs0 = d.cs.values[0, :]
ph = d.h.values[0, :]
d.close()
hab = (pcs0 + 1) * ph
NP = len(plon)
print('release from %s' % fns[0].name)
print('  %d particles, %d cells' % (NP, len(set(zip(plon, plat)))))
print('  height above bed %.2f to %.2f m, local depth %.1f to %.1f m'
      % (hab.min(), hab.max(), ph.min(), ph.max()))

for other in sorted(trk.glob('pcbot_3d*')):
    if other.name == args.run:
        continue
    o = sorted(other.glob('release_*.nc'))
    if len(o) == 0:
        continue
    do = xr.open_dataset(o[0])
    same = (do.sizes['Particle'] == NP
            and np.allclose(do.lon.values[0, :], plon)
            and np.allclose(do.cs.values[0, :], pcs0))
    do.close()
    print('  %-26s identical release: %s' % (other.name, same))

# ------------------------------------------------------------------ grid ---
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
h, mask = g.h.values, g.mask_rho.values
g.close()
hw = np.where(mask == 1, h, np.nan)
land = np.where(mask == 0, 1.0, np.nan)
dlon = float(np.diff(lon[0, :]).mean())
dlat = float(np.diff(lat[:, 0]).mean())

sect_df = pickle.load(open(Ldir['LOo'] / 'extract' / 'tef2'
                           / ('sect_df_%s.p' % args.gctag), 'rb'))


def face_xy(sn):
    """A tef2 face lies between the two rho cells it separates."""
    s = sect_df[sect_df.sn == sn]
    fx = 0.5 * (lon[s.jrp.values, s.irp.values] + lon[s.jrm.values, s.irm.values])
    fy = 0.5 * (lat[s.jrp.values, s.irp.values] + lat[s.jrm.values, s.irm.values])
    return fx, fy


# particles per cell, which is what the half-depth cap acts on
cells = {}
for x, y in zip(plon, plat):
    cells[(x, y)] = cells.get((x, y), 0) + 1
cx = np.array([k[0] for k in cells])
cy = np.array([k[1] for k in cells])
cn = np.array([cells[k] for k in cells])
print('  particles per cell: %d to %d (median %d)'
      % (cn.min(), cn.max(), np.median(cn)))

# ---------------------------------------------------------------- extent ---
# Fitted to the release AND all three pc sections, so the window is set by the
# system rather than by the release alone. Half a cell of margin on top, since
# shading='nearest' centres each cell on its rho point and limits taken at the
# rho points would cut the outer cells in half.
xs = [cx.min(), cx.max()]
ys = [cy.min(), cy.max()]
for sn in SECTS_EXTENT:
    fx, fy = face_xy(sn)
    xs += [fx.min(), fx.max()]
    ys += [fy.min(), fy.max()]
PAD = 4
XL = (min(xs) - PAD * dlon, max(xs) + PAD * dlon)
YL = (min(ys) - PAD * dlat, max(ys) + PAD * dlat)
DAR = 1 / np.cos(np.deg2rad(float(np.mean(YL))))
print('  extent lon %.4f..%.4f, lat %.4f..%.4f' % (XL + YL))

# ---------------------------------------------------------------- figure ---
fig, ax = plt.subplots(figsize=(12, 7))

# Bathymetry on oslo_r (pale shallow, dark deep) and land flat on BED. A single
# ramp for both put the deep channel at the same tone as the land.
ax.pcolormesh(lon, lat, hw, cmap=cmc.oslo_r, vmin=0, vmax=60,
              shading='nearest', zorder=0, rasterized=True)
ax.pcolormesh(lon, lat, land, cmap=ListedColormap([LAND]), shading='nearest',
              zorder=1)

# Discrete, because the count takes four values and a continuous ramp would
# invite reading a gradient into them. lajolla runs dark -> cream, so plain
# ascending puts the capped cells at the dark end: 10 levels is the
# unremarkable majority (122 of 165 cells) and wants to be quiet, while 7 is
# the rule actually doing something and wants to be seen. Warm ramp against
# the cool bathymetry, as on the pc4 map.
lev = np.arange(cn.min(), cn.max() + 2) - 0.5
cmap = ListedColormap(cmc.lajolla(np.linspace(0.20, 0.90, len(lev) - 1)))
sc = ax.scatter(cx, cy, c=cn, s=52, cmap=cmap, norm=BoundaryNorm(lev, cmap.N),
                edgecolor='k', linewidth=0.4, zorder=6)

for sn in SECTS_DRAW:
    fx, fy = face_xy(sn)
    # casing only a little wider than the line: at 5.5 against 2.0 the white
    # read as the section and the black as an artifact of it
    ax.plot(fx, fy, '-', color='w', lw=6.0, alpha=0.9, solid_capstyle='butt',
            zorder=7)
    ax.plot(fx, fy, '-', color='k', lw=3.5, solid_capstyle='butt', zorder=8)

pfun.add_coast(ax)
ax.set_xlim(*XL)
ax.set_ylim(*YL)
ax.set_aspect(DAR)
ax.tick_params(labelsize=FS - 2)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))

# No title and no section names: this is a figure panel, captioned elsewhere as
# on 20260811_pc4_points_map.py. The colourbar IS annotated, though -- it is
# the only thing in the panel a reader cannot work out from the geometry, and
# an unlabelled discrete ramp of four browns and creams is unreadable without
# the caption in hand.
cb = fig.colorbar(sc, ax=ax, fraction=0.030, pad=0.02,
                  ticks=np.arange(cn.min(), cn.max() + 1))
cb.set_label('particles in the cell', fontsize=FS - 2)
cb.ax.tick_params(labelsize=FS - 2)

fig.tight_layout()
fn_out = out_dir / 'pcbot_release_map.png'
fig.savefig(fn_out, dpi=300, bbox_inches='tight', transparent=True)
plt.close(fig)
print('\nwrote %s' % fn_out)
