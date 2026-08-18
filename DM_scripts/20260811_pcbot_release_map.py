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

STYLING is lifted from 20260811_pc4_points_map.py (which took it from
20260807_grid_bathy_ppt.py): cmocean deep bathymetry scaled to the depths IN
VIEW, filled land on LAND_COLOR, the section shore-to-shore from section_lines
with a white casing, the rcParams block, DAR aspect, pinned limits, and
transparent save. Two colourbars, because this figure carries two quantities:
depth, which is context, and particles per cell, which is the result.

run 20260811_pcbot_release_map.py
run 20260811_pcbot_release_map.py -run pcbot_3d_loDO
"""
import argparse
import pickle
import sys

import cmocean
import matplotlib as mpl
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

# ---- styling, as on 20260811_pc4_points_map.py ---------------------------
TEXT_COLOR = 'k'                 # 'k' for light slides, 'w' for dark slides
CMAP = cmocean.cm.deep
LAND_COLOR = '#e8e4dc'           # filled, not transparent: a see-through
                                 # landmask reads as the slide background and
                                 # kills the coastline

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
plon_g, plat_g = pfun.get_plon_plat(lon, lat)
hw = np.ma.masked_where(mask != 1, h)
land = np.ma.masked_where(mask == 1, mask)
dlon = float(np.diff(lon[0, :]).mean())
dlat = float(np.diff(lat[:, 0]).mean())

# Section lines shore to shore, as on the pc4 map -- the specified line, not the
# staircase of tef2 faces it snapped to.
sect_dir = Ldir['LOo'] / 'extract' / 'tef2' / ('sections_%s' % args.gctag)


def face_xy(sn):
    s = pickle.load(open(sect_dir / (sn + '.p'), 'rb'))
    return s.x.values.astype(float), s.y.values.astype(float)


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
fig, ax = plt.subplots(figsize=(13, 6.4))

ax.pcolormesh(plon_g, plat_g, land, shading='flat', zorder=0, rasterized=True,
              cmap=ListedColormap([LAND_COLOR]))
# Depth range from the cells IN VIEW, not the whole grid: the domain max out in
# the main basin flattens Penn Cove's 7-21 m into a single pale tone.
inview = ((lon >= XL[0]) & (lon <= XL[1]) & (lat >= YL[0]) & (lat <= YL[1])
          & (mask == 1))
vmax = np.ceil(np.nanmax(h[inview]) / 10) * 10
print('  depth range in view: 0 to %.0f m' % vmax)
cd = ax.pcolormesh(plon_g, plat_g, hw, cmap=CMAP, shading='flat', zorder=1,
                   vmin=0, vmax=vmax, rasterized=True)

for sn in SECTS_DRAW:
    fx, fy = face_xy(sn)
    ax.plot(fx, fy, '-', color='w', lw=7.0, zorder=3, solid_capstyle='round')
    ax.plot(fx, fy, '-', color=TEXT_COLOR, lw=4.0, zorder=4,
            solid_capstyle='round')

# Discrete, because the count takes four values and a continuous ramp would
# invite reading a gradient into them. lajolla runs dark -> cream, so plain
# ascending puts the capped cells at the dark end: 10 levels is the
# unremarkable majority (122 of 165 cells) and wants to be quiet, while 7 is
# the rule actually doing something and wants to be seen. Warm ramp against
# the cool bathymetry, as on the pc4 map.
#
# The pc4 map's ms=19 markers cannot be used here: at 165 cells about 12 pt
# apart on this axes they would overlap into a solid blob. Same treatment at a
# size the grid can carry -- white edge for legibility over the bathymetry,
# then a thin dark ring.
lev = np.arange(cn.min(), cn.max() + 2) - 0.5
cmap = ListedColormap(cmc.lajolla(np.linspace(0.20, 0.90, len(lev) - 1)))
sc = ax.scatter(cx, cy, c=cn, s=80, cmap=cmap, norm=BoundaryNorm(lev, cmap.N),
                edgecolor='w', linewidth=1.2, zorder=5)
ax.scatter(cx, cy, s=80, facecolors='none', edgecolor=TEXT_COLOR,
           linewidth=0.5, zorder=6)

pfun.add_coast(ax, color=TEXT_COLOR, linewidth=0.8)
# set_xticks/set_yticks re-autoscale, and a rounded tick outside the grid then
# drags the view past the domain edge -- so pin the limits afterwards
AA = [XL[0], XL[1], YL[0], YL[1]]
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

# No title and no section names: this is a figure panel, captioned elsewhere as
# on 20260811_pc4_points_map.py. The colourbar IS annotated, though -- it is
# the only thing in the panel a reader cannot work out from the geometry, and
# an unlabelled discrete ramp of four browns and creams is unreadable without
# the caption in hand.
fig.tight_layout()

# Two colourbars in EXPLICIT axes rather than two fig.colorbar(ax=ax) calls.
# Passing ax= twice makes matplotlib steal space from the main axes twice and
# anchor each bar independently, which leaves them staggered in x as well as
# stacked in y. Placing them by hand off the main axes position -- same x, same
# width, one above the other -- is the only way to get them truly aligned.
# Release on top, because it is the result; depth below, because it is context.
pos = ax.get_position()
CW = 0.016                       # bar width in figure fraction
CX = pos.x1 + 0.015
CGAP = 0.10 * pos.height
CH = (pos.height - CGAP) / 2

cax_p = fig.add_axes([CX, pos.y0 + CH + CGAP, CW, CH])
cbp = fig.colorbar(sc, cax=cax_p, ticks=np.arange(cn.min(), cn.max() + 1))
cbp.set_label('Particles in cell', color=TEXT_COLOR)

cax_d = fig.add_axes([CX, pos.y0, CW, CH])
cbd = fig.colorbar(cd, cax=cax_d, extend='max')
# depth increases downward, so the bar reads the way the water column does
cbd.ax.invert_yaxis()
cbd.set_label('Depth [m]', color=TEXT_COLOR)

for cb in (cbp, cbd):
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
    cb.outline.set_edgecolor(TEXT_COLOR)

fn_out = out_dir / 'pcbot_release_map.png'
fig.savefig(fn_out, **SAVE_KW)
plt.close(fig)
print('\nwrote %s' % fn_out)
