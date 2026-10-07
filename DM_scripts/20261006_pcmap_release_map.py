"""
Where the pcmap particles start: the whole Penn Cove release (tef2 segments
pc_cp_m + pc_cp_p + pc_lp_m, every wet cell landward of pc_lp, particles at
sigma cell centres about 2 m apart in the vertical; ic_from_tef2_segs(DZ=2) in
LO_user/tracker2/experiments.py). Same style as 20260811_pcbot_release_map.py.

The positions are read from the run, not re-derived from the experiment code:
the starting cell (j0, i0) of every particle in the 20261005_pcmap_reduce.py
files, which come from time index 0 of the track files. Every release must seed
the same cells with the same counts, and the script checks that across all the
reduced files it finds rather than assuming it.

WHAT THE COLOUR CARRIES
Each cell is coloured by how many particles it holds. With levels about 2 m
apart that is max(1, round(h / 2)), so the count is a map of local depth in
particle units: the deep channel and mouth carry the most particles, the head
and the shoals the fewest. The release is volume-uniform, which is why the
deep outer cove holds more of the cohort than its area alone would suggest.

The vertical structure of the release is NOT shown here (plan view).

Sections drawn: pc_cp, the inner/outer boundary of the analysis quadrants, and
pc_lp, the seaward edge of the release.

QUADRANTS are drawn as used in the analysis (pcmap_regions.py): inner = tef2
segment pc_cp_m (landward of pc_cp), outer = the rest of the cove; north /
south split by the pc_ew line drawn along the cove (dashed), and each quadrant
is labelled. A second figure,
pcmap_release_map_quadrants.png, colours the dots by quadrant instead of by
particle count. The extent is fitted to the release
plus pc_cp, pc_lj and pc_lp, as on the pcbot map.

run 20261006_pcmap_release_map.py
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
p.add_argument('-glob', default='pcmap_3d*', help='reduced files to read the release from')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
red_dir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_reduce' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261006_pcmap_release_map'
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
SECTS_DRAW = ['pc_cp', 'pc_lp']

# --------------------------------------------------------------- release ---
fns = sorted(red_dir.glob(args.glob + '.p'))
if len(fns) == 0:
    sys.exit('no reduced files in %s' % red_dir)
ref = None
n_same = 0
for fn in fns:
    P = pickle.load(open(fn, 'rb'))['P']
    cnt = P.groupby(['j0', 'i0']).size()
    if ref is None:
        ref, P0, ref_name = cnt, P, fn.name
    elif cnt.equals(ref):
        n_same += 1
    else:
        print('  WARNING: %s seeds different cells or counts' % fn.name)
print('release from %s: %d particles, %d cells; identical in %d of %d other files'
      % (ref_name, int(ref.sum()), len(ref), n_same, len(fns) - 1))
print('  height above bed %.2f to %.2f m, local depth %.1f to %.1f m'
      % (P0.hab0.min(), P0.hab0.max(), P0.h0.min(), P0.h0.max()))

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
NR, NC = lon.shape

# quadrants, as in 20261005_pcmap_reduce.py
seg = pickle.load(open(sorted((Ldir['LOo'] / 'extract' / 'tef2').glob(
    'seg_info_dict_%s_*.p' % args.gctag))[0], 'rb'))


def seg_mask(names):
    m = np.zeros((NR, NC), dtype=bool)
    for sname in names:
        a = np.array(seg[sname]['ji_list'])
        m[a[:, 0], a[:, 1]] = True
    return m


# cove, inner/outer and the pc_ew north/south split: the shared definition
from pcmap_regions import regions
REG = regions(Ldir, lon, lat)
cove, inner, north = REG['cove'], REG['inner'], REG['north']
QUAD = REG['QUAD'].copy()
ns_x, ns_y = REG['line']                  # the pc_ew N/S line
QNAMES = ['inner-N', 'inner-S', 'outer-N', 'outer-S']
QCOL = ['#e8455e', '#f0a04b', '#4565e8', '#45b0a8']

# Section lines shore to shore, as on the pc4 map -- the specified line, not the
# staircase of tef2 faces it snapped to.
sect_dir = Ldir['LOo'] / 'extract' / 'tef2' / ('sections_%s' % args.gctag)


def face_xy(sn):
    s = pickle.load(open(sect_dir / (sn + '.p'), 'rb'))
    return s.x.values.astype(float), s.y.values.astype(float)


# particles per cell, at the rho points of the starting cells
jj0 = ref.index.get_level_values(0).values
ii0 = ref.index.get_level_values(1).values
cx = lon[jj0, ii0]
cy = lat[jj0, ii0]
cn = ref.values
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
fig, ax = plt.subplots(figsize=(17, 8.4))

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

# Discrete, one colour per count, warm ramp against the cool bathymetry as on
# the pcbot map. Same marker treatment: white edge for legibility over the
# bathymetry, then a thin dark ring.
lev = np.arange(cn.min(), cn.max() + 2) - 0.5
cmap = ListedColormap(cmc.lajolla(np.linspace(0.20, 0.90, len(lev) - 1)))
sc = ax.scatter(cx, cy, c=cn, s=130, cmap=cmap, norm=BoundaryNorm(lev, cmap.N),
                edgecolor='w', linewidth=1.2, zorder=5)
ax.scatter(cx, cy, s=130, facecolors='none', edgecolor=TEXT_COLOR,
           linewidth=0.5, zorder=6)

pfun.add_coast(ax, color=TEXT_COLOR, linewidth=0.8)


def draw_quadrants(ax):
    """pc_ew N/S line (dashed, white-cased) and a label in each quadrant."""
    ax.plot(ns_x, ns_y, '-', color='w', lw=5.0, zorder=3.5, solid_joinstyle='miter')
    ax.plot(ns_x, ns_y, '--', color=TEXT_COLOR, lw=2.5, zorder=4.5, dashes=(4, 2))
    for k, qn in enumerate(QNAMES):
        jq, iq = np.where(QUAD == k)
        # put the label at the quadrant's median cell, nudged to the outside edge
        xq = np.median(lon[jq, iq])
        yq = np.percentile(lat[jq, iq], 88 if qn.endswith('N') else 12)
        ax.text(xq, yq, qn, ha='center', va='center', fontsize=17, fontweight='bold',
                color=TEXT_COLOR, zorder=8,
                bbox=dict(boxstyle='round,pad=0.25', fc='w', ec=TEXT_COLOR, lw=1.0, alpha=0.9))


draw_quadrants(ax)
# set_xticks/set_yticks re-autoscale, and a rounded tick outside the grid then
# drags the view past the domain edge -- so pin the limits afterwards
AA = [XL[0], XL[1], YL[0], YL[1]]
# evenly spaced round ticks inside the window (linspace + round gives uneven gaps)
ax.set_xticks(np.arange(np.ceil(AA[0] / 0.02) * 0.02, AA[1], 0.02).round(2))
ax.set_yticks(np.arange(np.ceil(AA[2] / 0.01) * 0.01, AA[3], 0.01).round(2))
ax.axis(AA)
ax.set_autoscale_on(False)
pfun.dar(ax)
ax.tick_params(length=6, labelrotation=0)
ax.set_xlabel('Longitude [$^{\\circ}$E]')
ax.set_ylabel('Latitude [$^{\\circ}$N]')
for s in ax.spines.values():
    s.set_visible(True)

# No title and no section names: a figure panel, captioned elsewhere, as on
# the pcbot map. The colourbars are annotated.
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

fn_out = out_dir / 'pcmap_release_map.png'
fig.savefig(fn_out, **SAVE_KW)
plt.close(fig)
print('\nwrote %s' % fn_out)

# ------------------------------------------- second figure: by quadrant ---
fig, ax = plt.subplots(figsize=(17, 8.4))
ax.pcolormesh(plon_g, plat_g, land, shading='flat', zorder=0, rasterized=True,
              cmap=ListedColormap([LAND_COLOR]))
cd = ax.pcolormesh(plon_g, plat_g, hw, cmap=CMAP, shading='flat', zorder=1,
                   vmin=0, vmax=vmax, rasterized=True)
for sn in SECTS_DRAW:
    fx, fy = face_xy(sn)
    ax.plot(fx, fy, '-', color='w', lw=7.0, zorder=3, solid_capstyle='round')
    ax.plot(fx, fy, '-', color=TEXT_COLOR, lw=4.0, zorder=4, solid_capstyle='round')
cq = QUAD[jj0, ii0]
for k in range(4):
    m = cq == k
    ax.scatter(cx[m], cy[m], s=130, color=QCOL[k], edgecolor='w', linewidth=1.2, zorder=5)
ax.scatter(cx, cy, s=130, facecolors='none', edgecolor=TEXT_COLOR, linewidth=0.5, zorder=6)
pfun.add_coast(ax, color=TEXT_COLOR, linewidth=0.8)
draw_quadrants(ax)
ax.set_xticks(np.arange(np.ceil(AA[0] / 0.02) * 0.02, AA[1], 0.02).round(2))
ax.set_yticks(np.arange(np.ceil(AA[2] / 0.01) * 0.01, AA[3], 0.01).round(2))
ax.axis(AA)
ax.set_autoscale_on(False)
pfun.dar(ax)
ax.tick_params(length=6, labelrotation=0)
ax.set_xlabel('Longitude [$^{\\circ}$E]')
ax.set_ylabel('Latitude [$^{\\circ}$N]')
fig.tight_layout()
pos = ax.get_position()
cax_d = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.016, pos.height])
cbd = fig.colorbar(cd, cax=cax_d, extend='max')
cbd.ax.invert_yaxis()
cbd.set_label('Depth [m]', color=TEXT_COLOR)
cbd.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
cbd.outline.set_edgecolor(TEXT_COLOR)
fn_out = out_dir / 'pcmap_release_map_quadrants.png'
fig.savefig(fn_out, **SAVE_KW)
plt.close(fig)
print('wrote %s' % fn_out)
print('  particles per release by quadrant: %s'
      % ', '.join('%s %d' % (QNAMES[k], int(cn[cq == k].sum())) for k in range(4)))
