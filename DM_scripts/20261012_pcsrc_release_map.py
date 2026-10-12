"""
Where the pcsrc_spu particles start: upper Saratoga only (tef2 segment pc_lp_p,
between pc_lp, sp_mid and skagit_sp), every 2nd cell in i and j, levels about
4 m apart at sigma cell centres (pcsrc_regions.SEED['spu'];
ic_graded_volume() in LO_user/tracker2/experiments.py). Same style as
20261006_pcmap_release_map.py.

The positions come from experiments.get_ic(), the call tracker.py makes, run on
grid.nc -- wb1_t0 bathymetry matches grid.nc exactly, so this is the release
the runs will get. Pass -fn00 to use a history file instead.

WHAT THE COLOUR CARRIES
Each seeded column is coloured by how many particles it holds,
max(1, round(h / 4)), so the count is a map of local depth in particle units.
The release is volume-uniform within the region.

The Penn Cove cells (small grey dots) are NOT seeded here: cove water comes
from the pcmap run at the same release time.

Sections drawn: pc_lp, sp_mid and skagit_sp, the edges of the release.

run 20261012_pcsrc_release_map.py
"""
import argparse
import pickle
import sys

import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cmcrameri import cm as cmc
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

p = argparse.ArgumentParser()
p.add_argument('-gctag', default='wb1_pc1')
p.add_argument('-exp', default='pcsrc_spu')
p.add_argument('-fn00', default='', help='file to seed from (default: the grid.nc)')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
out_dir = Ldir['LOo'] / 'DM_outs' / '20261012_pcsrc_release_map'
Lfun.make_dir(out_dir)
fn00 = args.fn00 if args.fn00 else str(Ldir['grid'] / 'grid.nc')

# ---- styling, as on 20261006_pcmap_release_map.py -------------------------
TEXT_COLOR = 'k'
CMAP = cmocean.cm.deep
LAND_COLOR = '#e8e4dc'
COVE_COLOR = '#8c8a85'
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
SECTS = ['pc_lp', 'sp_mid', 'skagit_sp']

# --------------------------------------------------------------- release ---
sys.path.append(str(Ldir['LOu'] / 'tracker2'))
import experiments
import pcsrc_regions as pr

plon0, plat0, pcs0 = experiments.get_ic(dict(exp_name=args.exp, gridname='wb1', fn00=fn00))

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

# particles per seeded column (they sit exactly on rho points)
cnt = pd.DataFrame(dict(x=plon0, y=plat0)).groupby(['x', 'y']).size()
cx = cnt.index.get_level_values(0).values
cy = cnt.index.get_level_values(1).values
cn = cnt.values
R = pr.regions(Ldir, lon, lat, mask == 1)
print('%s: %d particles in %d columns (%d to %d per column, median %d); '
      'region %d cells, h %.1f to %.1f m'
      % (args.exp, len(plon0), len(cn), cn.min(), cn.max(), np.median(cn),
         R['spu'].sum(), h[R['spu']].min(), h[R['spu']].max()))
covex, covey = lon[R['cove']], lat[R['cove']]

sect_dir = Ldir['LOo'] / 'extract' / 'tef2' / ('sections_%s' % args.gctag)


def face_xy(sn):
    s = pickle.load(open(sect_dir / (sn + '.p'), 'rb'))
    return s.x.values.astype(float), s.y.values.astype(float)


# ---------------------------------------------------------------- extent ---
# fitted to the release, the cove and the three bounding sections
xs = [cx.min(), cx.max(), covex.min()]
ys = [cy.min(), cy.max()]
for sn in SECTS:
    fx, fy = face_xy(sn)
    xs += [fx.min(), fx.max()]
    ys += [fy.min(), fy.max()]
PAD = 4
XL = (min(xs) - PAD * dlon, max(xs) + PAD * dlon)
YL = (min(ys) - PAD * dlat, max(ys) + PAD * dlat)

# ---------------------------------------------------------------- figure ---
fig, ax = plt.subplots(figsize=(15, 11))
ax.pcolormesh(plon_g, plat_g, land, shading='flat', zorder=0, rasterized=True,
              cmap=ListedColormap([LAND_COLOR]))
inview = ((lon >= XL[0]) & (lon <= XL[1]) & (lat >= YL[0]) & (lat <= YL[1])
          & (mask == 1))
vmax = np.ceil(np.nanmax(h[inview]) / 10) * 10
print('  depth range in view: 0 to %.0f m' % vmax)
cd = ax.pcolormesh(plon_g, plat_g, hw, cmap=CMAP, shading='flat', zorder=1,
                   vmin=0, vmax=vmax, rasterized=True)

for sn in SECTS:
    fx, fy = face_xy(sn)
    ax.plot(fx, fy, '-', color='w', lw=7.0, zorder=3, solid_capstyle='round')
    ax.plot(fx, fy, '-', color=TEXT_COLOR, lw=4.0, zorder=4, solid_capstyle='round')

# cove cells: not seeded here, the pcmap twin supplies them
ax.scatter(covex, covey, s=6, color=COVE_COLOR, edgecolor='none', zorder=4.5)

# discrete counts, warm ramp against the cool bathymetry, as on the pcmap map
step = max(1, int(np.ceil((cn.max() - cn.min() + 1) / 12)))
lev = np.arange(cn.min(), cn.max() + step + 1, step) - 0.5
cmap = ListedColormap(cmc.lajolla(np.linspace(0.20, 0.90, len(lev) - 1)))
sc = ax.scatter(cx, cy, c=cn, s=46, cmap=cmap, norm=BoundaryNorm(lev, cmap.N),
                edgecolor='w', linewidth=0.8, zorder=5)
ax.scatter(cx, cy, s=46, facecolors='none', edgecolor=TEXT_COLOR,
           linewidth=0.35, zorder=6)
pfun.add_coast(ax, color=TEXT_COLOR, linewidth=0.8)

ax.legend(handles=[Line2D([], [], marker='o', ls='none', ms=9, mfc=cmap(0.6), mec='w',
                          label='seeded column (%d particles)' % len(plon0)),
                   Line2D([], [], marker='o', ls='none', ms=4, mfc=COVE_COLOR, mec='none',
                          label='Penn Cove: from the pcmap twin')],
          loc='lower left', fontsize=15, framealpha=0.95)

AA = [XL[0], XL[1], YL[0], YL[1]]
ax.set_xticks(np.arange(np.ceil(AA[0] / 0.05) * 0.05, AA[1], 0.05).round(2))
ax.set_yticks(np.arange(np.ceil(AA[2] / 0.02) * 0.02, AA[3], 0.02).round(2))
ax.axis(AA)
ax.set_autoscale_on(False)
pfun.dar(ax)
ax.tick_params(length=6, labelrotation=0)
ax.set_xlabel('Longitude [$^{\\circ}$E]')
ax.set_ylabel('Latitude [$^{\\circ}$N]')
for s in ax.spines.values():
    s.set_visible(True)
fig.tight_layout()

pos = ax.get_position()
CW = 0.016
CX = pos.x1 + 0.015
CGAP = 0.10 * pos.height
CH = (pos.height - CGAP) / 2
cax_p = fig.add_axes([CX, pos.y0 + CH + CGAP, CW, CH])
cbp = fig.colorbar(sc, cax=cax_p, ticks=np.arange(cn.min(), cn.max() + 1, step))
cbp.set_label('Particles in column', color=TEXT_COLOR)
cax_d = fig.add_axes([CX, pos.y0, CW, CH])
cbd = fig.colorbar(cd, cax=cax_d, extend='max')
cbd.ax.invert_yaxis()
cbd.set_label('Depth [m]', color=TEXT_COLOR)
for cb in (cbp, cbd):
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
    cb.outline.set_edgecolor(TEXT_COLOR)

fn_out = out_dir / ('%s_release_map.png' % args.exp)
fig.savefig(fn_out, **SAVE_KW)
plt.close(fig)
print('saved %s' % fn_out)
