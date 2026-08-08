"""
Presentation-quality bathymetry maps of the cas7 and wb1 grids.

Fig 1  cas7 full domain, log-scaled bathymetry, with the wb1 grid extent drawn
       as a box; an inset zooms on the Salish Sea so the box is readable.
Fig 2  wb1 domain, linear bathymetry.

Styled for slides: transparent background, large fonts, and no text on the map
-- titles, cell counts and callouts belong on the slide, not in the figure.
TEXT_COLOR flips the whole figure between light-slide (dark text) and
dark-slide (light text) use.

Only needs LO_data/grids/{cas7,wb1}/grid.nc -- runs on the mac.

run 20260807_grid_bathy_ppt.py
"""
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import cmocean
from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)

# 'k' for light slides, 'w' for dark slides -- everything else follows
TEXT_COLOR = 'k'
CMAP = cmocean.cm.deep
BOX_COLOR = '#e04256'
# land is filled rather than left transparent -- a see-through landmask reads as
# whatever the slide background is, which kills the coastline
LAND_COLOR = '#e8e4dc'

mpl.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
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


# ------------------------------------------------------------------ helpers ---
def load_grid(gridname):
    """lon/lat rho, psi-corner lon/lat, h with land masked, and mean dx/dy [m]."""
    fn = Lfun.Lstart(gridname=gridname)['grid'] / 'grid.nc'
    ds = xr.open_dataset(fn)
    lon = ds.lon_rho.values
    lat = ds.lat_rho.values
    h = ds.h.values
    mask = ds.mask_rho.values
    # pm/pn are 1/dx and 1/dy in m^-1
    res = (float((1 / ds.pm.values).mean()), float((1 / ds.pn.values).mean()))
    ds.close()
    plon, plat = pfun.get_plon_plat(lon, lat)
    return (lon, lat, plon, plat, np.ma.masked_where(mask != 1, h),
            np.ma.masked_where(mask == 1, mask), res)


def draw_land(ax, plon, plat, land):
    """Flat fill of the land cells, under the bathymetry."""
    ax.pcolormesh(plon, plat, land, shading='flat', zorder=0, rasterized=True,
                  cmap=mpl.colors.ListedColormap([LAND_COLOR]))


def style_map(ax, aa, nx=4, ny=5):
    """Coastline, aspect, tick tidy-up. aa = [lon0, lon1, lat0, lat1]."""
    pfun.add_coast(ax, color=TEXT_COLOR, linewidth=0.8)
    # set_xticks/set_yticks re-autoscale, and a rounded tick outside the grid
    # then drags the view past the domain edge -- so pin the limits afterwards
    ax.set_xticks(np.linspace(aa[0], aa[1], nx).round(1))
    ax.set_yticks(np.linspace(aa[2], aa[3], ny).round(1))
    ax.axis(aa)
    ax.set_autoscale_on(False)
    pfun.dar(ax)
    ax.tick_params(length=6, labelrotation=0)
    ax.set_xlabel('Longitude [$^{\\circ}$E]')
    ax.set_ylabel('Latitude [$^{\\circ}$N]')
    for s in ax.spines.values():
        s.set_visible(True)


def style_cbar(cb, label):
    # depth increases downward, so the bar reads the way the water column does
    cb.ax.invert_yaxis()
    cb.set_label(label, color=TEXT_COLOR)
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
    cb.outline.set_edgecolor(TEXT_COLOR)


# -------------------------------------------------------------------- grids ---
lon7, lat7, plon7, plat7, h7, land7, res7 = load_grid('cas7')
lonw, latw, plonw, platw, hw, landw, resw = load_grid('wb1')

# wb1 extent, in psi corners so the box is the true outer edge of the grid
wb_aa = [plonw.min(), plonw.max(), platw.min(), platw.max()]
print('wb1 extent: lon %.3f to %.3f, lat %.3f to %.3f' % tuple(wb_aa))
# printed, not plotted -- these are the numbers to type onto the slide
print('cas7 %d x %d cells, %d x %d m mean resolution'
      % (h7.shape + tuple(round(r) for r in res7)))
print('wb1  %d x %d cells, %d x %d m mean resolution'
      % (hw.shape + tuple(round(r) for r in resw)))


def wb_box(ax, lw=3, label=None):
    r = Rectangle((wb_aa[0], wb_aa[2]), wb_aa[1] - wb_aa[0], wb_aa[3] - wb_aa[2],
                  fill=False, edgecolor=BOX_COLOR, lw=lw, zorder=30, label=label)
    ax.add_patch(r)
    return r


# ---------------------------------------------------------------- figure 1 ---
plt.close('all')
fig = plt.figure(figsize=(11, 13))
ax = fig.add_subplot(111)

aa7 = [plon7.min(), plon7.max(), plat7.min(), plat7.max()]
norm7 = LogNorm(vmin=10, vmax=4000)
draw_land(ax, plon7, plat7, land7)
cs = ax.pcolormesh(plon7, plat7, h7, cmap=CMAP, shading='flat', norm=norm7,
                   zorder=1, rasterized=True)
style_map(ax, aa7, nx=4, ny=6)

cb = fig.colorbar(cs, ax=ax, shrink=0.45, pad=0.02, extend='both')
style_cbar(cb, 'Depth [m]')

wb_box(ax, lw=3)

# inset: Salish Sea, where the wb1 box is actually legible. Parked over the
# Oregon corner so it does not cover the region it is zooming.
axi = ax.inset_axes([0.50, 0.09, 0.48, 0.34])
aa_ss = [-123.4, -122.0, 47.0, 48.7]
draw_land(axi, plon7, plat7, land7)
axi.pcolormesh(plon7, plat7, h7, cmap=CMAP, shading='flat', norm=norm7,
               zorder=1, rasterized=True)
pfun.add_coast(axi, color=TEXT_COLOR, linewidth=0.6)
axi.axis(aa_ss)
pfun.dar(axi)
axi.set_xticks([])
axi.set_yticks([])
axi.set_facecolor('none')
for s in axi.spines.values():
    s.set_edgecolor(TEXT_COLOR)
    s.set_linewidth(1.5)
wb_box(axi, lw=3)
mark_inset(ax, axi, loc1=2, loc2=1, fc='none', ec=TEXT_COLOR, lw=1.2, alpha=0.5)

fn1 = out_dir / '20260807_cas7_bathy_ppt.png'
fig.savefig(fn1, **SAVE_KW)
print('saved ' + str(fn1))


# ---------------------------------------------------------------- figure 2 ---
fig2 = plt.figure(figsize=(9, 12))
ax2 = fig2.add_subplot(111)

draw_land(ax2, plonw, platw, landw)
cs2 = ax2.pcolormesh(plonw, platw, hw, cmap=CMAP, shading='flat',
                     vmin=0, vmax=np.ceil(hw.max() / 10) * 10, zorder=1,
                     rasterized=True)
style_map(ax2, wb_aa, nx=4, ny=6)

# the frame IS the wb1 extent, so make it the same red box drawn on the cas7 map
for s in ax2.spines.values():
    s.set_edgecolor(BOX_COLOR)
    s.set_linewidth(3)

cb2 = fig2.colorbar(cs2, ax=ax2, shrink=0.6, pad=0.02, extend='max')
style_cbar(cb2, 'Depth [m]')

fn2 = out_dir / '20260807_wb1_bathy_ppt.png'
fig2.savefig(fn2, **SAVE_KW)
print('saved ' + str(fn2))

plt.close('all')
