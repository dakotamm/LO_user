"""
Map the hand-drawn wb1 domains (LO_output/section_lines/*.p) over the wb1
bathymetry.

Region-plot convention (DM, 2026.08.04):
  - the map window is the rectangular bounding box of the WHOLE region polygon,
    padded outward by PAD_CELLS grid cells so surrounding cells are visible --
    the polygon is never cropped;
  - only cells inside the `wb` polygon are drawn. Anything in the window but
    outside wb is masked out along with land.
So: rectangular extent, wb-clipped content.

Closed .p files are treated as regions, open ones as section lines.

Makes two figures in ~/Desktop/pltz:
  20260804_wb1_domains_overview.png  -- full wb1 grid, all polygons + lines
  20260804_wb1_domains_panels.png    -- one wb-clipped panel per region

Runs on the mac (only needs LO_data/grids/wb1/grid.nc), no ROMS output needed.

run 20260804_wb1_domains_map.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

Ldir = Lfun.Lstart(gridname='wb1')

sect_dir = Ldir['LOo'] / 'section_lines'
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)

# margin of grid cells to show around a region polygon
PAD_CELLS = 10

# master clip polygon -- every region plot shows only cells inside this
CLIP_POLY = 'wb'

# which regions get a zoom panel, and in what color (closed polys only)
REGION_COLORS = {
    'wb': 'tab:green',
    'pc': 'tab:red',
    'skagit_delta': 'tab:blue',
    'wb_north': 'tab:purple',
}

# ------------------------------------------------------------- section i/o ---
def load_sects():
    """All *.p in section_lines, split into closed (regions) and open (lines).

    Backups (*_backup_*) are skipped.
    """
    regions, lines = {}, {}
    for fn in sorted(sect_dir.glob('*.p')):
        name = fn.stem
        if 'backup' in name:
            continue
        df = pd.read_pickle(fn)
        closed = (abs(df.x.iloc[0] - df.x.iloc[-1]) < 1e-9 and
                  abs(df.y.iloc[0] - df.y.iloc[-1]) < 1e-9)
        (regions if closed else lines)[name] = df
    return regions, lines


def poly_mask(df, lon, lat):
    """Boolean array shaped like lon: True where the cell is inside the polygon."""
    pth = MplPath(np.column_stack([df.x.values, df.y.values]))
    pts = np.column_stack([lon.ravel(), lat.ravel()])
    return pth.contains_points(pts).reshape(lon.shape)


def region_aa(df, dx, dy, pad_cells=PAD_CELLS):
    """Rectangular window: full polygon bounds + a margin of grid cells."""
    return [df.x.min() - pad_cells * dx, df.x.max() + pad_cells * dx,
            df.y.min() - pad_cells * dy, df.y.max() + pad_cells * dy]


def plot_bathy(ax, plon, plat, h, aa, vmax=250):
    cs = ax.pcolormesh(plon, plat, h, cmap='Blues', vmin=0, vmax=vmax,
                       shading='flat', zorder=1)
    pfun.add_coast(ax, color='gray', linewidth=0.5)
    ax.axis(aa)
    pfun.dar(ax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(axis='x', labelrotation=45, labelsize=9)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    return cs


# ------------------------------------------------------------------- grid ---
dsg = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon = dsg.lon_rho.values
lat = dsg.lat_rho.values
mask_rho = dsg.mask_rho.values
h = dsg.h.values
dsg.close()

plon, plat = pfun.get_plon_plat(lon, lat)
dx = float(np.diff(lon[0, :]).mean())
dy = float(np.diff(lat[:, 0]).mean())

regions, lines = load_sects()
print('regions (closed): ' + ', '.join(regions))
print('lines   (open)  : ' + ', '.join(lines))

if CLIP_POLY not in regions:
    raise SystemExit('clip polygon %s.p is missing or not closed' % CLIP_POLY)

in_wb = poly_mask(regions[CLIP_POLY], lon, lat)
wet = mask_rho == 1

# full-grid bathy (land masked) and wb-clipped bathy (land + outside-wb masked)
h_land = np.ma.masked_where(~wet, h)
h_wb = np.ma.masked_where(~(wet & in_wb), h)

# river sources
riv = pd.read_csv(Ldir['grid'] / 'river_info.csv')
riv['lon'] = [lon[int(r), int(c)] for r, c in zip(riv.row_py, riv.col_py)]
riv['lat'] = [lat[int(r), int(c)] for r, c in zip(riv.row_py, riv.col_py)]

for name, df in regions.items():
    m = poly_mask(df, lon, lat) & wet
    print('%-14s %5d wet cells (%5d also inside %s)'
          % (name, int(m.sum()), int((m & in_wb).sum()), CLIP_POLY))


def draw_riv(ax, aa):
    sel = ((riv.lon > aa[0]) & (riv.lon < aa[1]) &
           (riv.lat > aa[2]) & (riv.lat < aa[3]))
    if not sel.any():
        return
    ax.plot(riv.lon[sel], riv.lat[sel], '*', color='yellow', markersize=15,
            markeredgecolor='k', markeredgewidth=0.6, zorder=25)
    for _, r in riv[sel].iterrows():
        ax.text(r.lon - 0.006, r.lat, r.rname, fontsize=8, ha='right',
                va='center', zorder=26)


# --------------------------------------------------------------- figure 1 ---
# overview: everything, on the full grid, unclipped so the wb outline is visible
# against the water it excludes
plt.close('all')
fig, ax = plt.subplots(figsize=(9, 11))

aa_full = [lon.min(), lon.max(), lat.min(), lat.max()]
cs = plot_bathy(ax, plon, plat, h_land, aa_full)
fig.colorbar(cs, ax=ax, shrink=0.6, label='depth h [m]')

# regions get a legend rather than inline labels -- the polygons overlap enough
# near the north end that inline text collides
for name, df in regions.items():
    c = REGION_COLORS.get(name, 'k')
    lw = 2.5 if name == CLIP_POLY else 2
    ax.plot(df.x, df.y, '-', color=c, lw=lw, zorder=20, label=name)

for name, df in lines.items():
    ax.plot(df.x, df.y, '-', color='magenta', lw=1.8, zorder=18)
    ax.text(df.x.mean(), df.y.mean(), '  ' + name, color='magenta', fontsize=8,
            ha='left', va='center', zorder=19)

ax.legend(loc='lower left', fontsize=9, framealpha=0.9, title='regions')

draw_riv(ax, aa_full)
ax.set_title('wb1 hand-drawn domains (2026.08.04)\n'
             'closed = regions, magenta = section lines')
fig.tight_layout()
fn_out = out_dir / '20260804_wb1_domains_overview.png'
fig.savefig(fn_out, dpi=200)
print('saved ' + str(fn_out))

# --------------------------------------------------------------- figure 2 ---
# one panel per region: rectangular window around the whole polygon, but only
# cells inside wb are drawn
names = [n for n in REGION_COLORS if n in regions]
fig2, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 8))

for ax, name in zip(np.atleast_1d(axes), names):
    df = regions[name]
    c = REGION_COLORS[name]
    aa = region_aa(df, dx, dy)

    # scale each panel to the depths actually in its window, otherwise the deep
    # Possession Sound trench flattens the shallow regions to near-white
    inwin = ((lon >= aa[0]) & (lon <= aa[1]) &
             (lat >= aa[2]) & (lat <= aa[3]) & wet & in_wb)
    vmax = float(np.percentile(h[inwin], 99)) if inwin.any() else 250

    cs = plot_bathy(ax, plon, plat, h_wb, aa, vmax=vmax)
    plt.colorbar(cs, ax=ax, shrink=0.5, label='depth h [m]')
    ax.plot(df.x, df.y, '-', color=c, lw=2.5, zorder=20)

    # section lines that fall in this window
    for lname, ldf in lines.items():
        sel = ((ldf.x > aa[0]) & (ldf.x < aa[1]) &
               (ldf.y > aa[2]) & (ldf.y < aa[3]))
        if sel.any():
            ax.plot(ldf.x, ldf.y, '-', color='magenta', lw=1.8, zorder=18)
            ax.text(ldf.x.mean(), ldf.y.mean(), '  ' + lname, color='magenta',
                    fontsize=8, ha='left', va='center', zorder=19)

    draw_riv(ax, aa)

    m = poly_mask(df, lon, lat) & wet & in_wb
    ax.set_title('%s\n%d wet cells (in %s)' % (name, int(m.sum()), CLIP_POLY),
                 color=c, fontsize=11)

fig2.tight_layout()
fn_out2 = out_dir / '20260804_wb1_domains_panels.png'
fig2.savefig(fn_out2, dpi=200)
print('saved ' + str(fn_out2))

plt.close('all')
