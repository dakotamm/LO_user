"""
Proposed north / south split of the Penn Cove quadrants. Two methods:

  -method line (default) a line drawn in LO_output/section_lines (-line,
         default pc_ew, the east-west line along the cove from the head to
         pc_lp). A cell is north if its centre lies north of the line at that
         cell's longitude (the line is linearly interpolated in longitude).
  -method area in each half of the cove separately (inner = tef2 segment
         pc_cp_m, outer = pc_cp_p + pc_lp_m), a straight line along that half's
         long axis that divides the half's WATER AREA in two (below).

How the line is built, per half:
  1. cell centres in metres (local tangent plane), each weighted by its area
     1 / (pm pn) from the grid
  2. the long axis is the principal axis of that area-weighted cloud of cells
  3. the line runs along that axis, shifted across it to the area-weighted
     median, so half the area lies on each side (a cell goes to the side its
     centre falls on, so the split is 50/50 to within one row of cells)

Drawn in the style of 20261006_pcmap_release_map.py: dots coloured by the
PROPOSED quadrant, the two proposed lines solid, the current per-column
staircase (20261005_pcmap_reduce.py) thin and dashed for comparison, and cells
that would change quadrant ringed. The new assignment is also saved
(pcmap_quadrant_split.p: QUAD array on the rho grid plus the line endpoints)
so the analysis can switch to it without recomputing.

Output: LO_output/DM_outs/20261006_pcmap_quadrant_split_map/
  pcmap_quadrant_split_map_<line or area>.png, pcmap_quadrant_split_<line or area>.p

run 20261006_pcmap_quadrant_split_map.py
run 20261006_pcmap_quadrant_split_map.py -method area
"""
import argparse
import pickle

import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

p = argparse.ArgumentParser()
p.add_argument('-gctag', default='wb1_pc1')
p.add_argument('-method', default='line', choices=['line', 'area'])
p.add_argument('-line', default='pc_ew', help='section_lines file for -method line')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
out_dir = Ldir['LOo'] / 'DM_outs' / '20261006_pcmap_quadrant_split_map'
Lfun.make_dir(out_dir)

TEXT_COLOR = 'k'
CMAP = cmocean.cm.deep
LAND_COLOR = '#e8e4dc'
mpl.rcParams.update({
    'font.size': 18, 'axes.labelsize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
    'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'text.color': TEXT_COLOR, 'axes.labelcolor': TEXT_COLOR, 'axes.edgecolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR, 'ytick.color': TEXT_COLOR,
    'savefig.transparent': True, 'figure.facecolor': 'none', 'axes.facecolor': 'none',
})
SAVE_KW = dict(dpi=300, bbox_inches='tight', transparent=True, facecolor='none')
QNAMES = ['inner-N', 'inner-S', 'outer-N', 'outer-S']
QCOL = ['#e8455e', '#f0a04b', '#4565e8', '#45b0a8']

# ------------------------------------------------------------------ grid ---
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat, h, mask = g.lon_rho.values, g.lat_rho.values, g.h.values, g.mask_rho.values
area = 1 / (g.pm.values * g.pn.values)
g.close()
NR, NC = lon.shape
dlon = float(np.diff(lon[0, :]).mean()); dlat = float(np.diff(lat[:, 0]).mean())
plon_g, plat_g = pfun.get_plon_plat(lon, lat)
seg = pickle.load(open(sorted((Ldir['LOo'] / 'extract' / 'tef2').glob(
    'seg_info_dict_%s_*.p' % args.gctag))[0], 'rb'))


def seg_mask(names):
    m = np.zeros((NR, NC), dtype=bool)
    for s in names:
        a = np.array(seg[s]['ji_list'])
        m[a[:, 0], a[:, 1]] = True
    return m


cove = seg_mask(['pc_cp_m', 'pc_cp_p', 'pc_lp_m'])
inner = seg_mask(['pc_cp_m'])
outer = cove & ~inner

# current split (per-column mean j), for comparison
jjc, iic = np.where(cove)
north_old = np.zeros((NR, NC), dtype=bool)
old_x, old_y = [], []
for i in np.unique(iic):
    jcol = jjc[iic == i]
    north_old[jcol[jcol > jcol.mean()], i] = True
    yb = lat[0, 0] + (jcol[jcol <= jcol.mean()].max() + 0.5) * dlat
    old_x += [lon[0, 0] + (i - 0.5) * dlon, lon[0, 0] + (i + 0.5) * dlon]
    old_y += [yb, yb]

# ------------------------------------------ equal-area long-axis lines ---
lat0 = lat[cove].mean(); lon0 = lon[cove].mean()
mx = 111320 * np.cos(np.deg2rad(lat0)); my = 110540
X = (lon - lon0) * mx; Y = (lat - lat0) * my
north_new = np.zeros((NR, NC), dtype=bool)
LINES = {}
if args.method == 'line':
    L = pd.read_pickle(Ldir['LOo'] / 'section_lines' / (args.line + '.p'))
    o = np.argsort(L.x.values)
    lxl, lyl = L.x.values[o].astype(float), L.y.values[o].astype(float)
    if lon[cove].min() < lxl[0] or lon[cove].max() > lxl[-1]:
        print('  note: %s spans lon %.4f..%.4f, cove cells %.4f..%.4f; extended flat beyond its ends'
              % (args.line, lxl[0], lxl[-1], lon[cove].min(), lon[cove].max()))
    yline = np.interp(lon, lxl, lyl)
    north_new = cove & (lat > yline)
    LINES[args.line] = (lxl, lyl)
    for name, m in [('inner', inner), ('outer', outer)]:
        is_n = north_new[m]; w = area[m]
        print('%s: %d cells, north side %.1f%% of area (%d cells), reassigned vs current %d (S->N %d, N->S %d)'
              % (name, m.sum(), 100 * w[is_n].sum() / w.sum(), is_n.sum(),
                 (north_new[m] != north_old[m]).sum(), (north_new[m] & ~north_old[m]).sum(),
                 (~north_new[m] & north_old[m]).sum()))
for name, m in ([('inner', inner), ('outer', outer)] if args.method == 'area' else []):
    x, y, w = X[m], Y[m], area[m]
    xc, yc = np.average(x, weights=w), np.average(y, weights=w)
    C = np.cov(np.vstack([x - xc, y - yc]), aweights=w)
    evals, evecs = np.linalg.eigh(C)
    ax_ = evecs[:, np.argmax(evals)]                 # long axis
    if ax_[0] < 0:
        ax_ = -ax_                                   # point it east
    nrm = np.array([-ax_[1], ax_[0]])                # left of east-pointing = north side
    s_n = (x - xc) * nrm[0] + (y - yc) * nrm[1]
    o = np.argsort(s_n); cw = np.cumsum(w[o]) / w.sum()
    off = s_n[o][np.searchsorted(cw, 0.5)]           # area-weighted median across the axis
    is_n = s_n > off
    jj_, ii_ = np.where(m)
    north_new[jj_[is_n], ii_[is_n]] = True
    # draw the line only across this half: from the west edge of its westernmost
    # cells to the east edge of its easternmost cells (pc_cp / pc_lp bound them)
    xw = (lon[m].min() - dlon / 2 - lon0) * mx; xe = (lon[m].max() + dlon / 2 - lon0) * mx
    t = (np.array([xw, xe]) - (xc + off * nrm[0])) / ax_[0]
    lx = xc + t * ax_[0] + off * nrm[0]; ly = yc + t * ax_[1] + off * nrm[1]
    LINES[name] = (lon0 + lx / mx, lat0 + ly / my)
    print('%s: %d cells, long axis %.0f deg from east, north side %.1f%% of area (%d cells), '
          'reassigned vs current %d (S->N %d, N->S %d)'
          % (name, m.sum(), np.degrees(np.arctan2(ax_[1], ax_[0])),
             100 * w[is_n].sum() / w.sum(), is_n.sum(),
             (north_new[m] != north_old[m]).sum(), (north_new[m] & ~north_old[m]).sum(),
             (~north_new[m] & north_old[m]).sum()))

QUAD = np.full((NR, NC), -1, dtype=int)
QUAD[cove] = (2 * (~inner) + (~north_new))[cove]
QUAD_OLD = np.full((NR, NC), -1, dtype=int)
QUAD_OLD[cove] = (2 * (~inner) + (~north_old))[cove]
changed = cove & (QUAD != QUAD_OLD)
for k, qn in enumerate(QNAMES):
    print('  %-8s %3d cells (was %3d), %.2f km2'
          % (qn, (QUAD == k).sum(), (QUAD_OLD == k).sum(), area[QUAD == k].sum() / 1e6))
tag = args.line if args.method == 'line' else 'area'
pickle.dump(dict(QUAD=QUAD, lines=LINES,
                 method=('north of section line %s' % args.line) if args.method == 'line'
                 else 'equal-area long-axis line per half'),
            open(out_dir / ('pcmap_quadrant_split_%s.p' % tag), 'wb'))

# ---------------------------------------------------------------- figure ---
sect_dir = Ldir['LOo'] / 'extract' / 'tef2' / ('sections_%s' % args.gctag)


def face_xy(sn):
    s = pickle.load(open(sect_dir / (sn + '.p'), 'rb'))
    return s.x.values.astype(float), s.y.values.astype(float)


xs, ys = [lon[cove].min(), lon[cove].max()], [lat[cove].min(), lat[cove].max()]
for sn in ['pc_cp', 'pc_lj', 'pc_lp']:
    fx, fy = face_xy(sn); xs += [fx.min(), fx.max()]; ys += [fy.min(), fy.max()]
AA = [min(xs) - 4 * dlon, max(xs) + 4 * dlon, min(ys) - 4 * dlat, max(ys) + 4 * dlat]
inview = (lon >= AA[0]) & (lon <= AA[1]) & (lat >= AA[2]) & (lat <= AA[3]) & (mask == 1)
vmax = np.ceil(np.nanmax(h[inview]) / 10) * 10

fig, ax = plt.subplots(figsize=(17, 8.4))
ax.pcolormesh(plon_g, plat_g, np.ma.masked_where(mask == 1, mask), shading='flat', zorder=0,
              rasterized=True, cmap=ListedColormap([LAND_COLOR]))
cd = ax.pcolormesh(plon_g, plat_g, np.ma.masked_where(mask != 1, h), cmap=CMAP, shading='flat',
                   zorder=1, vmin=0, vmax=vmax, rasterized=True)
for sn in ['pc_cp', 'pc_lp']:
    fx, fy = face_xy(sn)
    ax.plot(fx, fy, '-', color='w', lw=7.0, zorder=3, solid_capstyle='round')
    ax.plot(fx, fy, '-', color=TEXT_COLOR, lw=4.0, zorder=4, solid_capstyle='round')
ax.plot(old_x, old_y, '--', color='0.15', lw=1.8, zorder=6.5, dashes=(3, 2), label='current split')
for name, (lx, ly) in LINES.items():
    ax.plot(lx, ly, '-', color='w', lw=6.0, zorder=4.4, solid_capstyle='round')
    ax.plot(lx, ly, '-', color=TEXT_COLOR, lw=3.0, zorder=4.5, solid_capstyle='round',
            label=('proposed split (%s)' % args.line if args.method == 'line' else 'proposed split (equal area)')
            if name == list(LINES)[0] else None)
jq, iq = np.where(cove)
for k in range(4):
    m = QUAD[jq, iq] == k
    ax.scatter(lon[jq, iq][m], lat[jq, iq][m], s=130, color=QCOL[k], edgecolor='w', linewidth=1.2,
               zorder=5, label=QNAMES[k])
ax.scatter(lon[changed], lat[changed], s=330, facecolors='none', edgecolor=TEXT_COLOR, linewidth=2.2,
           zorder=6, label='changes quadrant (%d)' % changed.sum())
pfun.add_coast(ax, color=TEXT_COLOR, linewidth=0.8)
ax.set_xticks(np.arange(np.ceil(AA[0] / 0.02) * 0.02, AA[1], 0.02).round(2))
ax.set_yticks(np.arange(np.ceil(AA[2] / 0.01) * 0.01, AA[3], 0.01).round(2))
ax.axis(AA)
ax.set_autoscale_on(False)
pfun.dar(ax)
ax.tick_params(length=6)
ax.set_xlabel('Longitude [$^{\\circ}$E]')
ax.set_ylabel('Latitude [$^{\\circ}$N]')
ax.legend(fontsize=13, loc='lower right', framealpha=0.95)
fig.tight_layout()
pos = ax.get_position()
cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.016, pos.height])
cb = fig.colorbar(cd, cax=cax, extend='max')
cb.ax.invert_yaxis()
cb.set_label('Depth [m]')
fn_out = out_dir / ('pcmap_quadrant_split_map_%s.png' % tag)
fig.savefig(fn_out, **SAVE_KW)
plt.close(fig)
print('wrote %s' % fn_out)
