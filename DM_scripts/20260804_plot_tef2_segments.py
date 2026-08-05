"""
Map the tef2 segments and sections for a collection, so the geometry can be
checked before running an extraction.

Left panel is the whole region of interest, right panel zooms on Penn Cove.
Segments are filled by color and labelled with cell count, volume and mean
depth. Sections are drawn with an arrow showing the direction of POSITIVE
transport, which is the thing you need in order to get budget signs right.

run 20260804_plot_tef2_segments.py -gctag wb1_pc1 -riv riv00
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Patch

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

parser = argparse.ArgumentParser()
parser.add_argument('-gctag', default='wb1_pc1', type=str)
parser.add_argument('-riv', default='riv00', type=str)
args = parser.parse_args()

gridname = args.gctag.split('_')[0]
Ldir = Lfun.Lstart(gridname=gridname)

tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'
sect_df = pd.read_pickle(tef2_dir / ('sect_df_' + args.gctag + '.p'))
coll_dir = tef2_dir / ('sections_' + args.gctag)
seg_info = pickle.load(
    open(tef2_dir / ('seg_info_dict_' + args.gctag + '_' + args.riv + '.p'), 'rb'))
vol_df = pd.read_pickle(tef2_dir / ('vol_df_' + args.gctag + '.p'))

dsg = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon = dsg.lon_rho.values
lat = dsg.lat_rho.values
lon_u = dsg.lon_u.values
lat_u = dsg.lat_u.values
lon_v = dsg.lon_v.values
lat_v = dsg.lat_v.values
mask = dsg.mask_rho.values
h = dsg.h.values
dsg.close()

with open(coll_dir / 'bounding_sections.txt', 'r') as f:
    bounding = [s for s in f.read().split('\n') if s]

# stable color per segment, biggest first so Penn Cove keeps distinct colors
seg_names = sorted(seg_info.keys(), key=lambda k: -len(seg_info[k]['ji_list']))
colors = plt.get_cmap('tab10').colors
seg_color = {k: colors[i % len(colors)] for i, k in enumerate(seg_names)}

# paint segments onto a grid of indices into seg_names
paint = np.full(mask.shape, np.nan)
for i, k in enumerate(seg_names):
    ji = np.array(seg_info[k]['ji_list'])
    paint[ji[:, 0], ji[:, 1]] = i

cmap = plt.matplotlib.colors.ListedColormap([seg_color[k] for k in seg_names])


def draw(ax, aa, label_segs=True):
    # land / water backdrop
    land = np.where(mask == 1, np.nan, 1.0)
    ax.pcolormesh(lon, lat, np.ma.masked_invalid(land), cmap='Greys',
                  vmin=0, vmax=3, shading='nearest', zorder=1)
    water = np.where(mask == 1, 1.0, np.nan)
    ax.pcolormesh(lon, lat, np.ma.masked_invalid(water), cmap='Blues',
                  vmin=0, vmax=6, shading='nearest', zorder=2)
    ax.pcolormesh(lon, lat, np.ma.masked_invalid(paint), cmap=cmap,
                  vmin=-0.5, vmax=len(seg_names) - 0.5, shading='nearest',
                  alpha=0.75, zorder=3)
    pfun.add_coast(ax, color='k', linewidth=0.6)

    # sections, with an arrow for the positive-transport direction
    for sn in sect_df.sn.unique():
        d = sect_df[sect_df.sn == sn]
        du = d[d.uv == 'u']
        dv = d[d.uv == 'v']
        xs = np.concatenate([lon_u[du.j.values, du.i.values],
                             lon_v[dv.j.values, dv.i.values]])
        ys = np.concatenate([lat_u[du.j.values, du.i.values],
                             lat_v[dv.j.values, dv.i.values]])
        lw = 3.5 if sn in bounding else 2.5
        ax.plot(xs, ys, 's', color='magenta', markersize=lw, zorder=10)
        if xs.mean() < aa[0] or xs.mean() > aa[1]:
            continue
        if ys.mean() < aa[2] or ys.mean() > aa[3]:
            continue
        # arrow: mean outward normal weighted by pm
        ax.text(xs.mean(), ys.mean(), sn + ('*' if sn in bounding else ''),
                color='magenta', fontsize=9, fontweight='bold', ha='center',
                va='bottom', zorder=12,
                bbox=dict(fc='w', ec='none', alpha=0.6, pad=1))
        # Positive transport goes from the rho cell on the minus side to the
        # one on the plus side, so take the direction straight from those two
        # cells. Doing it from pm and uv separately gets diagonal sections
        # wrong, since their u and v faces carry different pm.
        r = d.iloc[len(d) // 2]
        x0, y0 = lon[r.jrm, r.irm], lat[r.jrm, r.irm]
        x1, y1 = lon[r.jrp, r.irp], lat[r.jrp, r.irp]
        ax.annotate('', xy=(x0 + 4 * (x1 - x0), y0 + 4 * (y1 - y0)),
                    xytext=(x0, y0), zorder=13,
                    arrowprops=dict(color='magenta', width=1.5, headwidth=8))

    if label_segs:
        for k in seg_names:
            ji = np.array(seg_info[k]['ji_list'])
            x = lon[ji[:, 0], ji[:, 1]].mean()
            y = lat[ji[:, 0], ji[:, 1]].mean()
            if x < aa[0] or x > aa[1] or y < aa[2] or y > aa[3]:
                continue
            V = vol_df.loc[k, 'volume m3']
            A = vol_df.loc[k, 'area m2']
            ax.text(x, y, '%s\n%d cells\n%.2f km3\n%.0f m mean' %
                    (k, len(ji), V / 1e9, V / A),
                    fontsize=8, ha='center', va='center', zorder=14,
                    bbox=dict(fc='w', ec='k', alpha=0.8, pad=2))

    pfun.dar(ax)
    ax.axis(aa)
    ax.set_xlabel('Longitude')


plt.close('all')
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

# left: whole region of interest, padded around all segments
allji = np.concatenate([np.array(seg_info[k]['ji_list']) for k in seg_names])
pad = 0.02
aa_all = [lon[allji[:, 0], allji[:, 1]].min() - pad,
          lon[allji[:, 0], allji[:, 1]].max() + pad,
          lat[allji[:, 0], allji[:, 1]].min() - pad,
          lat[allji[:, 0], allji[:, 1]].max() + pad]
draw(axes[0], aa_all)
axes[0].set_ylabel('Latitude')
axes[0].set_title('%s segments (* = bounding section)\nmagenta arrow = '
                  'positive transport direction' % args.gctag)

# right: Penn Cove
pcji = np.concatenate([np.array(seg_info[k]['ji_list'])
                       for k in seg_names if k != 'pc_lp_p'])
pad = 0.008
aa_pc = [lon[pcji[:, 0], pcji[:, 1]].min() - pad,
         lon[pcji[:, 0], pcji[:, 1]].max() + 3 * pad,
         lat[pcji[:, 0], pcji[:, 1]].min() - pad,
         lat[pcji[:, 0], pcji[:, 1]].max() + pad]
draw(axes[1], aa_pc)
axes[1].set_title('Penn Cove')

fig.tight_layout()
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)
fn_out = out_dir / ('20260804_tef2_segments_' + args.gctag + '.png')
fig.savefig(fn_out, dpi=200)
print('saved ' + str(fn_out))

print('\n' + vol_df.to_string())
print('\ntotal Penn Cove volume: %.3f km3' %
      (vol_df.loc[['pc_lp_m', 'pc_cp_p', 'pc_cp_m'], 'volume m3'].sum() / 1e9))
