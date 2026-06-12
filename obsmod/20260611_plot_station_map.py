"""
Station map for the King County (kc_whidbeyBasin) and Ecology (ecology_nc)
stations used in the wb1_t0_xn11abbur00 obs-model validation.

Shows the wb1 model domain (water shaded light blue, grid perimeter outlined),
the coastline, and every in-domain station, styled by source x otype:
  ecology ctd     - light gray filled
  kc_whidbey ctd  - dark gray filled
  ecology bottle  - blue filled
  kc_whidbey bottle - orange open ring
Stations sampled by both otypes show a filled dot inside a ring. Each station
is labeled by name.

    python 20260611_plot_station_map.py
    python 20260611_plot_station_map.py -test True   # show instead of save
"""

import argparse
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    'val_functions', str(Path(__file__).parent / '20260611_val_functions.py'))
vf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vf)

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', type=str, default=vf.DEFAULT_GTX)
parser.add_argument('-years', type=str, default='2024,2025')
parser.add_argument('-otypes', type=str, default='ctd,bottle')
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)
args = parser.parse_args()

Ldir = Lfun.Lstart()
if '_mac' in Ldir['lo_env']:
    pass
else:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt

gtx = args.gtagex
gridname = gtx.split('_')[0]
years = [y.strip() for y in args.years.split(',') if y.strip()]
otypes = [o.strip() for o in args.otypes.split(',') if o.strip()]

out_dir = vf.out_dir(Ldir)
Lfun.make_dir(out_dir)

# style per (source, otype); ctd drawn as filled dots, bottle as rings on top
STYLE = {
    ('ecology_nc', 'ctd'):
        dict(marker='o', mfc='0.8', mec='0.35', mew=0.8, ms=9, ls='',
             label='ecology ctd', zorder=4),
    ('kc_whidbeyBasin', 'ctd'):
        dict(marker='o', mfc='0.45', mec='0.15', mew=0.8, ms=9, ls='',
             label='kc_whidbey ctd', zorder=4),
    ('ecology_nc', 'bottle'):
        dict(marker='o', mfc='tab:blue', mec='k', mew=0.8, ms=9, ls='',
             label='ecology bottle', zorder=5),
    ('kc_whidbeyBasin', 'bottle'):
        dict(marker='o', mfc='none', mec='tab:orange', mew=2.5, ms=15, ls='',
             label='kc_whidbey bottle', zorder=6),
}

# ---- load grid ---------------------------------------------------------------
g = xr.open_dataset(Ldir['data'] / 'grids' / gridname / 'grid.nc')
lon = g['lon_rho'].values
lat = g['lat_rho'].values
mask = g['mask_rho'].values            # 1 = water, 0 = land
lon_ax = lon[0, :]                      # ~plaid grid: lon along xi, lat along eta
lat_ax = lat[:, 0]


def in_domain(x, y):
    """True if (x,y) is within the grid extent and lands on a water cell."""
    if not (lon.min() <= x <= lon.max() and lat.min() <= y <= lat.max()):
        return False
    ix = zfun.find_nearest_ind(lon_ax, x)
    iy = zfun.find_nearest_ind(lat_ax, y)
    return mask[iy, ix] == 1


# ---- collect in-domain stations: {(source,otype): {name:(lon,lat)}} ----------
def load_stations():
    out = {}
    names_xy = {}
    for source in vf.SOURCES:
        for otype in otypes:
            base = Ldir['LOo'] / 'obs' / source / otype
            d = {}
            for year in years:
                info_fn = base / ('info_' + year + '.p')
                if not info_fn.is_file():
                    continue
                info = pd.read_pickle(info_fn)
                for _, r in info.iterrows():
                    nm, x, y = r['name'], float(r['lon']), float(r['lat'])
                    if not in_domain(x, y):
                        continue
                    d[nm] = (x, y)
                    names_xy[nm] = (x, y)
            if d:
                out[(source, otype)] = d
    return out, names_xy


stations, names_xy = load_stations()

# ---- plot --------------------------------------------------------------------
pfun.start_plot(figsize=(10, 13), fs=12)
fig, ax = plt.subplots()

# water mask (light blue) and grid perimeter
ax.pcolormesh(lon, lat, np.ma.masked_where(mask == 0, mask),
              cmap=ListedColormap(['#cfe2f3']), shading='auto', zorder=0)
perim_lon = np.concatenate([lon[0, :], lon[:, -1], lon[-1, ::-1], lon[::-1, 0]])
perim_lat = np.concatenate([lat[0, :], lat[:, -1], lat[-1, ::-1], lat[::-1, 0]])
ax.plot(perim_lon, perim_lat, '-k', lw=1, zorder=2)
pfun.add_coast(ax)

# draw ctd first, then bottle rings on top, so combined stations show dot+ring
order = [k for k in STYLE if k[1] == 'ctd'] + [k for k in STYLE if k[1] == 'bottle']
for key in order:
    if key not in stations:
        continue
    xs = [v[0] for v in stations[key].values()]
    ys = [v[1] for v in stations[key].values()]
    ax.plot(xs, ys, **STYLE[key])

# station labels (once per unique station)
for nm, (x, y) in names_xy.items():
    ax.annotate(nm, (x, y), xytext=(6, 0), textcoords='offset points',
                fontsize=7, va='center', zorder=7)

pfun.dar(ax)
ax.axis([lon.min(), lon.max(), lat.min(), lat.max()])
ax.set_xlabel('lon'); ax.set_ylabel('lat')
ax.set_title('Whidbey Basin stations, %s' % '-'.join(years))

# legend: station categories present + the grid line
handles = [Line2D([0], [0], **STYLE[k]) for k in order if k in stations]
handles = [Line2D([0], [0], color='k', lw=1, label='wb1 grid')] + handles
ax.legend(handles=handles, loc='lower left', fontsize=9, framealpha=0.9)

fig.tight_layout()
name = 'station_map_%s' % gtx
if args.testing:
    plt.show()
else:
    fig.savefig(out_dir / (name + '.png'), bbox_inches='tight')
    print('Saved %s.png' % name)
plt.close(fig)
print('Done. %d unique in-domain stations.' % len(names_xy))
