"""
Plot all PRISM Salish Cruise CTD station locations (unique lat/lon).

Reads the processed cast info files produced by process_ctd.py and maps every
unique sampling position, sized/colored by the number of casts there.

Written by: Dakota Mascarenas
Initial author date: 2026/06/15

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun
Ldir = Lfun.Lstart()


# source location
source = 'prism'
otype = 'ctd'
in_dir = Ldir['LOo'] / 'obs' / source / otype
year_list = range(1998,2019)

# Gather one row per cast (lat/lon) from the info files.
info_df = pd.DataFrame()
for year in year_list:
    info_fn = in_dir / ('info_' + str(year) + '.p')
    if info_fn.is_file():
        info_df = pd.concat([info_df, pd.read_pickle(info_fn)])

# Count casts at each unique lat/lon position.
sta_df = (info_df.groupby(['lon','lat']).size()
          .reset_index(name='ncast'))
print('total casts: %d' % len(info_df))
print('unique lat/lon stations: %d' % len(sta_df))

# Map.
fig, ax = plt.subplots(figsize=(8,10))
pfun.add_coast(ax)
sc = ax.scatter(sta_df['lon'], sta_df['lat'], c=sta_df['ncast'], s=40,
                cmap='viridis', edgecolors='k', linewidth=0.4, zorder=5)
cb = fig.colorbar(sc, ax=ax, shrink=0.6)
cb.set_label('number of casts')

pad = 0.1
ax.set_xlim(sta_df['lon'].min()-pad, sta_df['lon'].max()+pad)
ax.set_ylim(sta_df['lat'].min()-pad, sta_df['lat'].max()+pad)
pfun.dar(ax)  # correct lon/lat aspect ratio
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('PRISM Salish Cruise CTD stations (%d unique positions)' % len(sta_df))
ax.grid(color='lightgray', linestyle='--', alpha=0.5)

plt.tight_layout()
out_fn = 'prism_ctd_stations.png'
plt.savefig(out_fn, dpi=150, bbox_inches='tight')
print('saved %s' % out_fn)
plt.show()
