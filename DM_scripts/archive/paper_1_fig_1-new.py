#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 12:21:46 2025

@author: dakotamascarenas
"""

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun
import matplotlib.pyplot as plt
import matplotlib.path as mpth
import xarray as xr
import numpy as np
import pandas as pd
import datetime

import seaborn.objects as so

import matplotlib.patches as mpatches


from warnings import filterwarnings
filterwarnings('ignore') # skip some warning messages

import seaborn as sns

import scipy.stats as stats

import D_functions as dfun

import pickle

import math

from scipy.interpolate import interp1d

import gsw

import matplotlib.path as mpth

import matplotlib.patches as patches

import cmocean

import matplotlib.patheffects as pe




Ldir = Lfun.Lstart(gridname='cas7')


fng = Ldir['grid'] / 'grid.nc'
dsg = xr.open_dataset(fng)
lon = dsg.lon_rho.values
lat = dsg.lat_rho.values
m = dsg.mask_rho.values
xp, yp = pfun.get_plon_plat(lon,lat)
depths = dsg.h.values
depths[m==0] = np.nan

lon_1D = lon[0,:]

lat_1D = lat[:,0]

# weird, to fix

mask_rho = np.transpose(dsg.mask_rho.values)
zm = -depths.copy()
zm[np.transpose(mask_rho) == 0] = np.nan
zm[np.transpose(mask_rho) != 0] = -1

zm_inverse = zm.copy()

zm_inverse[np.isnan(zm)] = -1

zm_inverse[zm==-1] = np.nan


X = lon[0,:] # grid cell X values
Y = lat[:,0] # grid cell Y values

plon, plat = pfun.get_plon_plat(lon,lat)


j1 = 570
j2 = 1170
i1 = 220
i2 = 652


# %%

poly_list = ['ps']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_his', 'kc_whidbeyBasin', 'nceiSalish', 'kc_pointJefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))

basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf.loc[odf['source'].isin(['kc_his', 'kc_whidbeyBasin', 'kc_pointJefferson', 'kc']), 'Data Source'] = 'King County'

odf.loc[odf['source'].isin(['ecology_nc', 'ecology_his']), 'Data Source'] = 'WA Dept. of Ecology'

odf.loc[odf['source'].isin(['collias']), 'Data Source'] = 'Collias'

odf.loc[odf['source'].isin(['nceiSalish']), 'Data Source'] = 'NCEI Salish Sea'


odf['site'] = odf['segment']

# %%

odf.loc[odf['otype'] == 'ctd', 'Sampling Type'] = 'CTD+DO'

odf.loc[odf['otype'] == 'bottle', 'Sampling Type'] = 'Bottle'

# %%

odf['Year'] = odf['year']

# %%

# Sort to ensure sequential years per group
df =odf.sort_values(['Data Source', 'Sampling Type', 'Year'])

# Detect breaks (year gap > 1)
df['block'] = (
    df.groupby(['Data Source', 'Sampling Type'])['Year']
      .diff().gt(1)
      .cumsum()
)

# Summarize each block into start and end year
coverage_blocks = (
    df.groupby(['Data Source', 'Sampling Type', 'block'])['Year']
      .agg(['min', 'max'])
      .reset_index()
)
coverage_blocks['duration'] = coverage_blocks['max'] - coverage_blocks['min'] + 1


# %%
# --- Visualization ---
#sns.set_theme(style="whitegrid")

sources = sorted(coverage_blocks['Data Source'].unique())
otypes = sorted(coverage_blocks['Sampling Type'].unique())

# Define color and hatch maps
palette = sns.color_palette("Set2", len(sources))
color_map = dict(zip(sources, palette))
hatch_map = dict(zip(otypes, ['', '//']))

# --- Plot ---
fig, ax = plt.subplots(figsize=(9, 2.5))

# Determine vertical offsets so otypes within a source don't overlap
offset = 0.05
group_gap = 0.1
y_positions = {}

current_y = 0
for src in sources: 
    # Assign y positions per otype within this source
    otypes_in_src = coverage_blocks.loc[coverage_blocks['Data Source'] == src, 'Sampling Type'].unique() 
    y_positions.update({
        (src, ot): current_y + i * offset
        for i, ot in enumerate(otypes_in_src)
    })
    current_y += group_gap  # add gap before next source group

# Plot each (source, otype) bar
for _, row in coverage_blocks.iterrows():
    ax.barh(
        y=y_positions[(row['Data Source'], row['Sampling Type'])],
        width=row['duration'],
        left=row['min'],
        height=0.02,
        color=color_map[row['Data Source']],
        hatch=hatch_map[row['Sampling Type']],
        edgecolor='black'
    )

# Set Y ticks to the midpoint of each source group
ax.set_yticks([y_positions[(src, otypes_in_src[0])] + offset/2 for src in sources])
ax.set_yticklabels(sources)

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)

ax.invert_yaxis()

hatch_legend = [
    mpatches.Patch(
        facecolor='white',
        edgecolor='black',
        hatch=hatch_map[o],
        label=o
    )
    for o in otypes
]

ax.legend(
    handles=hatch_legend,
    title="Sampling Type",
    loc="lower left"
)

ax.set_xlim(xmin=1930)


# --- Labels and style ---
#ax.set_xlabel("Year")
#ax.set_ylabel("Source")
#ax.set_title("Year Coverage by Source (grouped) and OType (hatch)")
#sns.despine(ax=ax)
plt.tight_layout()


plt.savefig("/Users/dakotamascarenas/Desktop/pltz/paper_1_fig_1-new.png", dpi=500)

