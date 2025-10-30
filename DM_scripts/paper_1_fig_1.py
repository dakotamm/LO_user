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

from cmcrameri import cm




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

odf.loc[odf['source'].isin(['kc_his', 'kc_whidbeyBasin', 'kc_pointJefferson', 'kc']), 'Data Source'] = 'King County (KC)'

odf.loc[odf['source'].isin(['ecology_nc', 'ecology_his']), 'Data Source'] = 'WA Dept. of Ecology (Eco.)'

odf.loc[odf['source'].isin(['collias']), 'Data Source'] = 'Collias (Col.)'

odf.loc[odf['source'].isin(['nceiSalish']), 'Data Source'] = 'NCEI Salish Sea (NCEI)'


odf['site'] = odf['segment']


odf.loc[odf['otype'] == 'ctd', 'Sampling Type'] = 'CTD+DO'

odf.loc[odf['otype'] == 'bottle', 'Sampling Type'] = 'Bottle'

odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'kc_his'), 'Sampling Type'] = 'Sonde (unknown type)'

odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'kc_pointJefferson') & (odf['year'] <= 1998), 'Sampling Type'] = 'Sonde (unknown type)'

odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'kc_pointJefferson') & (odf['year'] <= 1998), 'Sampling Type'] = 'Sonde (unknown type)'

#odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'ecology_his') & (odf['year'] <= 1988), 'Sampling Type'] = 'Sonde (unknown type)'


# %%

# Sort to ensure sequential years per group
df =odf.copy().sort_values(['Data Source', 'Sampling Type', 'year'])

# Detect breaks (year gap > 1)
df['block'] = (
    df.groupby(['Data Source', 'Sampling Type'])['year']
      .diff().gt(1)
      .cumsum()
)

# Summarize each block into start and end year
coverage_blocks = (
    df.groupby(['Data Source', 'Sampling Type', 'block'])['year']
      .agg(['min', 'max'])
      .reset_index()
)
coverage_blocks['duration'] = coverage_blocks['max'] - coverage_blocks['min'] + 1

new_block = pd.DataFrame({'Data Source':['WA Dept. of Ecology (Eco.)'], 'Sampling Type':['Sonde (unknown type)'],
                          'block':[5], 'min':[1973], 'max':[1989], 'duration':[17]})

coverage_blocks = pd.concat([coverage_blocks, new_block], ignore_index=True)


mosaic = [['map_source', 'map_source','type_series', 'type_series', 'type_series'],
          ['map_source', 'map_source','depth_time_series', 'depth_time_series', 'depth_time_series'],
          ['map_source', 'map_source','depth_time_series', 'depth_time_series', 'depth_time_series'],
          ['map_source', 'map_source','count_time_series','count_time_series','count_time_series'], 
          ['map_source', 'map_source','count_time_series','count_time_series','count_time_series']]

fig, axd = plt.subplot_mosaic(mosaic, figsize=(9,6), layout='constrained', gridspec_kw=dict(wspace=0.1))

plot_df = odf.groupby(['Data Source', 'cid']).first().reset_index()



#N = len(df['Data Source'].unique())

palette = [
    "#3A59B3",  # deep blue
    "#C7C445",  # yellow-green
    "#B0448E",  # magenta-violet
    "#EF5E3C"   # warm orange-red
]


ax = axd['map_source']

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

sns.scatterplot(data=plot_df, x='lon', y='lat', hue='Data Source', ax = ax, palette=palette, alpha=0.5, legend=False)

pfun.add_coast(ax)

pfun.dar(ax)

ax.set_xlim(X[i1],-121.4)#X[i2]) # Salish Sea
ax.set_ylim(Y[j1],Y[j2]) # Salish Sea

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-100, vmax=0, cmap = 'gray')

ax.set_xlabel('')

ax.set_ylabel('')

ax.set_xticks([-123.0, -122.6, -122.2], ['-123.0','-122.6', '-122.2'])

ax.text(0.05,0.025, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')

ax.set_xlim(-123.2, -122.1) 
 
ax.set_ylim(47,48.5)



ax = axd['type_series']

sources = sorted(coverage_blocks['Data Source'].unique())
otypes = sorted(coverage_blocks['Sampling Type'].unique())

color_map = dict(zip(sources, palette))
hatch_map = dict(zip(otypes, ['', '...', '|||']))
y_map = dict(zip(sources, [0, 0.1, 0.2, 0.3]))

for _, row in coverage_blocks.iterrows():
    if row['Sampling Type'] == 'Bottle':
        fill = True
    else:
        fill = False
    ax.barh(
        y=y_map[row['Data Source']],
        width=row['duration'],
        left=row['min'],
        height=0.075,
        color=color_map[row['Data Source']],
        hatch=hatch_map[row['Sampling Type']], 
        edgecolor='black',
        fill = fill
    )

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)

ax.invert_yaxis()

ax.text(0.025,0.1, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')

ax.set_xlim(1930, 2025)

ax.set_xticklabels([])  # Remove x-axis tick labels

ax.set_yticks([0,0.1,0.2,0.3],['Col.','KC', 'NCEI', 'Eco.'])

ax.set_ylabel('Data Source')





ax = axd['depth_time_series']

plot_df = odf.groupby(['Data Source','year', 'cid']).min().reset_index()

plot_df = plot_df.groupby(['Data Source', 'year']).mean(numeric_only=True).reset_index()

sns.scatterplot(data=plot_df, x='year', y='z', hue='Data Source', ax=ax, palette=palette)

ax.set_xlabel('')

ax.set_ylabel('Annual Avg. Cast Depth [m]')

ax.set_ylim(-250,0)

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)

ax.text(0.025,0.05, 'c', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')

ax.set_xlim(1930, 2025)

ax.set_xticklabels([])  # Remove x-axis tick labels





ax = axd['count_time_series']

plot_df = (odf
                      .groupby(['Data Source','year']).agg({'cid' :lambda x: x.nunique()})
                      .reset_index()
                      )

sns.scatterplot(data=plot_df, x='year', y='cid', hue='Data Source', ax=ax, palette=palette, legend=False)

ax.set_xlabel('')

ax.set_ylabel('Annual Cast Count')

ax.set_ylim(0,1300)

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)

ax.text(0.025,0.05, 'd', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')

ax.set_xlim(1930, 2025)


handles_count, labels_count = axd['depth_time_series'].get_legend_handles_labels()

otypes = sorted(coverage_blocks['Sampling Type'].unique())
hatch_map = dict(zip(otypes, ['', '...', '|||']))

leg_colors = {'Bottle': 'gray', 'CTD+DO':'white', 'Sonde (unknown type)': 'white'}
hatch_handles = [
    patches.Patch(facecolor=leg_colors[o], edgecolor='black', hatch=hatch_map[o], label=o)
    for o in otypes
]


leg1 = fig.legend(
    handles_count, labels_count,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.1),  # left side
    ncol=4,
    title='Data Source'
    )

leg2 = fig.legend(
    hatch_handles, [o for o in otypes],
    loc='lower center',
    bbox_to_anchor=(0.5, -0.2),  # right side
    ncol=len(otypes),
    title='Sampling Type'
)

axd['depth_time_series'].get_legend().remove()




plt.savefig('/Users/dakotamascarenas/Desktop/pltz/paper_1_fig_1-new-new.png', bbox_inches='tight', dpi=500, transparent=True)