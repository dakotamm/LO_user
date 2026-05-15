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

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))

basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf.loc[odf['source'].isin(['kc_taylor', 'kc_whidbey', 'kc_point_jefferson', 'kc']), 'Data Source'] = 'King County'

odf.loc[odf['source'].isin(['ecology_nc', 'ecology_his']), 'Data Source'] = 'WA Dept. of Ecology'

odf.loc[odf['source'].isin(['collias']), 'Data Source'] = 'Collias'

odf.loc[odf['source'].isin(['nceiSalish']), 'Data Source'] = 'NCEI Salish Sea'


odf['site'] = odf['segment']

# %%

mosaic = [['map_source', 'depth_time_series', 'depth_time_series'], ['map_source', 'count_time_series', 'count_time_series']] #, ['map_source', '.', '.'],]

fig, ax = plt.subplot_mosaic(mosaic, figsize=(9,5), layout='constrained', gridspec_kw=dict(wspace=0.1))

plot_df = odf.groupby(['Data Source', 'cid']).first().reset_index()

ax['map_source'].pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)


sns.scatterplot(data=plot_df, x='lon', y='lat', hue='Data Source', ax = ax['map_source'], palette='Set2', alpha=0.5, legend=False)

pfun.add_coast(ax['map_source'])

pfun.dar(ax['map_source'])

ax['map_source'].set_xlim(X[i1],-121.4)#X[i2]) # Salish Sea
ax['map_source'].set_ylim(Y[j1],Y[j2]) # Salish Sea

ax['map_source'].pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-100, vmax=0, cmap = 'gray')


#ax['map_source'].legend(loc='upper center', title ='Data Source') #, bbox_to_anchor=(0.5, -0.1), title='Data Source')

ax['map_source'].set_xlabel('')

ax['map_source'].set_ylabel('')

ax['map_source'].set_xticks([-123.0, -122.6, -122.2], ['-123.0','-122.6', '-122.2']) #['','-123.0', '', '-122.6', '', '-122.2'])

ax['map_source'].text(0.05,0.025, 'a', transform=ax['map_source'].transAxes, fontsize=14, fontweight='bold', color = 'k')

ax['map_source'].set_xlim(-123.2, -122.1) 
 
ax['map_source'].set_ylim(47,48.5)





plot_df = (odf
                      .groupby(['Data Source','year']).agg({'cid' :lambda x: x.nunique()})
                      .reset_index()
                     # .rename(columns={'cid':'cid_count'})
                      #.rename(columns={'datetime_first':'datetime'})
                      )

sns.scatterplot(data=plot_df, x='year', y='cid', hue='Data Source', ax=ax['count_time_series'], palette='Set2')

ax['count_time_series'].set_xlabel('')

ax['count_time_series'].set_ylabel('Annual Cast Count')

ax['count_time_series'].set_ylim(0,1300)

ax['count_time_series'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)

ax['count_time_series'].legend(loc='upper left', title ='Data Source') #, bbox_to_anchor=(0.5, -0.1), title='Data Source')

ax['count_time_series'].text(0.025,0.05, 'c', transform=ax['count_time_series'].transAxes, fontsize=14, fontweight='bold', color = 'k')





plot_df = odf.groupby(['Data Source','year', 'cid']).min().reset_index()

plot_df = plot_df.groupby(['Data Source', 'year']).mean(numeric_only=True).reset_index()

sns.scatterplot(data=plot_df, x='year', y='z', hue='Data Source', ax=ax['depth_time_series'], palette='Set2', legend=False)

ax['depth_time_series'].set_xlabel('')

ax['depth_time_series'].set_ylabel('Annual Avg. Cast Depth [m]')

ax['depth_time_series'].set_ylim(-250,0)

ax['depth_time_series'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)

ax['depth_time_series'].text(0.025,0.05, 'b', transform=ax['depth_time_series'].transAxes, fontsize=14, fontweight='bold', color = 'k')



plt.savefig('/Users/dakotamascarenas/Desktop/pltz/paper_1_fig_1.png', bbox_inches='tight', dpi=500, transparent=True)