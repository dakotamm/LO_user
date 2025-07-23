#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:39:28 2024

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



# %%

Ldir = Lfun.Lstart(gridname='cas7')

# %%

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

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))

# %%

basin_list = list(odf_dict.keys())

var_list = ['SA', 'CT', 'DO_mg_L'] #'NO3 (uM)', 'Chl (mg m-3)', 'NO2 (uM)', 'SiO4 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']

# %%

for key in odf_dict.keys():
    
    odf_dict[key] = (odf_dict[key]
                      .assign(
                          datetime=(lambda x: pd.to_datetime(x['time'], utc=True)),
                          year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                          month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                          season=(lambda x: pd.cut(x['month'],
                                                  bins=[0,3,6,9,12],
                                                  labels=['winter', 'spring', 'summer', 'fall'])),
                          DO_mg_L=(lambda x: x['DO (uM)']*32/1000),
                          date_ordinal=(lambda x: x['datetime'].apply(lambda x: x.toordinal())),
                          segment=(lambda x: key),
                          decade=(lambda x: pd.cut(x['year'],
                                                  bins=[1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2019, 2029],
                                                  labels=['1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'], right=True))
                              )
                      )
    
    for var in var_list:
        
        if var not in odf_dict[key].columns:
            
            odf_dict[key][var] = np.nan
            
    odf_dict[key] = pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'source', 'otype', 'decade', 'name'],
                                          value_vars=var_list, var_name='var', value_name = 'val')
    
# %%

odf = pd.concat(odf_dict.values(), ignore_index=True)

# %%

odf['source_type'] = odf['source'] + '_' + odf['otype']

# %%

odf.loc[odf['source'].isin(['kc_taylor', 'kc_whidbey', 'kc_point_jefferson', 'kc']), 'Data Source'] = 'King County'

odf.loc[odf['source'].isin(['ecology_nc']), 'Data Source'] = 'WA Dept. of Ecology'

odf.loc[odf['source'].isin(['collias']), 'Data Source'] = 'Collias'

odf.loc[odf['source'].isin(['nceiSalish']), 'Data Source'] = 'NCEI'



# %%

mosaic = [['map_source', '.'], ['map_source', '.'], ['count_time_series', 'count_time_series'], ['depth_time_series', 'depth_time_series']]



fig, ax = plt.subplot_mosaic(mosaic, figsize=(6,8), layout='constrained')

plot_df = odf.groupby(['Data Source', 'cid']).first().reset_index()

sns.scatterplot(data=plot_df, x='lon', y='lat', hue='Data Source', ax = ax['map_source'], palette='Set2', alpha=0.5)

pfun.add_coast(ax['map_source'])

pfun.dar(ax['map_source'])

ax['map_source'].set_xlim(-123.2, -122.1)

ax['map_source'].set_ylim(47,48.5)

ax['map_source'].legend(loc='lower left', bbox_to_anchor=(1, 0), title='Data Source')

ax['map_source'].set_xlabel('Longitude')

ax['map_source'].set_ylabel('Latitude')




plot_df = (odf
                      .groupby(['Data Source','year']).agg({'cid' :lambda x: x.nunique()})
                      .reset_index()
                     # .rename(columns={'cid':'cid_count'})
                      #.rename(columns={'datetime_first':'datetime'})
                      )

sns.scatterplot(data=plot_df, x='year', y='cid', hue='Data Source', ax=ax['count_time_series'], palette='Set2', alpha =0.5, legend=False)

ax['count_time_series'].set_xlabel('Year')

ax['count_time_series'].set_ylabel('Cast Count')

ax['count_time_series'].set_ylim(0,1250)

ax['count_time_series'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)



plot_df = odf.groupby(['Data Source','year', 'cid']).min().reset_index()

plot_df = plot_df.groupby(['Data Source', 'year']).mean(numeric_only=True).reset_index()

sns.scatterplot(data=plot_df, x='year', y='z', hue='Data Source', ax=ax['depth_time_series'], palette='Set2', alpha =0.5, legend=False)

ax['depth_time_series'].set_xlabel('Year')

ax['depth_time_series'].set_ylabel('Average Cast Depth [m]')

ax['depth_time_series'].set_ylim(-200,0)

ax['depth_time_series'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)


plt.savefig('/Users/dakotamascarenas/Desktop/pltz/source_locations_freq_depth.png', bbox_inches='tight', dpi=500)
