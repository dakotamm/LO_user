#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:29:56 2024

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




#poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson', 'mb', 'hc', 'ss', 'wb'] # 5 sites + 4 basins

poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)


# %%
low_DO_season_start = 213 #aug1

low_DO_season_end = 335 #nov30

odf_use = odf_depth_mean.copy()

odf_calc_use = odf_calc_long.copy()

odf_use_full = (odf_use
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

odf_calc_use_full = (odf_calc_use
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

odf_use_AugNov = odf_use_full[(odf_use_full['yearday'] >= low_DO_season_start) & (odf_use_full['yearday'] <= low_DO_season_end)]

odf_calc_use_AugNov = odf_calc_use_full[(odf_calc_use_full['yearday'] >= low_DO_season_start) & (odf_calc_use_full['yearday'] <= low_DO_season_end)]




for deep_DO_q in ['deep_DO_q50']:


    odf_depth_mean_deep_DO_less_than_percentile = odf_depth_mean_deep_DO_percentiles[odf_depth_mean_deep_DO_percentiles['val'] <= odf_depth_mean_deep_DO_percentiles[deep_DO_q]]

    cid_deep_DO_less_than_percentile = odf_depth_mean_deep_DO_less_than_percentile['cid']

    odf_use = odf_depth_mean[odf_depth_mean['cid'].isin(cid_deep_DO_less_than_percentile)]

    odf_calc_use = odf_calc_long[odf_calc_long['cid'].isin(cid_deep_DO_less_than_percentile)]

    odf_use = (odf_use
                      .dropna()
                      .assign(
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )
    
    odf_calc_use = (odf_calc_use
                      .dropna()
                      .assign(
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )
    
    odf_use = odf_use[(odf_use['yearday'] >= low_DO_season_start) & (odf_use['yearday'] <= low_DO_season_end)]
    
    odf_calc_use = odf_calc_use[(odf_calc_use['yearday'] >= low_DO_season_start) & (odf_calc_use['yearday'] <= low_DO_season_end)]
    
    if deep_DO_q == 'deep_DO_q25':
    
        odf_use_q25 = odf_use
        
        odf_calc_use_q25 = odf_calc_use
        
    elif deep_DO_q == 'deep_DO_q50':
        
        odf_use_q50 = odf_use
        
        odf_calc_use_q50 = odf_calc_use
        
    elif deep_DO_q == 'deep_DO_q75':
        
        odf_use_q75 = odf_use
        
        odf_calc_use_q75 = odf_calc_use

# %%


# %%


odf_use_DO_deep = odf_depth_mean[(odf_depth_mean['var'] == 'DO_mg_L') & (odf_depth_mean['surf_deep'] == 'deep')].reset_index(drop=True)


DO_min_deep_idx = odf_use_DO_deep.groupby(['site','var','year']).idxmin()['val'].to_numpy()

odf_use_DO_min_deep = odf_use_DO_deep[odf_use_DO_deep.index.isin(DO_min_deep_idx)]

# %%

site_list =  odf['site'].unique()




odf_use = odf_depth_mean.copy()

odf_calc_use = odf_calc_long.copy()

all_stats_filt = dfun.buildStatsDF(odf_use, site_list, odf_calc_use=odf_calc_use, odf_depth_mean_deep_DO_percentiles=odf_depth_mean_deep_DO_percentiles)


# %%

odf_use = (odf_use
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

# %%


mosaic = [['map_source', 'CT', 'CT'], ['map_source', 'SA', 'SA'], ['map_source', 'DO_mg_L', 'DO_mg_L']] #, ['map_source', '.', '.'],]

fig, axd = plt.subplot_mosaic(mosaic, figsize=(10,7), layout='constrained')


ax = axd['map_source']
 
ax.set_xlim(X[i1],-121.4)#X[i2]) # Salish Sea
ax.set_ylim(Y[j1],Y[j2]) # Salish Sea
        
ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

pfun.add_coast(ax)

pfun.dar(ax)

for site in long_site_list:
    
    path = path_dict[site]
    
        
    if site in ['point_jefferson', 'near_seattle_offshore']:
            

        patch = patches.PathPatch(path, facecolor='#e04256', edgecolor='white', zorder=1)#, label='>60-year history')
                
    else:
        
        patch = patches.PathPatch(path, facecolor='#4565e8', edgecolor='white', zorder=1)
        
    ax.add_patch(patch)




ax.set_xlim(-123.2, -122.1) 

ax.set_ylim(47,48.5)


ax.set_xlabel('')

ax.set_ylabel('')

ax.tick_params(axis='x', labelrotation=45)

palette = {'point_jefferson':'#e04256', 'lynch_cove_mid':'#4565e8'}


for var in var_list:
    
    ax = axd[var]
                        
    if 'DO' in var:
        
        label_var = '[DO]'
        
        ymin = 0
        
        ymax = 8
        
        marker = 'o'
        
        unit = r'[mg/L]'
        
    elif 'CT' in var:
        
        label_var = 'Temperature'
        
        ymin = 8
        
        ymax = 14
        
        marker = 'D'
        
        unit = r'[$^{\circ}$C]'
    
    else:
        
        label_var = 'Salinity'
        
        ymin = 28
        
        ymax = 34
        
        marker = 's'
        
        unit = r'[PSU]'

    for site in ['point_jefferson', 'lynch_cove_mid']:
        
        if var == 'DO_mg_L':
            
            plot_df_q50 = odf_use_q50[(odf_use_q50['site'] == site) & (odf_use_q50['var'] == var) & (odf_use_q50['surf_deep'] == 'deep')]
        
            sns.scatterplot(data=plot_df_q50, x='datetime', y = 'val',  ax=ax, color = palette[site], marker=marker)
            
        else:
            
            plot_df = odf_use[(odf_use['site'] == site) & (odf_use['var'] == var) & (odf_use['surf_deep'] == 'deep') & (odf_use['summer_non_summer'] == 'summer')]
            
            sns.scatterplot(data=plot_df, x='datetime', y = 'val',  ax=ax, color = palette[site], marker=marker)

          
        ax.set_ylim(ymin, ymax) 
        
        ax.axhspan(0,2, color = 'lightgray', alpha = 0.3, zorder=-5)
        
        ax.set_ylabel('Filtered ' + label_var + ' ' + unit)
        
        ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax.set_xlabel('')
        
        plt.tight_layout()
        
        
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/fig1_test.png', bbox_inches='tight', dpi=500, transparent=True)
    
