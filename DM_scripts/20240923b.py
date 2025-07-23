#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:29:15 2024

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

for site in ['point_jefferson']:
    
    fig, ax = plt.subplots(figsize = (7,3))
    
    plot_df_q50 = odf_use_q50[(odf_use_q50['site'] == site) & (odf_use_q50['var'] == 'DO_mg_L') & (odf_use_q50['surf_deep'] == 'deep')]

    sns.scatterplot(data=plot_df_q50, x='datetime', y = 'val',  ax=ax, color = '#ff4040')
      
    ax.set_ylim(4, 7) 
    
    ax.axhspan(0,2, color = 'lightgray', alpha = 0.3)
    
    ax.set_ylabel('Filtered DO [mg/L]')
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)

    ax.set_xlabel('')
    
    plt.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_q50DO_trend_0.png', bbox_inches='tight', dpi=500, transparent=True)




for site in ['point_jefferson']:
    
    fig, ax = plt.subplots(figsize = (7,3))
    
    plot_df_q50 = odf_use_q50[(odf_use_q50['site'] == site) & (odf_use_q50['var'] == 'DO_mg_L') & (odf_use_q50['surf_deep'] == 'deep')]

    sns.scatterplot(data=plot_df_q50, x='datetime', y = 'val',  ax=ax, color = '#ff4040')
    
    stat_df = all_stats_filt[(all_stats_filt['site'] == site) & (all_stats_filt['var'] == 'deep_DO_mg_L') & (all_stats_filt['deep_DO_q'] == 'deep_DO_q50') & (all_stats_filt['summer_non_summer'] == 'summer')]
    
    
    
    x = plot_df_q50['date_ordinal']

    y = plot_df_q50['val']

    x_plot = plot_df_q50['datetime']
     
    B0 = stat_df['B0'].iloc[0]
    
    B1 = stat_df['B1'].iloc[0]
    
    ax.plot([x_plot.min(), x_plot.max()], [B0 + B1*x.min(), B0 + B1*x.max()], alpha =0.7, color = 'black', linewidth = 2, label='Theil-Sen Slope')

    # def norm0_1(x):
    #     return (x - x.min())/ (x.max()-x.min())
    
    # x_norm = 2*norm0_1(x)-1

    # res = stats.theilslopes(y, x_norm, alpha=0.05)
    
    # ax.plot([x_plot.min(), x_plot.max()], [res[1] + res[2] * x_norm.min(), res[1] + res[2] * x_norm.max()], 'k--', alpha=0.7)
    # ax.plot([x_plot.min(), x_plot.max()], [res[1] + res[3] * x_norm.min(), res[1] + res[3] * x_norm.max()], 'k--', alpha=0.7)
    
    
      
    ax.set_ylim(4, 7) 
    
    ax.axhspan(0,2, color = 'lightgray', alpha = 0.3)
    
    ax.set_ylabel('Filtered DO [mg/L]')
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)

    ax.set_xlabel('')
    
    plt.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_q50DO_trend_1.png', bbox_inches='tight', dpi=500, transparent=True)


for site in ['point_jefferson']:
    
    fig, ax = plt.subplots(figsize = (7,3))
    
    plot_df_q50 = odf_use_q50[(odf_use_q50['site'] == site) & (odf_use_q50['var'] == 'DO_mg_L') & (odf_use_q50['surf_deep'] == 'deep')]

    sns.scatterplot(data=plot_df_q50, x='datetime', y = 'val',  ax=ax, color = '#ff4040')
    
    stat_df = all_stats_filt[(all_stats_filt['site'] == site) & (all_stats_filt['var'] == 'deep_DO_mg_L') & (all_stats_filt['deep_DO_q'] == 'deep_DO_q50') & (all_stats_filt['summer_non_summer'] == 'summer')]
    
    
    
    x = plot_df_q50['date_ordinal']

    y = plot_df_q50['val']

    x_plot = plot_df_q50['datetime']
     
    B0 = stat_df['B0'].iloc[0]
    
    B1 = stat_df['B1'].iloc[0]
    
    ax.plot([x_plot.min(), x_plot.max()], [B0 + B1*x.min(), B0 + B1*x.max()], alpha =0.7, color = 'black', linewidth = 2, label='Theil-Sen Slope')

    def norm0_1(x):
        return (x - x.min())/ (x.max()-x.min())
    
    x_norm = 2*norm0_1(x)-1

    res = stats.theilslopes(y, x_norm, alpha=0.05)
    
    ax.plot([x_plot.min(), x_plot.max()], [res[1] + res[2] * x_norm.min(), res[1] + res[2] * x_norm.max()], 'k--', alpha=0.7)
    ax.plot([x_plot.min(), x_plot.max()], [res[1] + res[3] * x_norm.min(), res[1] + res[3] * x_norm.max()], 'k--', alpha=0.7)
    
    
      
    ax.set_ylim(4, 7) 
    
    ax.axhspan(0,2, color = 'lightgray', alpha = 0.3)
    
    ax.set_ylabel('Filtered DO [mg/L]')
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)

    ax.set_xlabel('')
    
    plt.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_q50DO_trend_2.png', bbox_inches='tight', dpi=500, transparent=True)

for site in ['point_jefferson']:
    
    fig, ax = plt.subplots(figsize = (7,3))
    
    plot_df_q50 = odf_use_q50[(odf_use_q50['site'] == site) & (odf_use_q50['var'] == 'DO_mg_L') & (odf_use_q50['surf_deep'] == 'deep')]

    sns.scatterplot(data=plot_df_q50, x='datetime', y = 'val',  ax=ax, color = '#ff4040')
    
    stat_df = all_stats_filt[(all_stats_filt['site'] == site) & (all_stats_filt['var'] == 'deep_DO_mg_L') & (all_stats_filt['deep_DO_q'] == 'deep_DO_q50') & (all_stats_filt['summer_non_summer'] == 'summer')]
    
    
    
    x = plot_df_q50['date_ordinal']

    y = plot_df_q50['val']

    x_plot = plot_df_q50['datetime']
     
    B0 = stat_df['B0'].iloc[0]
    
    B1 = stat_df['B1'].iloc[0]
    
    ax.axhline(np.mean([B0 + B1*x.min(), B0 + B1*x.max()]), color = 'gray', linestyle = '--', alpha = 0.5)

    
    ax.plot([x_plot.min(), x_plot.max()], [B0 + B1*x.min(), B0 + B1*x.max()], alpha =0.7, color = 'black', linewidth = 2, label='Theil-Sen Slope')

    def norm0_1(x):
        return (x - x.min())/ (x.max()-x.min())
    
    x_norm = 2*norm0_1(x)-1

    res = stats.theilslopes(y, x_norm, alpha=0.05)
    
    ax.plot([x_plot.min(), x_plot.max()], [res[1] + res[2] * x_norm.min(), res[1] + res[2] * x_norm.max()], 'k--', alpha=0.7)
    ax.plot([x_plot.min(), x_plot.max()], [res[1] + res[3] * x_norm.min(), res[1] + res[3] * x_norm.max()], 'k--', alpha=0.7)
          
    ax.set_ylim(4, 7) 
    
    ax.axhspan(0,2, color = 'lightgray', alpha = 0.3)
    
    ax.set_ylabel('Filtered DO [mg/L]')
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)

    ax.set_xlabel('')
    
    plt.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_q50DO_trend_3.png', bbox_inches='tight', dpi=500, transparent=True)
