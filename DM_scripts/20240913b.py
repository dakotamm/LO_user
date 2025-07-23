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


# %%

for site in ['point_jefferson']:


    mosaic = [['map_source', 'annual_min_DO', 'annual_min_DO', 'annual_min_DO'], ['map_source', 'min_DO_yearday', 'min_DO_yearday', 'min_DO_yearday']] #, ['map_source', '.', '.'],]
    
    fig, ax = plt.subplot_mosaic(mosaic, figsize=(11,5), layout='constrained')
    
    plot_df = odf[odf['site'] == site].groupby(['site','cid']).first().reset_index()
        
    sns.scatterplot(data=plot_df[plot_df['site'] == site], x='lon', y='lat', ax = ax['map_source'], color = '#ff4040', alpha=0.3, legend=False)
    
    pfun.add_coast(ax['map_source'])
    
    pfun.dar(ax['map_source'])
    
    ax['map_source'].set_xlim(-123.2, -122.1)
    
    ax['map_source'].set_ylim(47,48.5)
    
    #ax['map_source'].legend(loc='upper center', title ='Data Source') #, bbox_to_anchor=(0.5, -0.1), title='Data Source')
    
    ax['map_source'].set_xlabel('')
    
    ax['map_source'].set_ylabel('')
    
    ax['map_source'].tick_params(axis='x', labelrotation=45)

    
    plot_df = odf_use_DO_min_deep[odf_use_DO_min_deep['site'] == site]
        
    sns.scatterplot(data=plot_df, x='year', y = 'val',  ax=ax['annual_min_DO'], color = 'gray')
        
    sns.scatterplot(data=plot_df, x='year', y = 'yearday',  ax=ax['min_DO_yearday'], color = 'gray')
    
    
    ax['annual_min_DO'].set_ylim(0, 10)
    
    ax['annual_min_DO'].axhspan(0,2, color = 'lightgray', alpha = 0.4)
    
    ax['annual_min_DO'].set_ylabel('Annual Min. [DO] [mg/L]')
    
    ax['min_DO_yearday'].set_ylim(0,366)
    
    ax['min_DO_yearday'].set_ylabel('Min. [DO] Yearday')
    
    ax['min_DO_yearday'].axhspan(low_DO_season_start, low_DO_season_end, color = 'lightgray', alpha = 0.4, zorder=-5)

    
    ax['annual_min_DO'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax['min_DO_yearday'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax['min_DO_yearday'].set_xlabel('')
    
    
    ax['annual_min_DO'].set_xlabel('')


    ax['min_DO_yearday'].set_xlabel('')



    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_min_DO_study_deep_PRESENT.png', bbox_inches='tight', dpi=500, transparent=True)
    
# %%

for site in ['point_jefferson']:


    mosaic = [['map_source', 'cast_yearday', 'cast_yearday', 'cast_yearday'], ['map_source', 'cast_min_DO', 'cast_min_DO', 'cast_min_DO']] #, ['map_source', '.', '.'],]
    
    fig, ax = plt.subplot_mosaic(mosaic, figsize=(11,5), layout='constrained')
    
    #plot_df = odf[odf['site'] == site].groupby(['site','cid']).first().reset_index()
            
    ax['map_source'].set_xlim(X[i1],-121.4)#X[i2]) # Salish Sea
    ax['map_source'].set_ylim(Y[j1],Y[j2]) # Salish Sea
        
    ax['map_source'].pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-100, vmax=0, cmap = 'gray', zorder=-5)
    
    pfun.add_coast(ax['map_source'])
    
    pfun.dar(ax['map_source'])
    
    for site_ in long_site_list:
    
        path = path_dict[site]
    
        
    if site in ['point_jefferson']:
    
    
        patch = patches.PathPatch(path, facecolor='#ff4040', edgecolor='white', zorder=1)#, label='>60-year history')
        
    else:
        
        patch = patches.PathPatch(path, facecolor='gray', edgecolor='white', zorder=1)
        
    ax['map_source'].add_patch(patch) 
    
    pfun.add_coast(ax['map_source'])
    
    pfun.dar(ax['map_source'])
    
    ax['map_source'].set_xlim(-123.2, -122.1)
    
    ax['map_source'].set_ylim(47,48.5)
        
    ax['map_source'].set_xlabel('')
    
    ax['map_source'].set_ylabel('')
    
    ax['map_source'].tick_params(axis='x', labelrotation=45)

    
    plot_df_full = odf_use_full[(odf_use_full['site'] == site) & (odf_use_full['var'] == 'DO_mg_L') & (odf_use_full['surf_deep'] == 'deep')]
    
    plot_df_AugNov = odf_use_AugNov[(odf_use_AugNov['site'] == site) & (odf_use_AugNov['var'] == 'DO_mg_L') & (odf_use_AugNov['surf_deep'] == 'deep')]
    
    
    plot_df_q50 = odf_use_q50[(odf_use_q50['site'] == site) & (odf_use_q50['var'] == 'DO_mg_L') & (odf_use_q50['surf_deep'] == 'deep')]


        
    sns.scatterplot(data=plot_df_full, x='year', y = 'val',  ax=ax['cast_min_DO'], color = 'gray')
    
    sns.scatterplot(data=plot_df_AugNov, x='year', y = 'val',  ax=ax['cast_min_DO'], color = 'black')
    
    
    sns.scatterplot(data=plot_df_q50, x='year', y = 'val',  ax=ax['cast_min_DO'], color = '#ff4040')

    
    
    sns.scatterplot(data=plot_df_full, x='year', y = 'yearday',  ax=ax['cast_yearday'], color = 'gray')
    
    sns.scatterplot(data=plot_df_AugNov, x='year', y = 'yearday',  ax=ax['cast_yearday'], color = 'black')
    
    
    sns.scatterplot(data=plot_df_q50, x='year', y = 'yearday',  ax=ax['cast_yearday'], color = '#ff4040')
    

    




    
    ax['cast_min_DO'].set_ylim(0, 15)
    
    ax['cast_min_DO'].axhspan(0,2, color = 'lightgray', alpha = 0.2)
    
    ax['cast_min_DO'].set_ylabel('Cast Deep DO [mg/L]')
    
    ax['cast_yearday'].set_ylim(0,366)
    
    ax['cast_yearday'].set_ylabel('Cast Yearday')
    
    ax['cast_yearday'].axhspan(low_DO_season_start, low_DO_season_end, color = 'lightgray', alpha = 0.4, zorder=-5)

    

    
    ax['cast_min_DO'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax['cast_yearday'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax['cast_min_DO'].set_xlabel('')


    ax['cast_yearday'].set_xlabel('')




    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_percentile_DO_study_PRESENT.png', bbox_inches='tight', dpi=500, transparent=True)
    
# %%

for site in ['point_jefferson']:


    mosaic = [['map_source', 'cast_yearday', 'cast_yearday', 'cast_yearday'], ['map_source', 'cast_min_DO', 'cast_min_DO', 'cast_min_DO']] #, ['map_source', '.', '.'],]
    
    fig, ax = plt.subplot_mosaic(mosaic, figsize=(11,5), layout='constrained')
    
    #plot_df = odf[odf['site'] == site].groupby(['site','cid']).first().reset_index()
        
    ax['map_source'].set_xlim(X[i1],-121.4)#X[i2]) # Salish Sea
    ax['map_source'].set_ylim(Y[j1],Y[j2]) # Salish Sea
        
    ax['map_source'].pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-100, vmax=0, cmap = 'gray', zorder=-5)
    
    pfun.add_coast(ax['map_source'])
    
    pfun.dar(ax['map_source'])
    
    for site_ in long_site_list:
    
        path = path_dict[site]
    
        
    if site in ['point_jefferson']:
    
    
        patch = patches.PathPatch(path, facecolor='#ff4040', edgecolor='white', zorder=1)#, label='>60-year history')
        
    else:
        
        patch = patches.PathPatch(path, facecolor='gray', edgecolor='white', zorder=1)
        
    ax['map_source'].add_patch(patch) 
    
    pfun.add_coast(ax['map_source'])
    
    pfun.dar(ax['map_source'])
    
    ax['map_source'].set_xlim(-123.2, -122.1)
    
    ax['map_source'].set_ylim(47,48.5)
        
    ax['map_source'].set_xlabel('')
    
    ax['map_source'].set_ylabel('')
    
    ax['map_source'].tick_params(axis='x', labelrotation=45)

    
    plot_df_full = odf_use_full[(odf_use_full['site'] == site) & (odf_use_full['var'] == 'DO_mg_L') & (odf_use_full['surf_deep'] == 'deep')]
    
    plot_df_AugNov = odf_use_AugNov[(odf_use_AugNov['site'] == site) & (odf_use_AugNov['var'] == 'DO_mg_L') & (odf_use_AugNov['surf_deep'] == 'deep')]
    
    
    plot_df_q50 = odf_use_q50[(odf_use_q50['site'] == site) & (odf_use_q50['var'] == 'DO_mg_L') & (odf_use_q50['surf_deep'] == 'deep')]


        
    sns.scatterplot(data=plot_df_full, x='year', y = 'val',  ax=ax['cast_min_DO'], color = 'gray')
    
    # sns.scatterplot(data=plot_df_AugNov, x='year', y = 'val',  ax=ax['cast_min_DO'], color = 'black')
    
    
    # sns.scatterplot(data=plot_df_q50, x='year', y = 'val',  ax=ax['cast_min_DO'], color = '#ff4040')

    
    
    sns.scatterplot(data=plot_df_full, x='year', y = 'yearday',  ax=ax['cast_yearday'], color = 'gray')
    
    # sns.scatterplot(data=plot_df_AugNov, x='year', y = 'yearday',  ax=ax['cast_yearday'], color = 'black')
    
    
    # sns.scatterplot(data=plot_df_q50, x='year', y = 'yearday',  ax=ax['cast_yearday'], color = '#ff4040')
    

    




    
    ax['cast_min_DO'].set_ylim(0, 15)
    
    ax['cast_min_DO'].axhspan(0,2, color = 'lightgray', alpha = 0.2)
    
    ax['cast_min_DO'].set_ylabel('Cast Deep DO [mg/L]')
    
    ax['cast_yearday'].set_ylim(0,366)
    
    ax['cast_yearday'].set_ylabel('Cast Yearday')
    
    ax['cast_yearday'].axhspan(low_DO_season_start, low_DO_season_end, color = 'lightgray', alpha = 0.4, zorder=-5)

    

    
    ax['cast_min_DO'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax['cast_yearday'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax['cast_min_DO'].set_xlabel('')


    ax['cast_yearday'].set_xlabel('')




    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_percentile_DO_study_PRESENT_0.png', bbox_inches='tight', dpi=500, transparent=True)
    
# %%

for site in ['point_jefferson']:


    mosaic = [['map_source', 'cast_yearday', 'cast_yearday', 'cast_yearday'], ['map_source', 'cast_min_DO', 'cast_min_DO', 'cast_min_DO']] #, ['map_source', '.', '.'],]
    
    fig, ax = plt.subplot_mosaic(mosaic, figsize=(11,5), layout='constrained')
    
    #plot_df = odf[odf['site'] == site].groupby(['site','cid']).first().reset_index()
        
    ax['map_source'].set_xlim(X[i1],-121.4)#X[i2]) # Salish Sea
    ax['map_source'].set_ylim(Y[j1],Y[j2]) # Salish Sea
        
    ax['map_source'].pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-100, vmax=0, cmap = 'gray', zorder=-5)
    
    pfun.add_coast(ax['map_source'])
    
    pfun.dar(ax['map_source'])
    
    for site_ in long_site_list:
    
        path = path_dict[site]
    
        
    if site in ['point_jefferson']:
    
    
        patch = patches.PathPatch(path, facecolor='#ff4040', edgecolor='white', zorder=1)#, label='>60-year history')
        
    else:
        
        patch = patches.PathPatch(path, facecolor='gray', edgecolor='white', zorder=1)
        
    ax['map_source'].add_patch(patch) 
    
    pfun.add_coast(ax['map_source'])
    
    pfun.dar(ax['map_source'])
    
    ax['map_source'].set_xlim(-123.2, -122.1)
    
    ax['map_source'].set_ylim(47,48.5)
        
    ax['map_source'].set_xlabel('')
    
    ax['map_source'].set_ylabel('')
    
    ax['map_source'].tick_params(axis='x', labelrotation=45)

    
    plot_df_full = odf_use_full[(odf_use_full['site'] == site) & (odf_use_full['var'] == 'DO_mg_L') & (odf_use_full['surf_deep'] == 'deep')]
    
    plot_df_AugNov = odf_use_AugNov[(odf_use_AugNov['site'] == site) & (odf_use_AugNov['var'] == 'DO_mg_L') & (odf_use_AugNov['surf_deep'] == 'deep')]
    
    
    plot_df_q50 = odf_use_q50[(odf_use_q50['site'] == site) & (odf_use_q50['var'] == 'DO_mg_L') & (odf_use_q50['surf_deep'] == 'deep')]


        
    sns.scatterplot(data=plot_df_full, x='year', y = 'val',  ax=ax['cast_min_DO'], color = 'gray')
    
    sns.scatterplot(data=plot_df_AugNov, x='year', y = 'val',  ax=ax['cast_min_DO'], color = 'black')
    
    
    # sns.scatterplot(data=plot_df_q50, x='year', y = 'val',  ax=ax['cast_min_DO'], color = '#ff4040')

    
    
    sns.scatterplot(data=plot_df_full, x='year', y = 'yearday',  ax=ax['cast_yearday'], color = 'gray')
    
    sns.scatterplot(data=plot_df_AugNov, x='year', y = 'yearday',  ax=ax['cast_yearday'], color = 'black')
    
    
    # sns.scatterplot(data=plot_df_q50, x='year', y = 'yearday',  ax=ax['cast_yearday'], color = '#ff4040')
    

    




    
    ax['cast_min_DO'].set_ylim(0, 15)
    
    ax['cast_min_DO'].axhspan(0,2, color = 'lightgray', alpha = 0.2)
    
    ax['cast_min_DO'].set_ylabel('Cast Deep DO [mg/L]')
    
    ax['cast_yearday'].set_ylim(0,366)
    
    ax['cast_yearday'].set_ylabel('Cast Yearday')
    
    ax['cast_yearday'].axhspan(low_DO_season_start, low_DO_season_end, color = 'lightgray', alpha = 0.4, zorder=-5)

    

    
    ax['cast_min_DO'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax['cast_yearday'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax['cast_min_DO'].set_xlabel('')


    ax['cast_yearday'].set_xlabel('')




    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_percentile_DO_study_PRESENT_1.png', bbox_inches='tight', dpi=500, transparent=True)