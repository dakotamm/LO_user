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




#poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson', 'mb', 'hc', 'ss', 'wb'] # 5 sites + 4 basins

poly_list = ['point_jefferson']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)


# %%

odf_use = odf_depth_mean_deep_DO_percentiles.copy()

odf_use_AugNov = odf_use[odf_use['month'].isin([8,9,10,11])]

odf_use_AugNov_q50 = odf_use_AugNov[odf_use_AugNov['val'] <= odf_use_AugNov['deep_DO_q50']]




# %%
    
# %%

for site in ['point_jefferson']:


    mosaic = [['map_source', 'cast_yearday', 'cast_yearday'], ['map_source', 'cast_deep_DO', 'cast_deep_DO']] #, ['map_source', '.', '.'],]
    
    fig, ax = plt.subplot_mosaic(mosaic, figsize=(9,5), layout='constrained')

    ax['map_source'].text(0.05,0.025, 'a', transform=ax['map_source'].transAxes, fontsize=14, fontweight='bold', color = 'k')
            
        
    ax['map_source'].pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-100, vmax=0, cmap = 'gray', zorder=-5)

    
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
    
    
    ax['map_source'].set_xticks([-123.0, -122.6, -122.2], ['-123.0','-122.6', '-122.2']) #['','-123.0', '', '-122.6', '', '-122.2'])
        
    ax['map_source'].set_xlabel('')
    
    ax['map_source'].set_ylabel('')
    
    
    



        
    sns.scatterplot(data=odf_use, x='year', y = 'val',  ax=ax['cast_deep_DO'], color = 'gray', label='all cast values')
    
    sns.scatterplot(data=odf_use_AugNov, x='year', y = 'val',  ax=ax['cast_deep_DO'], color = 'black', label='Aug-Nov cast values')
    
    
    sns.scatterplot(data=odf_use_AugNov_q50, x='year', y = 'val',  ax=ax['cast_deep_DO'], color = '#ff4040', label='Aug-Nov cast values <= 50th percentile')

    
    
    sns.scatterplot(data=odf_use, x='year', y = 'yearday',  ax=ax['cast_yearday'], color = 'gray')
    
    sns.scatterplot(data=odf_use_AugNov, x='year', y = 'yearday',  ax=ax['cast_yearday'], color = 'black')
    
    
    sns.scatterplot(data=odf_use_AugNov_q50, x='year', y = 'yearday',  ax=ax['cast_yearday'], color = '#ff4040')
    

    


    ax['cast_yearday'].text(0.025,0.05, 'b', transform=ax['cast_yearday'].transAxes, fontsize=14, fontweight='bold', color = 'k')

    ax['cast_deep_DO'].text(0.025,0.05, 'c', transform=ax['cast_deep_DO'].transAxes, fontsize=14, fontweight='bold', color = 'k')

    
    ax['cast_deep_DO'].set_ylim(0, 12)
    
    ax['cast_deep_DO'].axhspan(0,2, color = 'lightgray', alpha = 0.2)
    
    ax['cast_deep_DO'].set_ylabel('Cast Deep DO [mg/L]')
    
    ax['cast_yearday'].set_ylim(0,366)
    
    ax['cast_yearday'].set_ylabel('Cast Yearday')
    
    ax['cast_yearday'].axhspan(213, 335, color = 'lightgray', alpha = 0.4, zorder=-5, label='Aug-Nov')

    

    
    ax['cast_deep_DO'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax['cast_yearday'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax['cast_deep_DO'].set_xlabel('')


    ax['cast_yearday'].set_xlabel('')




    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/paper_1_fig_5.png', bbox_inches='tight', dpi=500, transparent=True)
    
