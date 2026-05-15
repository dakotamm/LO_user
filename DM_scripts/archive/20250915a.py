#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 15:02:13 2025

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

from pygam import LinearGAM, s

import matplotlib.patheffects as pe

# %%




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




poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson'] #, 'mb', 'hc', 'ss', 'wb'] # 5 sites + 4 basins

#poly_list = ['ps']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_his', 'kc_whidbeyBasin', 'nceiSalish', 'kc_pointJefferson '], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)

# %%

odf_use_seasonal_DO, odf_use_seasonal_CTSA, odf_use_annual_DO, odf_use_annual_CTSA = dfun.calcSeriesAvgs(odf_depth_mean, odf_depth_mean_deep_DO_percentiles, deep_DO_q = 'deep_DO_q50')

# %%

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'point_jefferson', 'site_label'] = 'PJ'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'near_seattle_offshore', 'site_label'] = 'NS'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'saratoga_passage_mid', 'site_label'] = 'SP'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'carr_inlet_mid', 'site_label'] = 'CI'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'lynch_cove_mid', 'site_label'] = 'LC'


odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'point_jefferson', 'site_type'] = 'Main Basin'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'near_seattle_offshore', 'site_type'] = 'Main Basin'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'saratoga_passage_mid', 'site_type'] = 'Sub-Basins'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'carr_inlet_mid', 'site_type'] = 'Sub-Basins'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'lynch_cove_mid', 'site_type'] = 'Sub-Basins'


odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'point_jefferson', 'site_num'] = 1

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'near_seattle_offshore', 'site_num'] = 2

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'saratoga_passage_mid', 'site_num'] = 4

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'carr_inlet_mid', 'site_num'] = 3

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'lynch_cove_mid', 'site_num'] = 5

# %%

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'point_jefferson', 'site_label'] = 'PJ'

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'near_seattle_offshore', 'site_label'] = 'NS'

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'saratoga_passage_mid', 'site_label'] = 'SP'

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'carr_inlet_mid', 'site_label'] = 'CI'

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'lynch_cove_mid', 'site_label'] = 'LC'


odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'point_jefferson', 'site_type'] = 'Main Basin'

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'near_seattle_offshore', 'site_type'] = 'Main Basin'

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'saratoga_passage_mid', 'site_type'] = 'Sub-Basins'

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'carr_inlet_mid', 'site_type'] = 'Sub-Basins'

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'lynch_cove_mid', 'site_type'] = 'Sub-Basins'


odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'point_jefferson', 'site_num'] = 1

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'near_seattle_offshore', 'site_num'] = 2

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'saratoga_passage_mid', 'site_num'] = 4

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'carr_inlet_mid', 'site_num'] = 3

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'lynch_cove_mid', 'site_num'] = 5



# %%

edgecolors = {'surf': 'white', 'deep': 'k'}

palette = {'grow': '#dd9404', 'loDO': '#e04256', 'winter': '#4565e8'}

markers  = {'DO_mg_L': 'o', 'CT':'D', 'SA':'s'}

#palette = {'point_jefferson': '#e04256', 'near_seattle_offshore': '#e04256', 'carr_inlet_mid':'#4565e8', 'saratoga_passage_mid':'#4565e8', 'lynch_cove_mid': '#4565e8'}

#palette = {'point_jefferson': '#e04256', 'near_seattle_offshore': '#e04256', 'carr_inlet_mid':'#4565e8', 'saratoga_passage_mid':'#4565e8', 'lynch_cove_mid': '#4565e8'}

ymins = {'DO_mg_L': 0, 'CT': 7, 'SA': 20}

# ymaxs = {'DO_mg_L': 0, 'CT': 7, 'SA': 20}



jitter = {'winter': -0.1, 'grow':0, 'loDO': 0.1}

mosaic = [['DO_mg_L', 'CT', 'SA']]


#mosaic = [['surf_DO_mg_L', 'surf_CT', 'surf_SA'], ['deep_DO_mg_L', 'deep_CT', 'deep_SA']]

fig, axd = plt.subplot_mosaic(mosaic, sharex=True, figsize=(9,2.5), layout='constrained', gridspec_kw=dict(wspace=0.1, hspace=0.1))

for var in var_list:
    
    ax_name = var
    
    ax = axd[ax_name]
    
    for depth in [ 'surf', 'deep']: 
        
        for site in long_site_list:
            
            for season in ['winter', 'grow', 'loDO']:
                
                plot_df = odf_use_seasonal_CTSA[(odf_use_seasonal_CTSA['var'] == var) & (odf_use_seasonal_CTSA['season'] == season) & (odf_use_seasonal_CTSA['site'] == site) & (odf_use_seasonal_CTSA['surf_deep'] == depth)]

                ax.scatter(plot_df['site_num'] + jitter[season], plot_df['val_mean'], color=palette[season], s=20, marker= markers[var], edgecolors=edgecolors[depth])
                 
                ax.plot([plot_df['site_num'] + jitter[season], plot_df['site_num'] + jitter[season]],[plot_df['val_ci95lo'], plot_df['val_ci95hi']], color=palette[season], alpha =0.5, zorder = -5, linewidth=1)
              
            # plot_df_ = odf_use_annual_CTSA[(odf_use_annual_CTSA['var'] == var) & (odf_use_annual_CTSA['site'] == site) & (odf_use_annual_CTSA['surf_deep'] == depth)]
            
            # ax.scatter(plot_df_['site_num'], plot_df_['val_mean'], color='gray', s=20, marker= markers[var], edgecolors=edgecolors[depth])
             
            # ax.plot([plot_df_['site_num'], plot_df_['site_num']],[plot_df_['val_ci95lo'], plot_df_['val_ci95hi']], color='gray', alpha =0.5, zorder = -5, linewidth=1)
            
                
    if var == 'DO_mg_L':  
    
        ax.axhspan(0,2, color = 'gray', alpha = 0.3, zorder=-6, label='Hypoxia') 
        
        ax.set_ylabel('Mean [DO] [mg/L]')
    
    elif var == 'CT':
        
        ax.set_ylabel('Mean Temperature [degC]')
        
    else:
        
        ax.set_ylabel('Mean Salinity [g/kg]')

    
    #ax.text(0.05,0.95, var, transform=ax.transAxes, verticalalignment='top', fontweight = 'bold', color='k')
     
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3, zorder = -6)

    ax.axhline(0, color='gray', linestyle = '--', zorder = -5) 

    #ax.set_ylabel(var)
     
    ax.set_ylim(ymin=ymins[var]) 
    
    ax.set_xticks([1,2,3,4,5],['PJ', 'NS', 'CI', 'SP', 'LC'])
        
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/seasonal_timeseries_avgs.png', dpi=500, transparent=True)