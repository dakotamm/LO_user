#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 16:37:23 2025

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

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_his', 'kc_whidbeyBasin', 'nceiSalish', 'kc_pointJefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)


# %%

site_list =  odf['site'].unique()




odf_use = odf_depth_mean.copy()




odf_calc_use = odf_calc_long.copy()

all_stats_filt = dfun.buildStatsDF(odf_use, site_list, odf_calc_use=odf_calc_use, odf_depth_mean_deep_DO_percentiles=odf_depth_mean_deep_DO_percentiles)


# %%

# c=0

all_stats_filt.loc[all_stats_filt['site'] == 'point_jefferson', 'site_label'] = 'PJ'

all_stats_filt.loc[all_stats_filt['site'] == 'near_seattle_offshore', 'site_label'] = 'NS'

all_stats_filt.loc[all_stats_filt['site'] == 'saratoga_passage_mid', 'site_label'] = 'SP'

all_stats_filt.loc[all_stats_filt['site'] == 'carr_inlet_mid', 'site_label'] = 'CI'

all_stats_filt.loc[all_stats_filt['site'] == 'lynch_cove_mid', 'site_label'] = 'LC'


all_stats_filt.loc[all_stats_filt['site'] == 'point_jefferson', 'site_type'] = 'Main Basin'

all_stats_filt.loc[all_stats_filt['site'] == 'near_seattle_offshore', 'site_type'] = 'Main Basin'

all_stats_filt.loc[all_stats_filt['site'] == 'saratoga_passage_mid', 'site_type'] = 'Sub-Basins'

all_stats_filt.loc[all_stats_filt['site'] == 'carr_inlet_mid', 'site_type'] = 'Sub-Basins'

all_stats_filt.loc[all_stats_filt['site'] == 'lynch_cove_mid', 'site_type'] = 'Sub-Basins'


all_stats_filt.loc[all_stats_filt['site'] == 'point_jefferson', 'site_num'] = 1

all_stats_filt.loc[all_stats_filt['site'] == 'near_seattle_offshore', 'site_num'] = 2

all_stats_filt.loc[all_stats_filt['site'] == 'saratoga_passage_mid', 'site_num'] = 4

all_stats_filt.loc[all_stats_filt['site'] == 'carr_inlet_mid', 'site_num'] = 3

all_stats_filt.loc[all_stats_filt['site'] == 'lynch_cove_mid', 'site_num'] = 5

# %%

all_stats_filt.loc[all_stats_filt['var'].isin(['surf_CT', 'surf_SA', 'surf_DO_mg_L']), 'surf_deep'] = 'surf'

all_stats_filt.loc[all_stats_filt['var'].isin(['deep_CT', 'deep_SA', 'deep_DO_mg_L']), 'surf_deep'] = 'deep'

all_stats_filt.loc[all_stats_filt['var'].isin(['surf_CT', 'deep_CT']), 'var'] = 'CT'

all_stats_filt.loc[all_stats_filt['var'].isin(['surf_SA', 'deep_SA']), 'var'] = 'SA'

all_stats_filt.loc[all_stats_filt['var'].isin(['surf_DO_mg_L', 'deep_DO_mg_L']), 'var'] = 'DO_mg_L'

# %%

all_stats_filt.loc[all_stats_filt['surf_deep'] == 'surf', 'depth_label'] = 'Surface'

all_stats_filt.loc[all_stats_filt['surf_deep'] == 'deep', 'depth_label'] = 'Bottom'

all_stats_filt.loc[all_stats_filt['var'] == 'CT', 'var_label'] = '[degC]'

all_stats_filt.loc[all_stats_filt['var'] == 'SA', 'var_label'] = '[g/kg]'

all_stats_filt.loc[all_stats_filt['var'] == 'DO_mg_L', 'var_label'] = '[mg/L]'

all_stats_filt.loc[all_stats_filt['season'] == 'grow', 'season_label'] = 'Apr-Jul (spring)'

all_stats_filt.loc[all_stats_filt['season'] == 'loDO', 'season_label'] = 'Aug-Nov (low-DO)'

all_stats_filt.loc[all_stats_filt['season'] == 'winter', 'season_label'] = 'Dec-Mar (winter)'



# %%

#markers = {'surf': '^', 'deep': 'v'}

#palette = {'surf': '#bddf26', 'deep': '#482173'}

#palette = {'Surface': '#bddf26', 'Bottom': '#482173'}

palette = {'Surface': 'white', 'Bottom': 'gray'}


#palette = {'point_jefferson': 'red', 'near_seattle_offshore': 'orange', 'carr_inlet_mid':'blue', 'saratoga_passage_mid':'purple', 'lynch_cove_mid': 'orchid'}

linecolors = {'Main Basin':'#e04256', 'Sub-Basins':'#4565e8'}

#linecolors = {'point_jefferson': '#e04256', 'near_seattle_offshore': '#e04256', 'carr_inlet_mid':'#4565e8', 'saratoga_passage_mid':'#4565e8', 'lynch_cove_mid': '#4565e8'}

#edgecolors = {'point_jefferson': 'k', 'near_seattle_offshore': 'k', 'carr_inlet_mid':'gray', 'saratoga_passage_mid':'gray', 'lynch_cove_mid': 'gray'}

jitter = {'Surface': -0.1, 'Bottom': 0.1}

markers = {'DO_mg_L': 'o', 'SA': 's', 'CT': '^'}
 

mosaic = [['Dec-Mar (winter)', 'Apr-Jul (spring)', 'Aug-Nov (low-DO)']]
 
#fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=True, figsize=(9,3), layout='constrained', gridspec_kw=dict(wspace=0.1, hspace=0.1))

for var in ['DO_mg_L', 'CT', 'SA']:
    
    fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=True, figsize=(9,2.5), layout='constrained', gridspec_kw=dict(wspace=0.1, hspace=0.1))
    
    for season in ['Dec-Mar (winter)', 'Apr-Jul (spring)', 'Aug-Nov (low-DO)']:
        
        ax_name = season
    
        ax = axd[ax_name]
        
        for site in long_site_list:
            
            for depth in ['Surface', 'Bottom']:
                
                plot_df = all_stats_filt[(all_stats_filt['var'] == var) & (all_stats_filt['season_label'] == season) & (all_stats_filt['site'] == site) & (all_stats_filt['depth_label'] == depth)]
                
                plot_df['slope_datetime_cent'] = plot_df['slope_datetime']*100
                 
                plot_df['slope_datetime_cent_95hi'] = plot_df['slope_datetime_s_hi']*100
                
                plot_df['slope_datetime_cent_95lo'] = plot_df['slope_datetime_s_lo']*100
                
                # ax.scatter(plot_df['site_num'] + jitter[depth], plot_df['slope_datetime_cent'], color=palette[depth], marker=markers[var], s=50, label = depth)
                                
                ax.scatter(plot_df['site_num'] + jitter[depth], plot_df['slope_datetime_cent'], color=palette[depth], edgecolors='k', marker=markers[var], s=20, label=depth) #, marker=markers[depth], edgecolors=edgecolors[site])
                
                ax.plot([plot_df['site_num'] + jitter[depth], plot_df['site_num'] + jitter[depth]],[plot_df['slope_datetime_cent_95lo'], plot_df['slope_datetime_cent_95hi']], color=linecolors[plot_df['site_type'].iloc[0]], alpha =1, zorder = -5, linewidth=1, label=plot_df['site_type'].iloc[0])
        
        ax.text(0.05,0.95, season, transform=ax.transAxes, verticalalignment='top', fontweight = 'bold', color='k')
         
        ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3, zorder = -6)
    
        ax.axhline(0, color='gray', linestyle = '--', zorder = -5) 
    
        ax.set_ylabel(all_stats_filt[all_stats_filt['var'] == var]['var_label'].iloc[0] + '/cent.')
        
        ax.set_xticks([1,2,3,4,5],['PJ', 'NS', 'CI', 'SP', 'LC'])
        
        if season == 'Dec-Mar (winter)':
            
            handles, labels = ax.get_legend_handles_labels()
                
            selected_handles = handles[:2]
            selected_labels = labels[:2]
            
            ax.legend(selected_handles, selected_labels, loc='lower left')
            
            #ax.legend()

        
        #ax.axvline(2.5, color = 'gray', linestyle = '--', zorder = -5)
    
    # if var == 'CT': 
        
    #     ax.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left')
        
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + var +'_trends.png', dpi=500, transparent=True)

# # %%

# markers = {'surf': '^', 'deep': 'v'}
# palette = {'point_jefferson': 'red', 'near_seattle_offshore': 'orange', 'carr_inlet_mid':'blue', 'saratoga_passage_mid':'purple', 'lynch_cove_mid': 'orchid'}

# mosaic = [['CT'], ['SA'], ['DO_mg_L']]

# fig, axd = plt.subplot_mosaic(mosaic, sharex=True, figsize=(6,9), layout='constrained', gridspec_kw=dict(wspace=0.1, hspace=0.1))

# for var in var_list:
    
#     ax = axd[var]
     
#     for depth in ['surf', 'deep']:
            
#         for site in ['point_jefferson', 'carr_inlet_mid', 'saratoga_passage_mid', 'lynch_cove_mid']:
            
#             plot_df = all_stats_filt[(all_stats_filt['var'] == var) & (all_stats_filt['season'] != 'allyear') & (all_stats_filt['site'] == site) & (all_stats_filt['surf_deep'] == depth)]
    
#             ax.scatter(plot_df['season'], plot_df['slope_datetime']*100, color=palette[site], marker=markers[depth], alpha=0.5, label = depth + '_' + site, s=50)
    
#     ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
    
#     ax.axhline(0, color='gray', linestyle = '--', zorder = -5)
    
#     ax.set_ylabel(var + '/cent.')
    
#     if var == 'CT':
        
#         ax.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left')
        
    
# plt.savefig('/Users/dakotamascarenas/Desktop/pltz/long_sites_trend_comp_new_no_ns.png', dpi=500)


