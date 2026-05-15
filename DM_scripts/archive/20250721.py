#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 13:46:42 2025

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




poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson'] # 5 sites + 4 basins

#poly_list = ['ps']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)

# %%

site_list =  odf['site'].unique()

stat_list = ['mk_ts','linreg']




odf_use = odf_depth_mean.copy()

odf_calc_use = odf_calc_long.copy()

all_stats_filt = dfun.buildStatsDF(odf_use, site_list, odf_calc_use=odf_calc_use, odf_depth_mean_deep_DO_percentiles=odf_depth_mean_deep_DO_percentiles,stat_list=stat_list)

# %%

alpha=0.05

for site in site_list:
    
    for season in ['loDO', 'winter', 'grow']:
        
        mosaic = [['surf_CT', 'deep_CT'], ['surf_SA', 'deep_SA'], ['surf_DO_mg_L', 'deep_DO_mg_L']]
        
        fig, axd = plt.subplot_mosaic(mosaic, sharex=True, layout='constrained')
        
        for var in var_list:
            
            for depth in ['surf', 'deep']:
                
                var_name = depth +'_' + var
            
                ax = axd[var_name]
            
                plot_df = odf_use[(odf_use['site'] == site) & (odf_use['season'] == season) & (odf_use['var'] == var) & (odf_use['surf_deep'] == depth)]
                
                stat_plot_df_both = all_stats_filt[(all_stats_filt['site'] == site) & (all_stats_filt['season'] == season) & (all_stats_filt['var'] == var_name)]
                
                for stat in stat_list:
                    
                    stat_plot_df = stat_plot_df_both[stat_plot_df_both['stat'] == stat]
                                    
                    if stat_plot_df['p'].iloc[0] < alpha:
                        
                        color = 'k'
                        
                    else:
                        
                        color = 'gray'
                
                    ax.scatter(stat_plot_df['stat'], stat_plot_df['slope_datetime']*100, color=color, s=10)
                    
                    ax.scatter(stat_plot_df['stat'], stat_plot_df['slope_datetime_s_lo']*100, color=color, s=2)
                    
                    ax.scatter(stat_plot_df['stat'], stat_plot_df['slope_datetime_s_hi']*100, color=color, s=2)


                                                    
                    #ax.axvline(stat_plot_df['stat'], stat_plot_df['slope_datetime_s_lo']*100, stat_plot_df['slope_datetime_s_hi']*100, color=color)
                
                ax.axhline(y=0, ls='--', color='lightgray')
                
                ax.set_ylabel(var_name)
                
        fig.suptitle = (site + ' ' + season)
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_' + season + '_figcomp_linreg_mkts.png', dpi=500)
        
# %%

alpha=0.05

for site in site_list:
    
    for season in ['loDO', 'winter', 'grow']:
        
        mosaic = [['surf_CT', 'deep_CT'], ['surf_SA', 'deep_SA'], ['surf_DO_mg_L', 'deep_DO_mg_L']]
        
        fig, axd = plt.subplot_mosaic(mosaic, sharex=True, layout='constrained')
        
        for var in var_list:
            
            for depth in ['surf', 'deep']:
                
                var_name = depth +'_' + var
            
                ax = axd[var_name]
            
                plot_df = odf_use[(odf_use['site'] == site) & (odf_use['season'] == season) & (odf_use['var'] == var) & (odf_use['surf_deep'] == depth)]
                
                stat_plot_df_linreg = all_stats_filt[(all_stats_filt['site'] == site) & (all_stats_filt['season'] == season) & (all_stats_filt['var'] == var_name) & (all_stats_filt['stat'] == 'linreg')]
                
                x = plot_df['date_ordinal']
                
                y = plot_df['val']
                                
                y_predicted = stat_plot_df_linreg['B0'].iloc[0] + stat_plot_df_linreg['B1'].iloc[0]*x
                
                residuals = y - y_predicted
                
                ax.plot(x, residuals)
                
                ax.axhline(y=0, ls='--', color='lightgray')
                
                ax.set_ylabel(var_name)
                
        fig.suptitle = (site + ' ' + season)
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_' + season + '_linreg_residuals.png', dpi=500)
        
# %%

alpha=0.05

for site in site_list:
    
    for season in ['loDO', 'winter', 'grow']:
        
        mosaic = [['surf_CT', 'deep_CT'], ['surf_SA', 'deep_SA'], ['surf_DO_mg_L', 'deep_DO_mg_L']]
        
        fig, axd = plt.subplot_mosaic(mosaic, sharex=True, layout='constrained')
        
        for var in var_list:
            
            for depth in ['surf', 'deep']:
                
                var_name = depth +'_' + var
            
                ax = axd[var_name]
            
                plot_df = odf_use[(odf_use['site'] == site) & (odf_use['season'] == season) & (odf_use['var'] == var) & (odf_use['surf_deep'] == depth)]
                
                stat_plot_df_linreg = all_stats_filt[(all_stats_filt['site'] == site) & (all_stats_filt['season'] == season) & (all_stats_filt['var'] == var_name) & (all_stats_filt['stat'] == 'linreg')]
                
                x = plot_df['date_ordinal']
                
                y = plot_df['val']
                                
                y_predicted = stat_plot_df_linreg['B0'].iloc[0] + stat_plot_df_linreg['B1'].iloc[0]*x
                
                residuals = y - y_predicted
                
                ax.hist(residuals)
                
                #ax.axhline(y=0, ls='--', color='lightgray')
                
                ax.set_ylabel(var_name)
                
        fig.suptitle = (site + ' ' + season)
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_' + season + '_linreg_residualhist.png', dpi=500)