#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 12:45:38 2025

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

seasonal_site_counts = (odf_depth_mean
                        .dropna()
                        .groupby(['site','year','surf_deep', 'season', 'var']).agg({'cid' :lambda x: x.nunique()})
                        .reset_index()
                        .rename(columns={'cid':'cid_count'})
                        )

# %%

for site in poly_list:
    
    fig, axd = plt.subplot_mosaic([['surf_CT', 'deep_CT'], ['surf_SA', 'deep_SA'], ['surf_DO_mg_L', 'deep_DO_mg_L']], sharex=True, layout='constrained')
    
    for var in var_list:
        
        for depth in ['surf', 'deep']:
            
            var_name = depth + '_' + var
            
            ax = axd[var_name]
            
            plot_df = seasonal_site_counts[(seasonal_site_counts['site'] == site) & (seasonal_site_counts['var'] == var) & (seasonal_site_counts['surf_deep'] == depth)]
            
            sns.scatterplot(data=plot_df, x='year', y='cid_count', hue='season', palette={'loDO':'red', 'winter': 'blue', 'grow':'gold'}, legend=False, ax=ax)
            
            ax.set_ylabel(var_name)
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_seasonalcounts.png',dpi=500)
    
# %%

for site in poly_list:
    
    fig, axd = plt.subplot_mosaic([[ 'deep_DO_mg_L']], sharex=True, layout='constrained')
    
    for var in ['DO_mg_L']:
        
        for depth in ['deep']:
            
            var_name = depth + '_' + var
            
            ax = axd[var_name]
            
            plot_df = odf_depth_mean[(odf_depth_mean['site'] == site) & (odf_depth_mean['var'] == var) & (odf_depth_mean['surf_deep'] == depth) & (odf_depth_mean['season'] == 'loDO')]
            
            sns.lineplot(data=plot_df, x='yearday', y='val', hue='year', ax=ax, legend=False, errorbar=None)
            
            ax.set_ylabel(var_name)
            
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_deepDOmgL_loDO_byyear.png',dpi=500)
    
# %%
    
for site in poly_list:
    
    fig, axd = plt.subplot_mosaic([[ 'deep_DO_mg_L']], sharex=True, layout='constrained')
    
    for var in ['DO_mg_L']:
        
        for depth in ['deep']:
            
            var_name = depth + '_' + var
            
            ax = axd[var_name]
            
            plot_df = odf_depth_mean_deep_DO_percentiles[(odf_depth_mean_deep_DO_percentiles['site'] == site) & (odf_depth_mean_deep_DO_percentiles['var'] == var) & (odf_depth_mean_deep_DO_percentiles['surf_deep'] == depth) & (odf_depth_mean_deep_DO_percentiles['season'] == 'loDO')]
            
            plot_df = plot_df[plot_df['val'] <= plot_df['deep_DO_q50']]
            
            sns.lineplot(data=plot_df, x='yearday', y='val', hue='year', ax=ax, legend=False, errorbar=None)
            
            ax.set_ylabel(var_name)
            
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_deepDOmgLq50_loDO_byyear.png',dpi=500)

# %%
    
for site in poly_list:
        
    for season in ['loDO', 'winter', 'grow']:
        
        fig, axd = plt.subplot_mosaic([['deep_DO_mg_L', 'deep_DO_mg_L_q50']], sharex=True, sharey=True, layout='constrained')

    
        for var in ['DO_mg_L']:
            
            for depth in ['deep']:
                
                var_name = depth + '_' + var
                
                ax = axd[var_name]
                
                plot_df = odf_depth_mean[(odf_depth_mean['site'] == site) & (odf_depth_mean['var'] == var) & (odf_depth_mean['surf_deep'] == depth) & (odf_depth_mean['season'] == season)]
                
                sns.lineplot(data=plot_df, x='yearday', y='val', hue='year', ax=ax, legend=False, errorbar=None, palette='plasma_r')
                
                ax.set_ylabel(var_name)
                
                ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
                                                    
                ax.set_title('all samples')
                 
                
                ax = axd[var_name + '_q50']
                
                plot_df = odf_depth_mean_deep_DO_percentiles[(odf_depth_mean_deep_DO_percentiles['site'] == site) & (odf_depth_mean_deep_DO_percentiles['var'] == var) & (odf_depth_mean_deep_DO_percentiles['surf_deep'] == depth) & (odf_depth_mean_deep_DO_percentiles['season'] == season)]
                
                plot_df = plot_df[plot_df['val'] <= plot_df['deep_DO_q50']]
                
                sns.lineplot(data=plot_df, x='yearday', y='val', hue='year', ax=ax, legend=False, errorbar=None, palette='plasma_r')
                
                ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
                                                    
                ax.set_title('median and below')

        plt.suptitle(site + ' ' + season)
            
                        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_deepDOmgLvsq50_' + season + '_byyear.png',dpi=500)