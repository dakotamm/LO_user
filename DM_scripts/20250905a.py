#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:11:45 2024

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

# %%

poly_list = ['ps']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_nc', 'ecology_his', 'kc', 'kc_taylor', 'kc_whidbey', 'kc_point_jefferson', 'nceiSalish'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))

# %%

basin_list = list(odf_dict.keys())

var_list = ['SA', 'CT', 'NO3 (uM)', 'Chl (mg m-3)', 'NO2 (uM)', 'SiO4 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']

# %%

odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)



# %%


# %%

for decade in ['1930', '1940', '1950', '1960', '1970', '1980','1990', '2000', '2010', '2020']: 
    
    for season in ['winter','grow','loDO']:
        
        for basin in basin_list:
            
            for var in var_list:
            
                fig, (ax, axx)  = plt.subplots(ncols = 2, figsize = (20,20))
                
                plt.rc('font', size=14)
                
                plot_df = odf[(odf['segment'] == basin) & (odf['season'] == season) & (odf['decade'] == decade) & (odf['var'] == var)].dropna(subset=['val'])
                
                if not plot_df.empty:
                
                    plot_df_map = plot_df.groupby('cid').first().reset_index()
                    
                    sns.scatterplot(data=plot_df_map, x='lon', y='lat', hue='source_type', hue_order = ['collias_bottle', 'ecology_nc_bottle', 'ecology_nc_ctd', 'ecology_his_bottle', 'ecology_his_ctd', 'kc_bottle', 'kc_ctd', 'kc_whidbey_ctd', 'kc_taylor_bottle', 'kc_point_jefferson_bottle', 'kc_point_jefferson_ctd', 'nceiSalish_bottle'], alpha = 0.5, ax = ax, legend = False)
                    
                    sns.scatterplot(data=plot_df, x='val', y='z', hue='source_type', hue_order = ['collias_bottle', 'ecology_nc_bottle', 'ecology_nc_ctd', 'ecology_his_bottle', 'ecology_his_ctd', 'kc_bottle', 'kc_ctd', 'kc_whidbey_ctd', 'kc_taylor_bottle', 'kc_point_jefferson_bottle', 'kc_point_jefferson_ctd', 'nceiSalish_bottle'], alpha=0.5, linewidth=0, ax = axx)
                    
                    ax.autoscale(enable=False)
                    
                    pfun.add_coast(ax)
                    
                    pfun.dar(ax)
                    
                    ax.set_xlim(-123.2, -122.1)
                    
                    ax.set_ylim(47,48.5)
                    
                    ax.set_title(decade + ' ' + season + ' ' + var.replace(' ', '_').replace('(','').replace(')',''))
                    
                    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                    
                    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/decadal/ps_' + decade + 's_' + season + '_' + var.replace(" ", "_").replace('(','').replace(')','') +'_raw.png', bbox_inches='tight', dpi=500)

# %%

for year in odf['year'].unique():
    
    for season in ['winter','grow','loDO']:

        for basin in basin_list:
            
            for var in var_list:
                
                fig, (ax, axx)  = plt.subplots(ncols = 2, figsize = (20,20))
                
                plt.rc('font', size=14)
                
                plot_df = odf[(odf['segment'] == basin) & (odf['season'] == season) & (odf['year'] == year) & (odf['var'] == var)].dropna(subset=['val'])
                
                if not plot_df.empty:
                
                    plot_df_map = plot_df.groupby('cid').first().reset_index()
                    
                    sns.scatterplot(data=plot_df_map, x='lon', y='lat', hue='source_type', hue_order = ['collias_bottle', 'ecology_nc_bottle', 'ecology_nc_ctd', 'ecology_his_bottle', 'ecology_his_ctd', 'kc_bottle', 'kc_ctd', 'kc_whidbey_ctd', 'kc_taylor_bottle', 'kc_point_jefferson_bottle', 'kc_point_jefferson_ctd', 'nceiSalish_bottle'], alpha = 0.5, ax = ax, legend = False)
                    
                    sns.scatterplot(data=plot_df, x='val', y='z', hue='source_type', hue_order = ['collias_bottle', 'ecology_nc_bottle', 'ecology_nc_ctd', 'ecology_his_bottle', 'ecology_his_ctd', 'kc_bottle', 'kc_ctd', 'kc_whidbey_ctd', 'kc_taylor_bottle', 'kc_point_jefferson_bottle', 'kc_point_jefferson_ctd', 'nceiSalish_bottle'], alpha=0.5, linewidth=0, ax = axx)
                    
                    ax.autoscale(enable=False)
                    
                    pfun.add_coast(ax)
                    
                    pfun.dar(ax)
                    
                    ax.set_xlim(-123.2, -122.1)
                    
                    ax.set_ylim(47,48.5)
                    
                    ax.set_title(str(year) + ' ' + season + ' ' + var.replace(' ', '_').replace('(','').replace(')',''))
                    
                    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                    
                    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/annual/ps_' + str(year) + '_' + season + '_' + var.replace(" ", "_").replace('(','').replace(')','') +'_raw.png', bbox_inches='tight', dpi=500)

