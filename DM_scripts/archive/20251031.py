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

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_nc', 'ecology_his', 'kc', 'kc_his', 'kc_whidbeyBasin', 'kc_pointJefferson', 'nceiSalish'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))

# %%

basin_list = list(odf_dict.keys())

var_list = ['SA', 'CT', 'DO (uM)']

# %%

odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf.loc[odf['source'] == 'collias', 'agency'] = 'collias'

odf.loc[odf['source'].isin(['ecology_his', 'ecology_nc']), 'agency'] = 'ecology'

odf.loc[odf['source'].isin(['kc', 'kc_his', 'kc_whidbeyBasin', 'kc_pointJefferson']), 'agency'] = 'kc'

odf.loc[odf['source'] == 'nceiSalish', 'agency'] = 'ncei'




# %%


# %%

for year in odf['year'].unique():
    
    for basin in basin_list:
                    
        fig, (ax, axx)  = plt.subplots(ncols = 2, figsize = (20,20))
        
        plt.rc('font', size=14)
        
        plot_df = odf[(odf['segment'] == basin) & (odf['year'] == year)].dropna(subset=['val'])
        
        if not plot_df.empty:
        
            plot_df_map = plot_df.groupby('cid').first().reset_index()
            
            sns.scatterplot(data=plot_df_map, x='lon', y='lat', hue='agency', hue_order = ['collias', 'ecology', 'kc', 'ncei'], alpha = 0.5, ax = ax, legend = False)
            
            sns.scatterplot(data=plot_df, x='val', y='z', hue='agency', hue_order = ['collias', 'ecology', 'kc', 'ncei'], alpha=0.5, linewidth=0, ax = axx)
            
            ax.autoscale(enable=False)
            
            pfun.add_coast(ax)
            
            pfun.dar(ax)
            
            ax.set_xlim(-123.2, -122.1)
            
            ax.set_ylim(47,48.5)
            
            ax.set_title(str(year))
            
            #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                            
            plt.savefig('/Users/dakotamascarenas/Desktop/pltz/annual/ps_' + str(year) + '_raw.png', bbox_inches='tight', dpi=500)
            
# %%

for year in odf['year'].unique():
    
    for basin in basin_list:
        
        for var in var_list:
                    
            fig, (ax, axx)  = plt.subplots(ncols = 2, figsize = (20,20))
            
            plt.rc('font', size=14)
            
            plot_df = odf[(odf['segment'] == basin) & (odf['var'] == var) & (odf['year'] == year)].dropna(subset=['val'])
            
            if not plot_df.empty:
            
                plot_df_map = plot_df.groupby('cid').first().reset_index()
                
                sns.scatterplot(data=plot_df_map, x='lon', y='lat', hue='agency', hue_order = ['collias', 'ecology', 'kc', 'ncei'], alpha = 0.5, ax = ax, legend = False)
                
                sns.scatterplot(data=plot_df, x='val', y='z', hue='agency', hue_order = ['collias', 'ecology', 'kc', 'ncei'], alpha=0.5, linewidth=0, ax = axx)
                
                ax.autoscale(enable=False)
                
                pfun.add_coast(ax)
                
                pfun.dar(ax)
                
                ax.set_xlim(-123.2, -122.1)
                
                ax.set_ylim(47,48.5)
                
                ax.set_title(str(year) + ' ' + var.replace(' ', '_').replace('(','').replace(')',''))
                
                #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                
                plt.savefig('/Users/dakotamascarenas/Desktop/pltz/annual/ps_' + str(year) + '_' + var.replace(' ', '_').replace('(','').replace(')','') +'_raw.png', bbox_inches='tight', dpi=500)

