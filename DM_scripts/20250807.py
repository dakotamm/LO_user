#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 12:12:34 2025

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



poly_list = ['mb', 'hc', 'ss', 'wb']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)


odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)


odf_use = (odf_depth_mean
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

# %%

odf_temp = odf.copy()

odf_temp = odf_temp.groupby(['name']).first().reset_index()

odf_temp['site'] = odf_temp['name']

odf_temp['basin'] = odf_temp['segment']

odf_use = pd.merge(odf_use, odf_temp[['site','basin']], how='left', on='site')

# %%

mosaic = [['surf_CT', 'surf_SA', 'surf_DO_mg_L'], ['deep_CT', 'deep_SA','deep_DO_mg_L']]

for season in ['grow', 'loDO', 'winter']:
    
    fig, axd = plt.subplot_mosaic(mosaic, sharex=True, figsize=(12,9), layout='constrained', gridspec_kw=dict(wspace=0.1, hspace=0.1))

    for depth in ['surf', 'deep']:
        
        for var in ['CT', 'SA', 'DO_mg_L']:
            
            ax_name = depth + '_' + var
            
            ax = axd[ax_name]
            
            for site in short_site_list:
            
                plot_df = odf_use[(odf_use['site'] == site) & (odf_use['season'] == season) & (odf_use['surf_deep'] == depth) & (odf_use['var'] == var)]
            
                gam = LinearGAM(s(0)) 
            
                x_pred = plot_df['datetime'].sort_values().unique()
            
                gam.fit(plot_df['datetime'], plot_df['val'])
            
                y_pred = gam.predict(x_pred)
            
                #ax.scatter(plot_df_grow['datetime'], plot_df_grow['mean_va'], marker = '.', alpha=0.1])
                ax.plot(x_pred, y_pred, label=site)
                
                ax.set_ylabel(ax_name)
                
                if ax_name == 'surf_CT':
                
                    ax.legend()
                    
                ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
                
                
                
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + season + '_ecology_GAM.png', dpi=500)

# %%

good_sites = ['SAR003','KSBP01', 'HCB007', 'CRR001', 'ADM003'] #, 'carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'point_jefferson', 'saratoga_passage_mid']


mosaic = [['surf_CT', 'surf_SA', 'surf_DO_mg_L'], ['deep_CT', 'deep_SA','deep_DO_mg_L']]

for season in ['grow', 'loDO', 'winter']:
    
    fig, axd = plt.subplot_mosaic(mosaic, sharex=True, figsize=(12,9), layout='constrained', gridspec_kw=dict(wspace=0.1, hspace=0.1))

    for depth in ['surf', 'deep']:
        
        for var in ['CT', 'SA', 'DO_mg_L']:
            
            ax_name = depth + '_' + var
            
            ax = axd[ax_name]
            
            for site in good_sites:
            
                plot_df = odf_use[(odf_use['site'] == site) & (odf_use['season'] == season) & (odf_use['surf_deep'] == depth) & (odf_use['var'] == var)]
            
                gam = LinearGAM(s(0)) 
            
                x_pred = plot_df['datetime'].sort_values().unique()
            
                gam.fit(plot_df['datetime'], plot_df['val'])
            
                y_pred = gam.predict(x_pred)
            
                #ax.scatter(plot_df_grow['datetime'], plot_df_grow['mean_va'], marker = '.', alpha=0.1])
                ax.plot(x_pred, y_pred, label=site)
                
                ax.set_ylabel(ax_name)
                
                if ax_name == 'surf_CT':
                
                    ax.legend()
                    
                ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
                
                
                
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + season + '_ecology_select_GAM.png', dpi=500)
            
# %%

good_sites = ['SAR003','KSBP01', 'HCB007', 'CRR001', 'ADM003'] #, 'carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'point_jefferson', 'saratoga_passage_mid']


mosaic = [['surf_CT', 'surf_SA', 'surf_DO_mg_L'], ['deep_CT', 'deep_SA','deep_DO_mg_L']]

for basin in big_basin_list:

    for season in ['grow', 'loDO', 'winter']:
    
        fig, axd = plt.subplot_mosaic(mosaic, sharex=True, figsize=(12,9), layout='constrained', gridspec_kw=dict(wspace=0.1, hspace=0.1))
        
        for depth in ['surf', 'deep']:
            
            for var in ['CT', 'SA', 'DO_mg_L']:
                
                ax_name = depth + '_' + var
                
                ax = axd[ax_name]
                
                basin_df = odf_use[(odf_use['basin'] == basin)]

                for site in basin_df['site'].unique():
                
                    plot_df = odf_use[(odf_use['basin'] == basin) & (odf_use['site'] == site) & (odf_use['season'] == season) & (odf_use['surf_deep'] == depth) & (odf_use['var'] == var)]
                
                    gam = LinearGAM(s(0)) 
                
                    x_pred = plot_df['datetime'].sort_values().unique()
                
                    gam.fit(plot_df['datetime'], plot_df['val'])
                
                    y_pred = gam.predict(x_pred)
                
                    #ax.scatter(plot_df_grow['datetime'], plot_df_grow['mean_va'], marker = '.', alpha=0.1])
                    ax.plot(x_pred, y_pred, label=site)
                    
                    ax.set_ylabel(ax_name)
                    
                    if ax_name == 'surf_CT':
                    
                        ax.legend()
                        
                    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
                
                
                
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + season + '_ecology_select_GAM.png', dpi=500)

