#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:35:10 2024

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




poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson', 'mb', 'hc', 'ss', 'wb'] # 5 sites + 4 basins

#poly_list = ['ps']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)

# %%

site_list =  odf['site'].unique()


odf_use_summer_DO, odf_use_summer_CTSA, odf_use_annual_DO, odf_use_annual_CTSA = dfun.calcSeriesAvgs(odf_depth_mean, odf_depth_mean_deep_DO_percentiles)



odf_use = odf_depth_mean.copy()

odf_calc_use = odf_calc_long.copy()

all_stats_filt = dfun.buildStatsDF(odf_use, site_list, odf_calc_use=odf_calc_use, odf_depth_mean_deep_DO_percentiles=odf_depth_mean_deep_DO_percentiles)


# %%

c=0

all_stats_filt = all_stats_filt.sort_values(by=['site'])

odf_use_summer_CTSA = odf_use_summer_CTSA.sort_values(by=['site'])

odf_use_summer_DO = odf_use_summer_DO.sort_values(by=['site'])

odf_use_annual_CTSA = odf_use_annual_CTSA.sort_values(by=['site'])

odf_use_annual_DO = odf_use_annual_DO.sort_values(by=['site'])





for site in all_stats_filt['site'].unique():
        
    all_stats_filt.loc[all_stats_filt['site'] == site, 'site_num'] = c
    
    odf_use_summer_CTSA.loc[odf_use_summer_CTSA['site'] == site, 'site_num'] = c

    odf_use_summer_DO.loc[odf_use_summer_DO['site'] == site, 'site_num'] = c

    odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == site, 'site_num'] = c

    odf_use_annual_DO.loc[odf_use_annual_DO['site'] == site, 'site_num'] = c

    c+=1
    
site_labels = sorted(site_list)



mosaic = [['DO_mg_L', 'CT', 'SA'], ['surf_DO_mg_L', 'surf_CT', 'surf_SA'], ['deep_DO_mg_L', 'deep_CT', 'deep_SA']]

fig, axd = plt.subplot_mosaic(mosaic, figsize=(10,8), layout='constrained', sharex=True, gridspec_kw=dict(wspace=0.1))

for var in ['DO_mg_L', 'CT', 'SA']:
    
    summer_palette = {'surf':'#E91E63', 'deep':'#673AB7'}
    
    annual_palette = {'surf':'#c8aca9', 'deep': '#a7a6ba'}
    
    if 'DO' in var:
        
        label_var = '[DO]'
        
        ymin = 0
        
        ymax = 12
        
        marker = 'o'
        
        unit = r'[mg/L]'
        
    elif 'CT' in var:
        
        label_var = 'Temperature'
        
        ymin = 8
        
        ymax = 16
        
        marker = 'D'
        
        unit = r'[$^{\circ}$C]'
    
    else:
        
        label_var = 'Salinity'
        
        ymin = 22
        
        ymax = 32
        
        marker = 's'
        
        unit = r'[PSU]'
        
        
    ax = axd[var]
    
    
    if 'DO' in var:
        
        ax.axhspan(0,2, color = 'lightgray', alpha = 0.2)
    
    
    if 'DO' in var:
        
        plot_df = odf_use_annual_DO[(odf_use_annual_DO['var'] == var) & (odf_use_annual_DO['site'].isin(long_site_list))]
        
    else:
        
        plot_df = odf_use_annual_CTSA[(odf_use_annual_CTSA['var'] == var) & (odf_use_annual_CTSA['site'].isin(long_site_list))]

    plot_df = plot_df.sort_values(by=['site']).reset_index()



    
    sns.scatterplot(data = plot_df, x= 'site_num', y = 'val_ci95hi', hue ='surf_deep', palette = annual_palette, marker=marker, ax = ax, s= 10, legend=False)

    sns.scatterplot(data = plot_df, x= 'site_num', y = 'val_ci95lo', hue ='surf_deep', palette = annual_palette, marker=marker, ax = ax, s= 10, legend=False)

    sns.scatterplot(data = plot_df, x= 'site_num', y = 'val_mean', hue ='surf_deep', palette = annual_palette, marker=marker, ax = ax, s =50, legend=False)
    
    for idx in plot_df.index:
        
        if plot_df.loc[idx,'surf_deep'] == 'surf':
            
            ax.plot([plot_df.loc[idx,'site_num'], plot_df.loc[idx,'site_num']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color=annual_palette['surf'], alpha =0.7, zorder = -5, linewidth=1)

        else:
            
            ax.plot([plot_df.loc[idx,'site_num'], plot_df.loc[idx,'site_num']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color=annual_palette['deep'], alpha =0.7, zorder = -4, linewidth=1)
    
    label = 'Mean ' + label_var
        
    ax.text(0.05,0.05, label, transform=ax.transAxes, verticalalignment='bottom', fontweight = 'bold', color='k')
    
    if 'DO' in var:
        
        ax.axhspan(0,2, color = 'lightgray', alpha = 0.2)
    
    
    if 'DO' in var:
        
        plot_df = odf_use_summer_DO[(odf_use_summer_DO['var'] == var) & (odf_use_summer_DO['site'].isin(long_site_list)) & (odf_use_summer_DO['summer_non_summer'] == 'summer')]
        
    else:
        
        plot_df = odf_use_summer_CTSA[(odf_use_summer_CTSA['var'] == var) & (odf_use_summer_CTSA['site'].isin(long_site_list))& (odf_use_summer_DO['summer_non_summer'] == 'summer')]

    plot_df = plot_df.sort_values(by=['site']).reset_index()



    
    sns.scatterplot(data = plot_df, x= 'site_num', y = 'val_ci95hi', hue ='surf_deep', palette = summer_palette, marker=marker, ax = ax, s= 10, legend=False)

    sns.scatterplot(data = plot_df, x= 'site_num', y = 'val_ci95lo', hue ='surf_deep', palette = summer_palette, marker=marker, ax = ax, s= 10, legend=False)

    sns.scatterplot(data = plot_df, x= 'site_num', y = 'val_mean', hue ='surf_deep', palette = summer_palette, marker=marker, ax = ax, s =50, legend=False)
    
    for idx in plot_df.index:
        
        if plot_df.loc[idx,'surf_deep'] == 'surf':
            
            ax.plot([plot_df.loc[idx,'site_num'], plot_df.loc[idx,'site_num']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color=summer_palette['surf'], alpha =0.7, zorder = -5, linewidth=1)

        else:
            
            ax.plot([plot_df.loc[idx,'site_num'], plot_df.loc[idx,'site_num']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color=summer_palette['deep'], alpha =0.7, zorder = -4, linewidth=1)
    
    
    

        
        
    
    # ymin = -max(abs(plot_df['slope_datetime_cent']))*2.5
    
    # ymax = max(abs(plot_df['slope_datetime_cent']))*2.5
    
    #ax.set_xticks(sorted(all_stats_filt['site_num'].unique().tolist()),site_labels, rotation=90) 
    
    ax.set_xticks([20,21,22,23,24], ['1', '2', '3', '4', '5'])
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
                            
    ax.axhline(0, color='gray', linestyle = '--', zorder = -5)
    
    ax.set_ylabel(unit, wrap=True)
    
    ax.set_xlabel('')
    
    ax.set_ylim(ymin, ymax) 
        
    
    

for var in ['surf_DO_mg_L', 'surf_CT', 'surf_SA', 'deep_DO_mg_L', 'deep_CT', 'deep_SA']:
    
    for stat in ['mk_ts']:
                
        for deep_DO_q in ['deep_DO_q50']:
            
            if 'surf' in var:
                
                label_depth = 'Surface'
                
                color = '#E91E63'
                
            else:
                
                label_depth = 'Deep'
                
                color = '#673AB7'

            
            if 'DO' in var:
                
                label_var = '[DO]'
                
                ymin = -4
                
                ymax = 4
                
                marker = 'o'
                
                unit = r'[mg/L]/century'
            
            elif 'CT' in var:
                
                label_var = 'Temperature'
                
                ymin = -10
                
                ymax = 10
                
                marker = 'D'
                
                unit = r'[$^{\circ}$C]/century'
            
            else:
                
                label_var = 'Salinity'
                
                ymin = -4
                
                ymax = 4
                
                marker = 's'
                
                unit = r'[PSU]/century'

                
                
            
            ax = axd[var]

            plot_df = all_stats_filt[(all_stats_filt['stat'] == stat) & (all_stats_filt['var'] == var) & (all_stats_filt['site'].isin(long_site_list)) & (all_stats_filt['deep_DO_q'] == deep_DO_q)]
            
            plot_df = plot_df.sort_values(by=['site']).reset_index()
            
            plot_df['slope_datetime_cent'] = plot_df['slope_datetime']*100
            
            plot_df['slope_datetime_cent_95hi'] = plot_df['slope_datetime_s_hi']*100
            
            plot_df['slope_datetime_cent_95lo'] = plot_df['slope_datetime_s_lo']*100

    
            
            sns.scatterplot(data = plot_df[plot_df['summer_non_summer'] == 'all'], x= 'site_num', y = 'slope_datetime_cent_95hi', hue ='summer_non_summer', palette = {'all':'lightgray', 'summer':color}, marker=marker, ax = ax, s= 10, legend=False, hue_order = ['all', 'summer'], zorder=-5)
    
            sns.scatterplot(data = plot_df[plot_df['summer_non_summer'] == 'all'], x= 'site_num', y = 'slope_datetime_cent_95lo', hue ='summer_non_summer', palette = {'all':'lightgray', 'summer':color}, marker=marker, ax = ax, s= 10, legend=False, hue_order = ['all', 'summer'], zorder=-5)
    
            sns.scatterplot(data = plot_df[plot_df['summer_non_summer'] == 'all'], x= 'site_num', y = 'slope_datetime_cent', hue ='summer_non_summer', palette = {'all':'lightgray', 'summer':color}, marker=marker, ax = ax, s =50, legend=False, hue_order = ['all', 'summer'], zorder=-5)
            
            
            sns.scatterplot(data = plot_df[plot_df['summer_non_summer'] == 'summer'], x= 'site_num', y = 'slope_datetime_cent_95hi', hue ='summer_non_summer', palette = {'all':'lightgray', 'summer':color}, marker=marker, ax = ax, s= 10, legend=False, hue_order = ['all', 'summer'])
    
            sns.scatterplot(data = plot_df[plot_df['summer_non_summer'] == 'summer'], x= 'site_num', y = 'slope_datetime_cent_95lo', hue ='summer_non_summer', palette = {'all':'lightgray', 'summer':color}, marker=marker, ax = ax, s= 10, legend=False, hue_order = ['all', 'summer'])
    
            sns.scatterplot(data = plot_df[plot_df['summer_non_summer'] == 'summer'], x= 'site_num', y = 'slope_datetime_cent', hue ='summer_non_summer', palette = {'all':'lightgray', 'summer':color}, marker=marker, ax = ax, s =50, legend=False, hue_order = ['all', 'summer'])
            
            for idx in plot_df.index:
                
                if plot_df.loc[idx,'summer_non_summer'] == 'all':
                    
                    ax.plot([plot_df.loc[idx,'site_num'], plot_df.loc[idx,'site_num']],[plot_df.loc[idx,'slope_datetime_cent_95lo'], plot_df.loc[idx,'slope_datetime_cent_95hi']], color='lightgray', alpha =0.7, zorder = -5, linewidth=1)

                else:
                    
                    ax.plot([plot_df.loc[idx,'site_num'], plot_df.loc[idx,'site_num']],[plot_df.loc[idx,'slope_datetime_cent_95lo'], plot_df.loc[idx,'slope_datetime_cent_95hi']], color=color, alpha =0.7, zorder = -4, linewidth=1)
            
            
            label = label_depth + ' ' + label_var
                
            ax.text(0.05,0.05, label, transform=ax.transAxes, verticalalignment='bottom', fontweight = 'bold', color='k')
                
                
            
            # ymin = -max(abs(plot_df['slope_datetime_cent']))*2.5
            
            # ymax = max(abs(plot_df['slope_datetime_cent']))*2.5
            
            #ax.set_xticks(sorted(all_stats_filt['site_num'].unique().tolist()),site_labels, rotation=90) 
            
            ax.set_xticks([20,21,22,23,24], ['1', '2', '3', '4', '5'])
            
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
                                    
            ax.axhline(0, color='gray', linestyle = '--', zorder = -5)
            
            ax.set_ylabel(unit, wrap=True)
            
            ax.set_xlabel('')
            
            ax.set_ylim(ymin, ymax)
            
            
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + 'longsites_' + stat + '_' + deep_DO_q + '_slopes_wannualavg_newcode.png', dpi=500,transparent=False, bbox_inches='tight')