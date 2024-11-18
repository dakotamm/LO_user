#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:42:41 2024

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

odf_use = odf_depth_mean.copy()

odf_calc_use = odf_calc_long.copy()


# %%

all_stats_filt = dfun.buildStatsDF(odf_use, site_list, odf_calc_use=odf_calc_use, odf_depth_mean_deep_DO_percentiles=odf_depth_mean_deep_DO_percentiles, alpha=0.05,  deep_DO_q_list = ['deep_DO_q50'], season_list = ['all', 'summer'], stat_list = ['mk_ts'], depth_list=['surf', 'deep'])
# %%

odf_use_summer_DO, odf_use_summer_CTSA, odf_use_annual_DO, odf_use_annual_CTSA = dfun.calcSeriesAvgs(odf_depth_mean, odf_depth_mean_deep_DO_percentiles, deep_DO_q = 'deep_DO_q50')

# %%

# %%


site_label_dict = {'point_jefferson':'PJ', 'near_seattle_offshore':'NS', 'carr_inlet_mid':'CI', 'lynch_cove_mid':'LC', 'saratoga_passage_mid':'SP'}

for site in site_label_dict.keys():
    

    all_stats_filt.loc[all_stats_filt['site'] == site, 'site_label'] = site_label_dict[site]
    
    odf_use_summer_CTSA.loc[odf_use_summer_CTSA['site'] == site, 'site_label'] = site_label_dict[site]
    
    odf_use_summer_DO.loc[odf_use_summer_DO['site'] == site, 'site_label'] = site_label_dict[site]
    
    odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == site, 'site_label'] = site_label_dict[site]
    
    odf_use_annual_DO.loc[odf_use_annual_DO['site'] == site, 'site_label'] = site_label_dict[site]
    
    
odf_use_summer_DO = odf_use_summer_DO[odf_use_summer_DO['var'] == 'DO_mg_L']

odf_use_annual_DO = odf_use_annual_DO[odf_use_annual_DO['var'] == 'DO_mg_L']

odf_use_summer_CTSA = odf_use_summer_CTSA[odf_use_summer_CTSA['var'] != 'DO_mg_L']

odf_use_annual_CTSA = odf_use_annual_CTSA[odf_use_annual_CTSA['var'] != 'DO_mg_L']


    

odf_use_annual_CTSA['summer_non_summer'] = 'all'

odf_use_annual_DO['summer_non_summer'] = 'all'

    

odf_use_means = pd.concat([odf_use_summer_CTSA, odf_use_summer_DO, odf_use_annual_CTSA, odf_use_annual_DO])

# %%

import matplotlib.transforms as transforms


offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
trans = plt.gca().transData

# then use plot.scatter instead of seaborn?


# %%

multiplier=1


sites = ['point_jefferson']


mosaic = [['CT', 'SA', 'DO_mg_L'],['surf_CT', 'surf_SA', 'surf_DO_mg_L'], ['deep_CT', 'deep_SA', 'deep_DO_mg_L']]

fig, axd = plt.subplot_mosaic(mosaic, figsize=(11,5), layout='constrained', gridspec_kw=dict(wspace=0.1))

for var in ['DO_mg_L', 'CT', 'SA']:
    
    palette = {'surf':'#E91E63', 'deep':'#673AB7'}
                
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
    
    
    # if 'DO' in var:
        
    #     plot_df = odf_use_annual_DO[(odf_use_annual_DO['var'] == var) & (odf_use_annual_DO['site'].isin(sites))]
        
    # else:
        
    #     plot_df = odf_use_annual_CTSA[(odf_use_annual_CTSA['var'] == var) & (odf_use_annual_CTSA['site'].isin(sites))]
    
    plot_df = odf_use_means[(odf_use_means['summer_non_summer']!='non_summer') & (odf_use_means['var'] == var) & (odf_use_means['site'].isin(sites))]

    plot_df = plot_df.sort_values(by=['site']).reset_index()
    
    plot_df['season_depth'] = plot_df['summer_non_summer'] + '_' + plot_df['surf_deep']
    
    plot_df = plot_df.sort_values(by=['season_depth']).reset_index()
    
    plot_df = plot_df[plot_df['summer_non_summer'] == 'summer']
    
    
    
    #new_palette = {'all_deep': '#673AB7', 'summer_deep': '#673AB7', 'all_surf':'#E91E63', 'summer_surf':'#E91E63'}
        
    
    sns.scatterplot(data = plot_df, x= 'surf_deep', y = 'val_ci95hi', hue ='surf_deep', palette = palette, ax = ax, s=10*multiplier, legend = False)

    sns.scatterplot(data = plot_df, x= 'surf_deep', y = 'val_ci95lo', hue ='surf_deep', palette = palette, ax = ax,  s=10*multiplier, legend=False)
    
    sns.scatterplot(data = plot_df, x= 'surf_deep', y = 'val_mean', hue ='surf_deep', palette = palette,  ax = ax,  s=50*multiplier, legend=False)

    
    for idx in plot_df.index:
        
        if plot_df.loc[idx,'surf_deep'] == 'surf':
                        
            ax.plot([plot_df.loc[idx,'surf_deep'], plot_df.loc[idx,'surf_deep']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color=palette['surf'], alpha =0.7, zorder = -4, linewidth=1)

        
        else:
                        
            ax.plot([plot_df.loc[idx,'surf_deep'], plot_df.loc[idx,'surf_deep']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color=palette['deep'], alpha =0.7, zorder = -4, linewidth=1)

    
    label = 'Mean ' + label_var
        
    ax.text(0.05,0.05, label, transform=ax.transAxes, verticalalignment='bottom', fontweight = 'bold', color='k')
    
    if 'DO' in var:
        
        ax.axhspan(0,2, color = 'lightgray', alpha = 0.2)
        
        
        
    

    
    #ax.set_xticks(['PJ'], ['Point Jefferson'])
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
                            
    ax.axhline(0, color='gray', linestyle = '--', zorder = -5)
    
    ax.set_ylabel(unit, wrap=True)
    
    ax.set_xlabel('')
    
    ax.set_ylim(ymin, ymax) 
    
    ax.set_xticks(['deep', 'surf'], [ 'Bottom 20%', 'Surface 5m'])

        
    
    

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

            plot_df = all_stats_filt[(all_stats_filt['stat'] == stat) & (all_stats_filt['var'] == var) & (all_stats_filt['site'].isin(sites)) & (all_stats_filt['deep_DO_q'] == deep_DO_q) & (all_stats_filt['summer_non_summer'] != 'non_summer')]
            

            plot_df = plot_df.sort_values(by=['site']).reset_index()
            
            if 'deep' in var:
                
            
                plot_df['season_depth'] = plot_df['summer_non_summer'] + '_deep'
                
            elif 'surf' in var:
                
                plot_df['season_depth'] = plot_df['summer_non_summer'] + '_surf'

            
            plot_df = plot_df.sort_values(by=['season_depth']).reset_index()
            
            plot_df['slope_datetime_cent'] = plot_df['slope_datetime']*100
            
            plot_df['slope_datetime_cent_95hi'] = plot_df['slope_datetime_s_hi']*100
            
            plot_df['slope_datetime_cent_95lo'] = plot_df['slope_datetime_s_lo']*100
            
            
            
            sns.scatterplot(data = plot_df, x= 'season_depth', y = 'slope_datetime_cent_95hi', color = color, style = 'summer_non_summer', markers={'summer':marker,'all':'.'}, ax = ax, legend=False, hue_order = ['deep', 'surf'], s=10*multiplier)

            sns.scatterplot(data = plot_df, x= 'season_depth', y = 'slope_datetime_cent_95lo', color = color, style = 'summer_non_summer', markers={'summer':marker,'all':'.'}, ax = ax, legend=False, hue_order = ['deep', 'surf'], s=10*multiplier)
    
            sns.scatterplot(data = plot_df, x= 'season_depth', y = 'slope_datetime_cent', color = color, style = 'summer_non_summer', markers={'summer':marker,'all':'.'}, ax = ax, legend=False, hue_order = ['deep', 'surf'], s=50*multiplier)

    
            for idx in plot_df.index:
                
                if plot_df.loc[idx,'summer_non_summer'] == 'all':
                    
                    ax.plot([plot_df.loc[idx,'season_depth'], plot_df.loc[idx,'season_depth']],[plot_df.loc[idx,'slope_datetime_cent_95lo'], plot_df.loc[idx,'slope_datetime_cent_95hi']], color=color, alpha =0.7, zorder = -5, linewidth=1, linestyle=':')

                else:
                    
                    ax.plot([plot_df.loc[idx,'season_depth'], plot_df.loc[idx,'season_depth']],[plot_df.loc[idx,'slope_datetime_cent_95lo'], plot_df.loc[idx,'slope_datetime_cent_95hi']], color=color, alpha =0.7, zorder = -4, linewidth=1)
            
            
            label = label_depth + ' ' + label_var
                
            ax.text(0.05,0.05, label, transform=ax.transAxes, verticalalignment='bottom', fontweight = 'bold', color='k')
                
                
            
            # ymin = -max(abs(plot_df['slope_datetime_cent']))*2.5
            
            # ymax = max(abs(plot_df['slope_datetime_cent']))*2.5 
            
            #ax.set_xticks(sorted(all_stats_filt['site_num'].unique().tolist()),site_labels, rotation=90) 
            
            #ax.set_xticks(['PJ'], ['Point Jefferson'])
            
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
                                    
            ax.axhline(0, color='gray', linestyle = '--', zorder = -5)
            
            ax.set_ylabel(unit, wrap=True)
            
            ax.set_xlabel('')
            
            ax.set_ylim(ymin, ymax)
            
            
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/point_jefferson_' + stat + '_' + deep_DO_q + '_slopes_wannualavg.png', dpi=500,transparent=False, bbox_inches='tight')