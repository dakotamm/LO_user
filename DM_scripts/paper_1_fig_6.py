#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:29:56 2024

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

# %%

# poly_list = ['ps']

# odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_his', 'kc_whidbeyBasin', 'nceiSalish', 'kc_pointJefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


# basin_list = list(odf_dict.keys())

# var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


# odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# # %%

# odf['ix_iy'] = odf['ix'].astype(str).apply(lambda x: x.zfill(4)) + '_' + odf['iy'].astype(str).apply(lambda x: x.zfill(4))


# # %%

# odf_ixiy_unique = odf.groupby(['ix_iy']).first().reset_index()

# %%



#poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson', 'mb', 'hc', 'ss', 'wb'] # 5 sites + 4 basins

poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_his', 'kc_whidbeyBasin', 'nceiSalish', 'kc_pointJefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)

# %%

odf_use = odf_depth_mean.copy()

odf_calc_use = odf_calc_long.copy()






for deep_DO_q in ['deep_DO_q50']:


    odf_depth_mean_deep_DO_less_than_percentile = odf_depth_mean_deep_DO_percentiles[odf_depth_mean_deep_DO_percentiles['val'] <= odf_depth_mean_deep_DO_percentiles[deep_DO_q]]

    cid_deep_DO_less_than_percentile = odf_depth_mean_deep_DO_less_than_percentile['cid']

    odf_use_DO_q = odf_depth_mean[odf_depth_mean['cid'].isin(cid_deep_DO_less_than_percentile)]

    odf_calc_use_DO_q = odf_calc_long[odf_calc_long['cid'].isin(cid_deep_DO_less_than_percentile)]

    odf_use_DO_q = (odf_use_DO_q
                      .dropna()
                      .assign(
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )
    
    odf_calc_use_DO_q = (odf_calc_use_DO_q
                      .dropna()
                      .assign(
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )

    
    odf_use_DO_q_AugNov = odf_use_DO_q[odf_use_DO_q['season'] == 'loDO']
    
    odf_calc_use_DO_q_AugNov = odf_calc_use_DO_q[odf_calc_use_DO_q['season'] == 'loDO']

    
    if deep_DO_q == 'deep_DO_q25':
    
        odf_use_q25 = odf_use_DO_q_AugNov
        
        odf_calc_use_q25 = odf_calc_use_DO_q_AugNov
        
    elif deep_DO_q == 'deep_DO_q50':
        
        odf_use_q50 = odf_use_DO_q_AugNov
        
        odf_calc_use_q50 = odf_calc_use_DO_q_AugNov
        
    elif deep_DO_q == 'deep_DO_q75':
        
        odf_use_q75 = odf_use_DO_q_AugNov
        
        odf_calc_use_q75 = odf_calc_use_DO_q_AugNov


# %%

odf_use = (odf_use
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

# %%

all_stats_filt = dfun.buildStatsDF(odf_use, site_list, odf_calc_use=odf_calc_use, odf_depth_mean_deep_DO_percentiles=odf_depth_mean_deep_DO_percentiles)



# %%

red =     "#EF5E3C"   # warm orange-red ##ff4040 #e04256

blue =     "#3A59B3"  # deep blue #4565e8


mosaic = [['CT', 'CT'], ['SA', 'SA'], [ 'DO_mg_L', 'DO_mg_L']] #, ['map_source', '.', '.'],]

fig, axd = plt.subplot_mosaic(mosaic, figsize=(9,6), layout='constrained', gridspec_kw=dict(wspace=0.1), sharex = True)




palette = {'point_jefferson':red, 'lynch_cove_mid':blue}


for var in var_list:
    
    ax = axd[var]
                        
    if 'DO' in var:
        
        label_var = '[DO]'
        
        ymin = 0
        
        ymax = 7
        
        marker = 'o'
        
        unit = r'[mg/L]'
        
    elif 'CT' in var:
        
        label_var = 'Temperature'
        
        ymin = 8
        
        ymax = 14
        
        marker = 'D'
        
        unit = r'[$^{\circ}$C]'
    
    else:
        
        label_var = 'Salinity'
        
        ymin = 29
        
        ymax = 32
        
        marker = 's'
        
        unit = r'[g/kg]'

    for site in ['point_jefferson', 'lynch_cove_mid']:
        
        if site == 'point_jefferson':
            
            site_label = 'Point Jefferson'
        
        else:
            
            site_label = 'Lynch Cove'
        
        if var == 'DO_mg_L':
            
            plot_df_q50 = odf_use_q50[(odf_use_q50['site'] == site) & (odf_use_q50['var'] == var) & (odf_use_q50['surf_deep'] == 'deep')] #already filtered to loDO
        
            sns.scatterplot(data=plot_df_q50, x='datetime', y = 'val',  ax=ax, color = palette[site], marker=marker)
            
            stat_df = all_stats_filt[(all_stats_filt['site'] == site) & (all_stats_filt['var'] == 'deep_DO_mg_L') & (all_stats_filt['deep_DO_q'] == 'deep_DO_q50') & (all_stats_filt['season'] == 'loDO')]

            x = plot_df_q50['date_ordinal']

            y = plot_df_q50['val']

            x_plot = plot_df_q50['datetime']
             
            B0 = stat_df['B0'].iloc[0]
            
            B1 = stat_df['B1'].iloc[0]
            
            ax.plot([x_plot.min(), x_plot.max()], [B0 + B1*x.min(), B0 + B1*x.max()], alpha =0.7, color = palette[site], linewidth = 2)      
            
            ax.axhline(np.mean([B0 + B1*x.min(), B0 + B1*x.max()]), color = palette[site], linestyle = '--', alpha = 0.5)

            def norm0_1(x):
                return (x - x.min())/ (x.max()-x.min())
            
            x_norm = 2*norm0_1(x)-1

            res = stats.theilslopes(y, x_norm, alpha=0.05)
            
            x_vals = np.array([x_plot.min(), x_plot.max()])
            y_upper = res[1] + res[2] * np.array([x_norm.min(), x_norm.max()])
            y_lower = res[1] + res[3] * np.array([x_norm.min(), x_norm.max()])
            
            ax.fill_between(x_vals, y_lower, y_upper, color=palette[site], alpha=0.2)

        elif var == 'CT':
            
            plot_df = odf_use[(odf_use['site'] == site) & (odf_use['var'] == var) & (odf_use['surf_deep'] == 'deep') & (odf_use['season'] == 'loDO')]
            
            sns.scatterplot(data=plot_df, x='datetime', y = 'val',  ax=ax, color = palette[site], marker=marker)
            
            ax.scatter(x=0, y =0, color = palette[site], marker='o', label = site_label)
            
            stat_df = all_stats_filt[(all_stats_filt['site'] == site) & (all_stats_filt['var'] == 'deep_CT') & (all_stats_filt['deep_DO_q'] == 'deep_DO_q50') & (all_stats_filt['season'] == 'loDO')]

            x = plot_df['date_ordinal']

            y = plot_df['val']

            x_plot = plot_df['datetime']
             
            B0 = stat_df['B0'].iloc[0]
            
            B1 = stat_df['B1'].iloc[0]
            
            ax.plot([x_plot.min(), x_plot.max()], [B0 + B1*x.min(), B0 + B1*x.max()], alpha =0.7, color = palette[site], linewidth = 2)   
            
            ax.axhline(np.mean([B0 + B1*x.min(), B0 + B1*x.max()]), color = palette[site], linestyle = '--', alpha = 0.5)

            def norm0_1(x):
                return (x - x.min())/ (x.max()-x.min())
            
            x_norm = 2*norm0_1(x)-1

            res = stats.theilslopes(y, x_norm, alpha=0.05)
            
            x_vals = np.array([x_plot.min(), x_plot.max()])
            y_upper = res[1] + res[2] * np.array([x_norm.min(), x_norm.max()])
            y_lower = res[1] + res[3] * np.array([x_norm.min(), x_norm.max()])
            
            ax.fill_between(x_vals, y_lower, y_upper, color=palette[site], alpha=0.2)

            
        elif var == 'SA':
            
            plot_df = odf_use[(odf_use['site'] == site) & (odf_use['var'] == var) & (odf_use['surf_deep'] == 'deep') & (odf_use['season'] == 'loDO')]
            
            sns.scatterplot(data=plot_df, x='datetime', y = 'val',  ax=ax, color = palette[site], marker=marker)
            
            stat_df = all_stats_filt[(all_stats_filt['site'] == site) & (all_stats_filt['var'] == 'deep_SA') & (all_stats_filt['deep_DO_q'] == 'deep_DO_q50') & (all_stats_filt['season'] == 'loDO')]

            x = plot_df['date_ordinal']

            y = plot_df['val']

            x_plot = plot_df['datetime']
             
            B0 = stat_df['B0'].iloc[0]
            
            B1 = stat_df['B1'].iloc[0]
            
            ax.plot([x_plot.min(), x_plot.max()], [B0 + B1*x.min(), B0 + B1*x.max()], alpha =0.7, color = palette[site], linewidth = 2)  
            
            ax.axhline(np.mean([B0 + B1*x.min(), B0 + B1*x.max()]), color = palette[site], linestyle = '--', alpha = 0.5)

            def norm0_1(x):
                return (x - x.min())/ (x.max()-x.min())
            
            x_norm = 2*norm0_1(x)-1

            res = stats.theilslopes(y, x_norm, alpha=0.05)
            
            x_vals = np.array([x_plot.min(), x_plot.max()])
            y_upper = res[1] + res[2] * np.array([x_norm.min(), x_norm.max()])
            y_lower = res[1] + res[3] * np.array([x_norm.min(), x_norm.max()])
            
            ax.fill_between(x_vals, y_lower, y_upper, color=palette[site], alpha=0.2)
            
    
    if var == 'DO_mg_L':  
    
        ax.axhspan(0,2, color = 'lightgray', alpha = 0.5, zorder=-5, label='Hypoxia')
        
        ax.legend(loc='lower left')
        
        ax.text(0.0125,0.05, 'c', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')
        
    
    elif var == 'CT':
        
        ax.legend(ncol=2, loc='upper left')
        
        ax.text(0.0125,0.05, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')
        
    else:
        
        ax.text(0.0125,0.05, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')

            
            
    ax.set_ylim(ymin, ymax) 
            
    ax.set_ylabel(label_var + ' ' + unit)
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax.set_xlabel('')
    
handles_CT, labels_CT = axd['CT'].get_legend_handles_labels()

handles_DO_mg_L, labels_DO_mg_L = axd['DO_mg_L'].get_legend_handles_labels()

handles = handles_CT + handles_DO_mg_L

labels = labels_CT + labels_DO_mg_L



fig.legend(
    handles, labels,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.01),  # left side
    ncol=3
    #title='Data Source'
    )

axd['CT'].get_legend().remove()

axd['DO_mg_L'].get_legend().remove()

        
        
#plt.savefig('/Users/dakotamascarenas/Desktop/pltz/paper_1_fig_6.png', bbox_inches='tight', dpi=500, transparent=True)
    
