#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:03:38 2024

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




poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson'] #, 'mb', 'hc', 'ss', 'wb'] # 5 sites + 4 basins


odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)

# %%

odf_use, odf_calc_use = dfun.annualDepthAverageDF(odf_depth_mean, odf_calc_long)

# %%

odf_ds = odf_use[odf_use['var'] == 'SA']


odf_ds_pj = odf_ds[odf_ds['site'] == 'point_jefferson']


odf_ds_pj['val_pj'] = odf_ds_pj['val']


odf_ds = pd.merge(odf_ds, odf_ds_pj[['summer_non_summer', 'surf_deep', 'year', 'val_pj']], how='left', on=['summer_non_summer', 'surf_deep', 'year'])

# %%

odf_ds['val_diff'] = odf_ds['val_pj'] - odf_ds['val']

odf_ds['var'] = 'ds'

odf_ds['val'] = odf_ds['val_diff']

# %%


stats_df = pd.DataFrame()

stat = 'mk_ts'

alpha = 0.05

for site in site_list:
    
    for season in ['summer']:
        
        for depth in ['surf', 'deep']:
    
            for var in odf_ds['var'].unique():
            
                
                mask = (odf_ds['site'] == site) & (odf_ds['summer_non_summer'] == season) & (odf_ds['surf_deep'] == depth) & (odf_ds['var'] == var)
                        
                        
                
                plot_df = odf_ds[mask].dropna()
                
                x = plot_df['year']
                
                #x_plot = plot_df['datetime']
                
                y = plot_df['val']
                

                        
                plot_df['stat'] = stat
                
                reject_null, p_value, Z = dfun.mann_kendall(y, alpha) #dfun
                            
                plot_df['p'] = p_value
        
        
                result = stats.theilslopes(y,x,alpha=alpha)
        
                B1 = result.slope
        
                B0 = result.intercept
                
                plot_df['B1'] = B1

                plot_df['B0'] = B0
                
                plot_df['hi_sB1'] = result.high_slope
                
                plot_df['lo_sB1']  = result.low_slope

                #slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_plot.max().year - x_plot.min().year)
        
                #plot_df['slope_datetime'] = slope_datetime #per year
                
                #slope_datetime_s_hi = (B0 + high_sB1*x.max() - (B0 + high_sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
                
                #slope_datetime_s_lo = (B0 + low_sB1*x.max() - (B0 + low_sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
                
                #plot_df['slope_datetime_s_hi'] = slope_datetime_s_hi #per year
                
                #plot_df['slope_datetime_s_lo'] = slope_datetime_s_lo #per year
                
                if depth != 'all':
                                            
                    plot_df['var'] = plot_df['surf_deep'] + '_' + plot_df['var']
                                                                                        
                plot_df_concat = plot_df[['site','stat','var', 'p', 'hi_sB1', 'lo_sB1', 'B1', 'B0']].head(1) #slope_datetime_unc_cent, slope_datetime_s
                                
                plot_df_concat['summer_non_summer'] = season
    
                stats_df = pd.concat([stats_df, plot_df_concat])
                
# %%

fig, ax = plt.subplots(nrows = 2, figsize=(9,6), sharex=True, sharey=True)

sns.scatterplot(data=odf_ds[(odf_ds['summer_non_summer']=='summer') & (odf_ds['site'] != 'point_jefferson') & (odf_ds['surf_deep'] == 'surf')], x='year', y = 'val', hue = 'site', palette='Set2', ax = ax[0], alpha =0.8, legend =False)

ax[0].set_title(r'$\Delta$S [PJ SA - Site SA], Aug-Nov annual avgs.')


sns.scatterplot(data=odf_ds[(odf_ds['summer_non_summer']=='summer') & (odf_ds['site'] != 'point_jefferson') & (odf_ds['surf_deep'] == 'deep')], x='year', y = 'val', hue = 'site', palette='Set2', ax = ax[1], alpha =0.8)


ax[0].grid(color = 'lightgray', linestyle = '--', alpha=0.3)


ax[1].grid(color = 'lightgray', linestyle = '--', alpha=0.3)

ax[0].set_xlabel('')

ax[1].set_xlabel('')

ax[0].set_ylabel('[PSU]')

ax[1].set_ylabel('[PSU]')


ax[0].text(0.05,0.05, 'surface', transform=ax[0].transAxes, verticalalignment='bottom', fontweight = 'bold', color='k')

ax[1].text(0.05,0.05, 'deep', transform=ax[1].transAxes, verticalalignment='bottom', fontweight = 'bold', color='k')



ax[0].plot([1953, 2023], [-43.6969 + 0.023526*1953, -43.6969 + 0.023526*2023], color = 'orchid', linestyle = 'dashed', linewidth=2, label='+2.35 PSU/century +/- ~2')

ax[0].legend()








plt.savefig('/Users/dakotamascarenas/Desktop/pltz/dS_long_annual_avg_trends.png', bbox_inches='tight', dpi=500, transparent=False)


