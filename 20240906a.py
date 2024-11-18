#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:50:28 2024

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

poly_list = ['ps']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L'] #'SA', 'CT', #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf_ecology_nc = odf[odf['source'] == 'ecology_nc']

odf_ecology_nc['site'] = odf_ecology_nc['source']


odf['site'] = odf['segment']


odf = pd.concat([odf, odf_ecology_nc])


odf_use = dfun.annualAverageDF(odf)

# %%

site_list =  odf['site'].unique()

all_stats_filt = dfun.buildStatsDF(odf_use, site_list, deep_DO_q_list = ['all'], season_list = ['all'], stat_list = ['mk_ts'], depth_list=['all'])

# %%

fig, ax = plt.subplots(figsize=(6, 4))
        
for site in ['ps', 'ecology_nc']:
    
    plot_df = odf_use[(odf_use['site'] == site) & (odf_use['var'] == 'DO_mg_L')]

    
    plot_df['val'] = plot_df['val_mean']
    
    if site == 'ps':
    
        y_spot = 0.1
        
        color = 'black'
    
    elif site == 'ecology_nc':
        
        y_spot = 0.2
        
        color = '#E91E63'
    
    sns.scatterplot(data=plot_df, x='year', y = 'val',  color=color,ax=ax, alpha=0.7)
    
    for idx in plot_df.index:
            
        ax.plot([plot_df.loc[idx,'year'], plot_df.loc[idx,'year']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']],  color=color, alpha =0.7, zorder = -4, linewidth=1)
        
    filt_df = all_stats_filt[(all_stats_filt['site'] == site) & (all_stats_filt['var'] == 'DO_mg_L')]
    

    x = plot_df['date_ordinal']
    
    x_plot = plot_df['year']
    
    y = plot_df['val']
    
    p = filt_df['p'].iloc[0]
        
    B0 = filt_df['B0'].iloc[0]
    
    B1 = filt_df['B1'].iloc[0]
    
    slope_datetime = filt_df['slope_datetime'].iloc[0]
    
    slope_datetime_s_hi = filt_df['slope_datetime_s_hi'].iloc[0]
    
    slope_datetime_s_lo = filt_df['slope_datetime_s_lo'].iloc[0]

    
    if p > 0.05:
        
        color = 'gray'

    
    ax.plot([x_plot.min(), x_plot.max()], [B0 + B1*x.min(), B0 + B1*x.max()], color=color, alpha =0.7, linestyle = 'dashed', linewidth=2)

    ax.text(0.98,y_spot, site + ' theilsen = ' + str(np.round(slope_datetime*100,2)) + '/cent. +' + str(np.round(slope_datetime_s_hi*100 - slope_datetime*100,2)) +'/-' + str(np.round(slope_datetime*100-slope_datetime_s_lo*100,2)), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, color=color, bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

ax.text(0.05, 0.95, 'Puget Sound Annual Average [DO]', horizontalalignment='left', verticalalignment='top', transform = ax.transAxes, fontweight='bold')



ax.set_ylim(0, 12)

ax.axhspan(0,2, color = 'lightgray', alpha = 0.2)

ax.set_ylabel(r'DO [mg/L]')

ax.set_xlabel('')



ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)



plt.tight_layout()


plt.savefig('/Users/dakotamascarenas/Desktop/pltz/allps_ecology_surfdeepDO_timeseries_newcode.png', bbox_inches='tight', dpi=500, transparent=False)



