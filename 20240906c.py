#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:06:04 2024

@author: dakotamascarenas
"""

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




odf_use = odf_depth_mean.copy()

odf_calc_use = odf_calc_long.copy()

all_stats_filt = dfun.buildStatsDF(odf_use, site_list, odf_calc_use=odf_calc_use, odf_depth_mean_deep_DO_percentiles=odf_depth_mean_deep_DO_percentiles)


# %%

c=0

all_stats_filt = all_stats_filt.sort_values(by=['site'])





for site in all_stats_filt['site'].unique():
        
    all_stats_filt.loc[all_stats_filt['site'] == site, 'site_num'] = c
    

    c+=1
    
site_labels = sorted(site_list)


fig, ax = plt.subplots(figsize=(4,3))

marker = 'H'

label = r'$\Delta$ [DO]'

unit = r'[mg/L]/century'


ymin = -3
 
ymax = 3

    
for stat in ['mk_ts']:
    
    for season in ['summer']:
    
        for deep_DO_q in ['deep_DO_q50']:

        
            plot_df = all_stats_filt[(all_stats_filt['stat'] == stat) & (all_stats_filt['summer_non_summer'] == season) & (all_stats_filt['site'].isin(long_site_list)) & (all_stats_filt['deep_DO_q'] == deep_DO_q)]
            
            plot_df = plot_df.sort_values(by=['site'])
            
            plot_df['slope_datetime_cent'] = plot_df['slope_datetime']*100
            
            plot_df['slope_datetime_cent_95hi'] = plot_df['slope_datetime_s_hi']*100
            
            plot_df['slope_datetime_cent_95lo'] = plot_df['slope_datetime_s_lo']*100
            
            
            color = '#E91E63'
             
            marker = 'h'
    
            
            sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_sol'], x= 'site_num', y = 'slope_datetime_cent_95hi', color = color, marker=marker, ax = ax, s= 10, legend=False)
    
            sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_sol'], x= 'site_num', y = 'slope_datetime_cent_95lo', color = color, marker=marker, ax = ax, s= 10, legend=False)
    
            sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_sol'], x= 'site_num', y = 'slope_datetime_cent', color = color, marker=marker, ax = ax, s =50, label='Deep Saturation')
            
            
            
            color = '#673AB7'
            
            marker = 'o'
            
            
            sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_mg_L'], x= 'site_num', y = 'slope_datetime_cent_95hi', color = color, marker=marker, ax = ax, s= 10, legend=False)
    
            sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_mg_L'], x= 'site_num', y = 'slope_datetime_cent_95lo', color = color, marker=marker, ax = ax, s= 10, legend=False)
    
            sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_mg_L'], x= 'site_num', y = 'slope_datetime_cent', color = color, marker=marker, ax = ax, s =50, label='Deep Concentration')
            
            
        
            
            for idx in plot_df.index:
                
                if plot_df.loc[idx,'var'] == 'deep_DO_sol':
                    
                    color = '#E91E63'
                    
                    ax.plot([plot_df.loc[idx,'site_num'], plot_df.loc[idx,'site_num']],[plot_df.loc[idx,'slope_datetime_cent_95lo'], plot_df.loc[idx,'slope_datetime_cent_95hi']], color=color, alpha =0.7, zorder = -4, linewidth=1)
            
                elif plot_df.loc[idx,'var'] == 'deep_DO_mg_L':
                    
                    color = '#673AB7'
                    
                    ax.plot([plot_df.loc[idx,'site_num'], plot_df.loc[idx,'site_num']],[plot_df.loc[idx,'slope_datetime_cent_95lo'], plot_df.loc[idx,'slope_datetime_cent_95hi']], color=color, alpha =0.7, zorder = -4, linewidth=1)
            
            
                                
            ax.text(0.05,0.05, label, transform=ax.transAxes, verticalalignment='bottom', fontweight = 'bold')
                
            
            ax.set_xticks([20,21,22,23,24], ['1', '2', '3', '4', '5'])
            
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
                                    
            ax.axhline(0, color='gray', linestyle = '--', zorder = -5)
            
            ax.set_ylabel(unit, wrap=True)
            
            ax.set_xlabel('')
            
            ax.set_ylim(ymin, ymax)
            
            
fig.tight_layout()
            
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + 'longsites_' + stat + '_' + season + '_' + deep_DO_q + '_slopes_deepDOsoldeepDO_newcode.png', dpi=500,transparent=False, bbox_inches='tight')