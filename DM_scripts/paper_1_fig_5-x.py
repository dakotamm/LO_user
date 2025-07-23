#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:38:42 2024

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

site_list =  odf['site'].unique()




odf_use = odf_depth_mean.copy()

odf_calc_use = odf_calc_long.copy()

all_stats_filt = dfun.buildStatsDF(odf_use, site_list, odf_calc_use=odf_calc_use, odf_depth_mean_deep_DO_percentiles=odf_depth_mean_deep_DO_percentiles)



# %%

c=0

all_stats_filt.loc[all_stats_filt['site'] == 'point_jefferson', 'site_label'] = 'PJ'

all_stats_filt.loc[all_stats_filt['site'] == 'near_seattle_offshore', 'site_label'] = 'NS'

all_stats_filt.loc[all_stats_filt['site'] == 'saratoga_passage_mid', 'site_label'] = 'SP'

all_stats_filt.loc[all_stats_filt['site'] == 'carr_inlet_mid', 'site_label'] = 'CI'

all_stats_filt.loc[all_stats_filt['site'] == 'lynch_cove_mid', 'site_label'] = 'LC'


all_stats_filt.loc[all_stats_filt['site'] == 'point_jefferson', 'site_type'] = 'Main Basin'

all_stats_filt.loc[all_stats_filt['site'] == 'near_seattle_offshore', 'site_type'] = 'Main Basin'

all_stats_filt.loc[all_stats_filt['site'] == 'saratoga_passage_mid', 'site_type'] = 'Sub-Basins'

all_stats_filt.loc[all_stats_filt['site'] == 'carr_inlet_mid', 'site_type'] = 'Sub-Basins'

all_stats_filt.loc[all_stats_filt['site'] == 'lynch_cove_mid', 'site_type'] = 'Sub-Basins'


all_stats_filt.loc[all_stats_filt['site'] == 'point_jefferson', 'site_num'] = 1

all_stats_filt.loc[all_stats_filt['site'] == 'near_seattle_offshore', 'site_num'] = 2

all_stats_filt.loc[all_stats_filt['site'] == 'saratoga_passage_mid', 'site_num'] = 3

all_stats_filt.loc[all_stats_filt['site'] == 'carr_inlet_mid', 'site_num'] = 4

all_stats_filt.loc[all_stats_filt['site'] == 'lynch_cove_mid', 'site_num'] = 5

# %%

fig, ax = plt.subplots(figsize=(4,3))

#label = r'$\frac{d}{dt}$ [DO]'

unit = r'[mg/L]/century'


ymin = -1.5
 
ymax = 1.5


palette = {'Main Basin':'#e04256', 'Sub-Basins':'#4565e8'}

palette_label = {'Main Basin Observed DO Trend':'#e04256', 'Sub-Basins Observed DO Trend':'#4565e8'}


# marker="$\circ$"


    
for stat in ['mk_ts']:
    
    for season in ['loDO']:
    
        for deep_DO_q in ['deep_DO_q50']:

        
            plot_df = all_stats_filt[(all_stats_filt['stat'] == stat) & (all_stats_filt['season'] == season) & (all_stats_filt['site'].isin(long_site_list)) & (all_stats_filt['deep_DO_q'] == deep_DO_q) & (all_stats_filt['var'].isin(['deep_DO_sol','deep_DO_mg_L']))]
            
            plot_df = plot_df.sort_values(by=['site'])
            
            plot_df['slope_datetime_cent'] = plot_df['slope_datetime']*100
            
            plot_df['slope_datetime_cent_95hi'] = plot_df['slope_datetime_s_hi']*100
            
            plot_df['slope_datetime_cent_95lo'] = plot_df['slope_datetime_s_lo']*100
            
            plot_df.loc[plot_df['site_type'] == 'Main Basin', 'site_type_label'] = 'Main Basin Observed DO Trend'
            
            plot_df.loc[plot_df['site_type'] == 'Sub-Basins', 'site_type_label'] = 'Sub-Basins Observed DO Trend'

            
            
            marker = "$\circ$"
    
            
            # sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_sol'], x= 'site_num', y = 'slope_datetime_cent_95hi', color = '#dd9404', marker=marker, ax = ax, s= 20, legend=False)
    
            # sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_sol'], x= 'site_num', y = 'slope_datetime_cent_95lo', color = '#dd9404', marker=marker, ax = ax, s= 20, legend=False)
    
            sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_sol'], x= 'site_num', y = 'slope_datetime_cent', color = '#dd9404', marker=marker, ax = ax, s =150, label= 'Expected DO Trend')
            
            
            marker = 'o'
            
            sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_mg_L'], x= 'site_num', y = 'slope_datetime_cent_95hi', hue = 'site_type', palette=palette, marker=marker, ax = ax, s= 10, legend=False)
    
            sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_mg_L'], x= 'site_num', y = 'slope_datetime_cent_95lo', hue = 'site_type', palette=palette, marker=marker, ax = ax, s= 10, legend=False)
               
            sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_mg_L'], x= 'site_num', y = 'slope_datetime_cent', hue = 'site_type_label', palette=palette_label, marker=marker, ax = ax, s =50)
            
        
            
        
            
            for idx in plot_df.index:
                
                if plot_df.loc[idx, 'var'] == 'deep_DO_mg_L':
                
                    if plot_df.loc[idx,'site_type'] == 'Main Basin':
                        
                        color = palette['Main Basin']
                                            
                        ax.plot([plot_df.loc[idx,'site_num'], plot_df.loc[idx,'site_num']],[plot_df.loc[idx,'slope_datetime_cent_95lo'], plot_df.loc[idx,'slope_datetime_cent_95hi']], color=color, alpha =0.7, zorder = -4, linewidth=1)
                
                    else:
                        
                        color = palette['Sub-Basins']
                        
                        ax.plot([plot_df.loc[idx,'site_num'], plot_df.loc[idx,'site_num']],[plot_df.loc[idx,'slope_datetime_cent_95lo'], plot_df.loc[idx,'slope_datetime_cent_95hi']], color=color, alpha =0.7, zorder = -4, linewidth=1)
            
                # else:
                    
                #     color = '#dd9404'
                    
                #     ax.plot([plot_df.loc[idx,'site_num'], plot_df.loc[idx,'site_num']],[plot_df.loc[idx,'slope_datetime_cent_95lo'], plot_df.loc[idx,'slope_datetime_cent_95hi']], color=color, alpha =0.7, zorder = -3, linewidth=1)

            
                                
            #ax.text(0.05,0.05, label, transform=ax.transAxes, verticalalignment='bottom', fontweight = 'bold')
                
            
            ax.set_xticks([1,2,3,4,5],['PJ', 'NS', 'SP', 'CI', 'LC'])
            
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
                                    
            ax.axhline(0, color='gray', linestyle = '--', zorder = -5)
            
            ax.set_ylabel(unit, wrap=True)
            
            ax.set_xlabel('')
            
            ax.set_ylim(ymin, ymax)
            
            ax.legend()
            
            
fig.tight_layout()
            
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/paper_1_fig_5.png', dpi=500,transparent=True, bbox_inches='tight')