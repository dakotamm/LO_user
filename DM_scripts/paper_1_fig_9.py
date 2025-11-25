#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 16:37:23 2025

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




poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson'] #, 'mb', 'hc', 'ss', 'wb'] # 5 sites + 4 basins

#poly_list = ['ps']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_his', 'kc_whidbeyBasin', 'nceiSalish', 'kc_pointJefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


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

# c=0

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

all_stats_filt.loc[all_stats_filt['site'] == 'saratoga_passage_mid', 'site_num'] = 4

all_stats_filt.loc[all_stats_filt['site'] == 'carr_inlet_mid', 'site_num'] = 3

all_stats_filt.loc[all_stats_filt['site'] == 'lynch_cove_mid', 'site_num'] = 5

# %%

all_stats_filt.loc[all_stats_filt['var'].isin(['surf_CT', 'surf_SA', 'surf_DO_mg_L', 'surf_DO_sol']), 'surf_deep'] = 'surf'

all_stats_filt.loc[all_stats_filt['var'].isin(['deep_CT', 'deep_SA', 'deep_DO_mg_L', 'deep_DO_sol']), 'surf_deep'] = 'deep'

all_stats_filt.loc[all_stats_filt['var'].isin(['surf_CT', 'deep_CT']), 'var'] = 'CT'

all_stats_filt.loc[all_stats_filt['var'].isin(['surf_SA', 'deep_SA']), 'var'] = 'SA'

all_stats_filt.loc[all_stats_filt['var'].isin(['surf_DO_mg_L', 'deep_DO_mg_L']), 'var'] = 'DO_mg_L'

all_stats_filt.loc[all_stats_filt['var'].isin(['surf_DO_sol', 'deep_DO_sol']), 'var'] = 'DO_sol'



# %%

all_stats_filt.loc[all_stats_filt['surf_deep'] == 'surf', 'depth_label'] = 'Surface'

all_stats_filt.loc[all_stats_filt['surf_deep'] == 'deep', 'depth_label'] = 'Bottom'

all_stats_filt.loc[all_stats_filt['var'] == 'CT', 'var_label'] = '[°C]'

all_stats_filt.loc[all_stats_filt['var'] == 'SA', 'var_label'] = '[g/kg]'

all_stats_filt.loc[all_stats_filt['var'] == 'DO_mg_L', 'var_label'] = '[mg/L]'

all_stats_filt.loc[all_stats_filt['var'] == 'DO_sol', 'var_label'] = '[mg/L]'

all_stats_filt.loc[all_stats_filt['season'] == 'grow', 'season_label'] = 'Spring (Apr-Jul)'

all_stats_filt.loc[all_stats_filt['season'] == 'loDO', 'season_label'] = 'Low-DO (Aug-Nov)'

all_stats_filt.loc[all_stats_filt['season'] == 'winter', 'season_label'] = 'Winter (Dec-Mar)'





# %%
# palette = {'Surface': 'white', 'Bottom': 'gray'}


# #palette = {'point_jefferson': 'red', 'near_seattle_offshore': 'orange', 'carr_inlet_mid':'blue', 'saratoga_passage_mid':'purple', 'lynch_cove_mid': 'orchid'}

# linecolors = {'Main Basin':'#e04256', 'Sub-Basins':'#4565e8'}

# #linecolors = {'point_jefferson': '#e04256', 'near_seattle_offshore': '#e04256', 'carr_inlet_mid':'#4565e8', 'saratoga_passage_mid':'#4565e8', 'lynch_cove_mid': '#4565e8'}

# #edgecolors = {'point_jefferson': 'k', 'near_seattle_offshore': 'k', 'carr_inlet_mid':'gray', 'saratoga_passage_mid':'gray', 'lynch_cove_mid': 'gray'}

# jitter = {'Surface': -0.15, 'Bottom': 0.15}

# markers = {'DO_mg_L': 'o', 'SA': 's', 'CT': '^'}

# plot_labels = {'Winter (Dec-Mar)': 'a', 'Grow (Apr-Jul)': 'b', 'Lo-DO (Aug-Nov)':'c'}
 

# mosaic = [['Winter (Dec-Mar)', 'Grow (Apr-Jul)', 'Lo-DO (Aug-Nov)']]
 
# #fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=True, figsize=(9,3), layout='constrained', gridspec_kw=dict(wspace=0.1, hspace=0.1))

# for var in ['DO_mg_L']:
    
#     fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=True, figsize=(9,2.5), layout='constrained', gridspec_kw=dict(wspace=0.1, hspace=0.1))
    
#     for season in ['Winter (Dec-Mar)', 'Grow (Apr-Jul)', 'Lo-DO (Aug-Nov)']:
        
#         ax_name = season
    
#         ax = axd[ax_name]
        
#         for site in long_site_list:
            
#             for depth in ['Surface', 'Bottom']:
                
#                 plot_df = all_stats_filt[(all_stats_filt['var'] == var) & (all_stats_filt['season_label'] == season) & (all_stats_filt['site'] == site) & (all_stats_filt['depth_label'] == depth)]
                
#                 plot_df['slope_datetime_cent'] = plot_df['slope_datetime']*100
                 
#                 plot_df['slope_datetime_cent_95hi'] = plot_df['slope_datetime_s_hi']*100
                
#                 plot_df['slope_datetime_cent_95lo'] = plot_df['slope_datetime_s_lo']*100
                
#                 # ax.scatter(plot_df['site_num'] + jitter[depth], plot_df['slope_datetime_cent'], color=palette[depth], marker=markers[var], s=50, label = depth)
                                
#                 ax.scatter(plot_df['site_num'] + jitter[depth], plot_df['slope_datetime_cent'], color=palette[depth], edgecolors='k', marker=markers[var], s=30, label=depth) #, marker=markers[depth], edgecolors=edgecolors[site])
                
#                 ax.plot([plot_df['site_num'] + jitter[depth], plot_df['site_num'] + jitter[depth]],[plot_df['slope_datetime_cent_95lo'], plot_df['slope_datetime_cent_95hi']], color=linecolors[plot_df['site_type'].iloc[0]], alpha =1, zorder = -5, linewidth=1, label=plot_df['site_type'].iloc[0])
        
#                 plot_df_ = all_stats_filt[(all_stats_filt['season_label'] == season) & (all_stats_filt['var'] == 'DO_sol') & (all_stats_filt['site'] == site) & (all_stats_filt['depth_label'] == depth)]
                
#                 # plot_df_ = plot_df_.sort_values(by=['site'])
                
#                 plot_df_['slope_datetime_cent'] = plot_df_['slope_datetime']*100
                
#                 plot_df_['slope_datetime_cent_95hi'] = plot_df_['slope_datetime_s_hi']*100
                
#                 plot_df_['slope_datetime_cent_95lo'] = plot_df_['slope_datetime_s_lo']*100
                
#                 # plot_df.loc[plot_df['site_type'] == 'Main Basin', 'site_type_label'] = 'Main Basin Trend'
                
#                 # plot_df.loc[plot_df['site_type'] == 'Sub-Basins', 'site_type_label'] = 'Sub-Basins Trend'

                
                
#                 marker_ = 'o'
                
#                 #marker_ = "$\circ$"
                
#                 color_ = '#dd9404'
        
                
#                 # sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_sol'], x= 'site_num', y = 'slope_datetime_cent_95hi', color = '#dd9404', marker=marker, ax = ax, s= 20, legend=False)
        
#                 # sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_sol'], x= 'site_num', y = 'slope_datetime_cent_95lo', color = '#dd9404', marker=marker, ax = ax, s= 20, legend=False)
        
#                 # sns.scatterplot(data = plot_df_, x= 'site_num', y = 'slope_datetime_cent', color = '#dd9404', marker=marker_, ax = ax, s =150, label= 'Solubility-Based Trend')
                
#                 ax.scatter(plot_df_['site_num'] + jitter[depth], plot_df_['slope_datetime_cent'], color=color_, marker=marker_, s=50, alpha= 1, label= 'Sol.-Based Trend', zorder =-6) #, marker=markers[depth], edgecolors=edgecolors[site])

#                 ax.plot([plot_df_['site_num'] + jitter[depth], plot_df_['site_num'] + jitter[depth]],[plot_df_['slope_datetime_cent_95lo'], plot_df_['slope_datetime_cent_95hi']], color=color_, alpha =1, zorder = -6, linewidth=2)

                
#                 #sns.scatterplot(data = plot_df, x= 'site_num', y = 'slope_datetime_cent', hue = 'site_type_label', palette=palette_, marker=marker, ax = ax, s =50) 

           
                
#         ax.text(0.05,0.05, plot_labels[season] + ' ' + season, transform=ax.transAxes, verticalalignment='bottom', fontweight = 'bold', color='k')
         
#         ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3, zorder = -6)
    
#         ax.axhline(0, color='gray', linestyle = '--', zorder = -5) 
    
#         ax.set_ylabel(all_stats_filt[all_stats_filt['var'] == var]['var_label'].iloc[0] + '/century')
        
#         ax.set_xticks([1,2,3,4,5],['PJ', 'NS', 'CI', 'SP', 'LC'])
        
#         if season == 'Winter (Dec-Mar)': 
            
#             handles, labels = ax.get_legend_handles_labels()
            
#             selected_handles = [handles[1], handles[-2]]
#             selected_labels = [labels[1], labels[-2]]
            
            
#             ax.legend(selected_handles, selected_labels, loc ='upper left')
            
#             #ax.set_ylim(ymax=8)
            
#             # ax.legend()
            
#         elif season == 'Grow (Apr-Jul)':
            
#             handles, labels = ax.get_legend_handles_labels()
                
#             selected_handles = [handles[2]]
#             selected_labels = [labels[2]]
            
#             ax.legend(selected_handles, selected_labels, loc ='upper left')
            
#             ax.set_ylabel('')
            
#         else:
            
#             handles, labels = ax.get_legend_handles_labels()
                
#             selected_handles = [handles[0], handles[3]]
#             selected_labels = [labels[0], labels[3]]
            
#             ax.legend(selected_handles, selected_labels)
            
#             ax.set_ylabel('')
            
#             #ax.legend()

        
#         #ax.axvline(2.5, color = 'gray', linestyle = '--', zorder = -5)
        
#     if var == 'CT':
        
#         ax.set_ylim(ymin=-2)
        
#     # elif var == 'DO_mg_L':
        
#     #     ax.set_ylim(ymin=-5, ymax=7)
    
#     # if var == 'CT':  
        
#     #     ax.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left')
        
    
#     plt.savefig('/Users/dakotamascarenas/Desktop/pltz/paper_1_fig_10_surfdeep_allseason.png', dpi=500, transparent=True)
    

# %%
palette = {'Surface': 'white', 'Bottom': 'gray'}

red =     "#EF5E3C"   # warm orange-red ##ff4040 #e04256

blue =     "#3A59B3"  # deep blue #4565e8

yellow =     "#C7C445"  # yellow-green '#dd9404'


#palette = {'point_jefferson': 'red', 'near_seattle_offshore': 'orange', 'carr_inlet_mid':'blue', 'saratoga_passage_mid':'purple', 'lynch_cove_mid': 'orchid'}

linecolors = {'Main Basin':'k', 'Sub-Basins':'k'}

#linecolors = {'point_jefferson': '#e04256', 'near_seattle_offshore': '#e04256', 'carr_inlet_mid':'#4565e8', 'saratoga_passage_mid':'#4565e8', 'lynch_cove_mid': '#4565e8'}

#edgecolors = {'point_jefferson': 'k', 'near_seattle_offshore': 'k', 'carr_inlet_mid':'gray', 'saratoga_passage_mid':'gray', 'lynch_cove_mid': 'gray'}

jitter = {'Surface': -0.15, 'Bottom': 0}

markers = {'DO_mg_L': 'o', 'SA': 's', 'CT': '^'}

plot_labels = {'Winter (Dec-Mar)': 'a', 'Spring (Apr-Jul)': 'b', 'Low-DO (Aug-Nov)':'c'}
 

mosaic = [['Winter (Dec-Mar)', 'Spring (Apr-Jul)', 'Low-DO (Aug-Nov)']]
 
#fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=True, figsize=(9,3), layout='constrained', gridspec_kw=dict(wspace=0.1, hspace=0.1))

for var in ['DO_mg_L']:
    
    fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=True, figsize=(9,2.5), layout='constrained', gridspec_kw=dict(wspace=0.1, hspace=0.1))
    
    for season in ['Winter (Dec-Mar)', 'Spring (Apr-Jul)', 'Low-DO (Aug-Nov)']:
        
        ax_name = season
    
        ax = axd[ax_name]
        
        for site in long_site_list:
            
            for depth in ['Bottom']:
                
                plot_df = all_stats_filt[(all_stats_filt['var'] == var) & (all_stats_filt['season_label'] == season) & (all_stats_filt['site'] == site) & (all_stats_filt['depth_label'] == depth)]
                
                plot_df['slope_datetime_cent'] = plot_df['slope_datetime']*100
                 
                plot_df['slope_datetime_cent_95hi'] = plot_df['slope_datetime_s_hi']*100
                
                plot_df['slope_datetime_cent_95lo'] = plot_df['slope_datetime_s_lo']*100
                
                # ax.scatter(plot_df['site_num'] + jitter[depth], plot_df['slope_datetime_cent'], color=palette[depth], marker=markers[var], s=50, label = depth)
                                
                ax.scatter(plot_df['site_num'] + jitter[depth], plot_df['slope_datetime_cent'], color=palette[depth], edgecolors='k', marker=markers[var], s=50, label=depth) #, marker=markers[depth], edgecolors=edgecolors[site])
                
                ax.plot([plot_df['site_num'] + jitter[depth], plot_df['site_num'] + jitter[depth]],[plot_df['slope_datetime_cent_95lo'], plot_df['slope_datetime_cent_95hi']], color=linecolors[plot_df['site_type'].iloc[0]], alpha =1, zorder = -5, linewidth=1, label=plot_df['site_type'].iloc[0])
        
                plot_df_ = all_stats_filt[(all_stats_filt['season_label'] == season) & (all_stats_filt['var'] == 'DO_sol') & (all_stats_filt['site'] == site) & (all_stats_filt['depth_label'] == depth)]
                
                # plot_df_ = plot_df_.sort_values(by=['site'])
                
                plot_df_['slope_datetime_cent'] = plot_df_['slope_datetime']*100
                
                plot_df_['slope_datetime_cent_95hi'] = plot_df_['slope_datetime_s_hi']*100
                
                plot_df_['slope_datetime_cent_95lo'] = plot_df_['slope_datetime_s_lo']*100
                
                # plot_df.loc[plot_df['site_type'] == 'Main Basin', 'site_type_label'] = 'Main Basin Trend'
                
                # plot_df.loc[plot_df['site_type'] == 'Sub-Basins', 'site_type_label'] = 'Sub-Basins Trend'

                
                
                marker_ = 'o'
                
                #marker_ = "$\circ$"
                
                color_ = yellow
        
                
                # sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_sol'], x= 'site_num', y = 'slope_datetime_cent_95hi', color = '#dd9404', marker=marker, ax = ax, s= 20, legend=False)
        
                # sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_sol'], x= 'site_num', y = 'slope_datetime_cent_95lo', color = '#dd9404', marker=marker, ax = ax, s= 20, legend=False)
        
                # sns.scatterplot(data = plot_df_, x= 'site_num', y = 'slope_datetime_cent', color = '#dd9404', marker=marker_, ax = ax, s =150, label= 'Solubility-Based Trend')
                
                ax.scatter(plot_df_['site_num'] + jitter[depth], plot_df_['slope_datetime_cent'], color=color_, marker=marker_, s=100, alpha= 1, label= 'Sol.-Based Trend', zorder =6) #, marker=markers[depth], edgecolors=edgecolors[site])

                ax.plot([plot_df_['site_num'] + jitter[depth], plot_df_['site_num'] + jitter[depth]],[plot_df_['slope_datetime_cent_95lo'], plot_df_['slope_datetime_cent_95hi']], color=color_, alpha =1, zorder = 6, linewidth=2)

                
                #sns.scatterplot(data = plot_df, x= 'site_num', y = 'slope_datetime_cent', hue = 'site_type_label', palette=palette_, marker=marker, ax = ax, s =50) 

           
                
        ax.text(0.05,0.05, plot_labels[season], transform=ax.transAxes, verticalalignment='bottom', fontweight = 'bold', color='k', fontsize=14)
          
        ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3, zorder = -7)
    
        ax.axhline(0, color='gray', linestyle = '--', zorder = -7) 
    
        ax.set_ylabel(all_stats_filt[all_stats_filt['var'] == var]['var_label'].iloc[0] + '/century')
        
        ax.set_xticks([1,2,3,4,5],['PJ', 'NS', 'CI', 'SP', 'LC'])
        
        ax.set_title(season, fontweight='bold', fontsize=10)
        
        ax.set_ylim(-2,1)
        
        if season == 'Winter (Dec-Mar)': 
            
            # handles, labels = ax.get_legend_handles_labels()
            
            # selected_handles = [handles[1], handles[-2]] 
            # selected_labels = [labels[1], labels[-2]]
            
             
            # ax.legend(selected_handles, selected_labels, loc ='upper left', fontsize=12)
            
            ax.set_ylim(ymax=3)
            
            # ax.legend()
            
        elif season == 'Spring (Apr-Jul)':
            
            handles, labels = ax.get_legend_handles_labels()
                
            selected_handles_ = [handles[2]]
            selected_labels_ = [labels[2]] 
            
            ax.legend(selected_handles_, selected_labels_, loc ='upper left',  fontsize=12)
            
            ax.set_ylabel('')
            
        else:
            
            # handles, labels = ax.get_legend_handles_labels()
                
            # selected_handles = [handles[0], handles[3]]
            # selected_labels = [labels[0], labels[3]]
             
            # ax.legend(selected_handles, selected_labels)
            
            ax.set_ylabel('')
            
            #ax.legend()

        
        #ax.axvline(2.5, color = 'gray', linestyle = '--', zorder = -5)
        
    if var == 'CT':
        
        ax.set_ylim(ymin=-2)
        
    # elif var == 'DO_mg_L':
        
    #     ax.set_ylim(ymin=-5, ymax=7)
    
    # if var == 'CT':  
        
    #     ax.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left')
    
    handles = selected_handles_ #selected_handles +  ... 

    labels = selected_labels_ # selected_labels + ...

    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.01),  # left side
        ncol=len(handles)
        #title='Data Source'
        )

    #axd['Winter (Dec-Mar)'].get_legend().remove()

    axd['Spring (Apr-Jul)'].get_legend().remove()
        
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/paper_1_fig_9.png', bbox_inches='tight', dpi=500, transparent=True)


# %%

plot_df_ = all_stats_filt[(all_stats_filt['var'] == 'DO_sol') & (all_stats_filt['site'].isin(long_site_list)) & (all_stats_filt['surf_deep'] == 'deep') & (all_stats_filt['season'] != 'allyear')]

plot_df_['slope_datetime_sat'] = plot_df_['slope_datetime']

plot_df = all_stats_filt[(all_stats_filt['var'] == 'DO_mg_L') & (all_stats_filt['site'].isin(long_site_list)) & (all_stats_filt['surf_deep'] == 'deep') & (all_stats_filt['season'] != 'allyear')]

plot_df = plot_df.sort_values(by=['site']).reset_index()

plot_df = pd.merge(plot_df, plot_df_[['site', 'season', 'slope_datetime_sat']], on=['site', 'season'], how='left')

plot_df['pct_expl'] = plot_df['slope_datetime_sat']/plot_df['slope_datetime']