#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:39:49 2024

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







c=2

        
    
mosaic = [['loDO_deep_CT', 'loDO_deep_SA', 'loDO_deep_DO_mg_L'],
          ['winter_deep_CT', 'winter_deep_SA', 'winter_deep_DO_mg_L'],
          ['grow_deep_CT', 'grow_deep_SA', 'grow_deep_DO_mg_L'],
          ['loDO_surf_CT', 'loDO_surf_SA', 'loDO_surf_DO_mg_L'],
            ['winter_surf_CT', 'winter_surf_SA', 'winter_surf_DO_mg_L'],
            ['grow_surf_CT', 'grow_surf_SA', 'grow_surf_DO_mg_L']]

fig, axd = plt.subplot_mosaic(mosaic, figsize=(9,11), layout='constrained', sharex=True, gridspec_kw=dict(wspace=0.1))

    
for season in ['loDO', 'winter', 'grow']:
    
    for depth  in ['deep', 'surf']:

        if season == 'loDO':
            
            label_season = 'Aug-Nov'
        
        elif season == 'winter':
            
            label_season = 'Dec-Mar'
            
        elif season == 'grow':
            
            label_season = 'Apr-Jul'
    
        for var in [depth +'_CT', depth+'_SA', depth+'_DO_mg_L']: #, 'deep_DO_mg_L', 'deep_CT', 'deep_SA']:
            
            for stat in ['mk_ts']:
                        
                for deep_DO_q in ['deep_DO_q50']:
                    
                    if 'surf' in var:
                        
                        label_depth = 'Surf.'
                        
                        #color = '#E91E63'
                        
                    else:
                        
                        label_depth = 'Deep'
                        
                        #color = '#673AB7'
        
                    
                    if 'DO' in var:
                        
                        if depth == 'deep':
                        
                            if season == 'loDO':
                            
                                label_var = 'c ' + label_season + ' ' + label_depth + ' [DO]'
                                
                            elif season == 'winter':
                                
                                label_var = 'f ' + label_season + ' ' + label_depth + ' [DO]'
                                
                            elif season == 'grow':
                                
                                label_var = 'i ' + label_season + ' ' + label_depth + ' [DO]'
                                
                        else:
                            
                            if season == 'loDO':
                            
                                label_var = 'l ' + label_season + ' ' + label_depth + ' [DO]'
                                
                            elif season == 'winter':
                                
                                label_var = 'o ' + label_season + ' ' + label_depth + ' [DO]'
                                
                            elif season == 'grow':
                                
                                label_var = 'r ' + label_season + ' ' + label_depth + ' [DO]'
                                                
                        ymin = -3
                        
                        ymax = 3
                            
                        marker = 'o'
                        
                        unit = r'[mg/L]/century'
                    
                    elif 'CT' in var:
                        
                        if depth == 'deep':
                        
                            if season == 'loDO':
                            
                                label_var = 'a ' + label_season + ' ' + label_depth + ' Temp.'
                                
                            elif season == 'winter':
                                
                                label_var = 'd ' + label_season + ' ' + label_depth + ' Temp.'
                                
                            elif season == 'grow':
                                
                                label_var = 'g ' + label_season + ' ' + label_depth + ' Temp.'
                                
                        else:
                            
                            if season == 'loDO':
                            
                                label_var = 'j ' + label_season + ' ' + label_depth + ' Temp.'
                                
                            elif season == 'winter':
                                
                                label_var = 'm ' + label_season + ' ' + label_depth + ' Temp.'
                                
                            elif season == 'grow':
                                
                                label_var = 'p ' + label_season + ' ' + label_depth + ' Temp.'
                        
                                                 
                        
                        ymin = -5
                        
                        ymax = 5
                            
                        
                        marker = 'D'
                        
                        unit = r'[$^{\circ}$C]/century'
                    
                    else:
                        
                        if depth == 'deep':
                        
                            if season == 'loDO':
                            
                                label_var = 'b ' + label_season + ' ' + label_depth + ' Sal.'
                                
                            elif season == 'winter':
                                
                                label_var = 'e ' + label_season + ' ' + label_depth + ' Sal.'
                                
                            elif season == 'grow':
                                
                                label_var = 'h ' + label_season + ' ' + label_depth + ' Sal.'
                                
                        else:
                            
                            if season == 'loDO':
                            
                                label_var = 'k ' + label_season + ' ' + label_depth + ' Sal.'
                                
                            elif season == 'winter':
                                
                                label_var = 'n ' + label_season + ' ' + label_depth + ' Sal.'
                                
                            elif season == 'grow':
                                
                                label_var = 'q ' + label_season + ' ' + label_depth + ' Sal.'
                    
                            
                        ymin = -4
                        
                        ymax = 4
                        
                        marker = 's'
                        
                        unit = r'[g/kg]/century'
        
                    ax = axd[season + '_' + var]
        
                    plot_df = all_stats_filt[(all_stats_filt['stat'] == stat) & (all_stats_filt['var'] == var) & (all_stats_filt['site'].isin(long_site_list)) & (all_stats_filt['deep_DO_q'] == deep_DO_q) & (all_stats_filt['season'] == season)]
                    
                    plot_df = plot_df.sort_values(by=['site']).reset_index()
                    
                    plot_df['slope_datetime_cent'] = plot_df['slope_datetime']*100
                    
                    plot_df['slope_datetime_cent_95hi'] = plot_df['slope_datetime_s_hi']*100
                    
                    plot_df['slope_datetime_cent_95lo'] = plot_df['slope_datetime_s_lo']*100
                    
                    
                    palette = {'Main Basin':'#e04256', 'Sub-Basins':'#4565e8'}
                    
                    palette_ = {'Main Basin Trend':'#e04256', 'Sub-Basins Trend':'#4565e8'}
        
        
                    
                    sns.scatterplot(data = plot_df, x= 'site_num', y = 'slope_datetime_cent_95hi', hue ='site_type', palette = palette, marker=marker, ax = ax, s= 10, legend=False)
            
                    sns.scatterplot(data = plot_df, x= 'site_num', y = 'slope_datetime_cent_95lo', hue ='site_type', palette = palette, marker=marker, ax = ax, s= 10, legend=False)
                    
                    
                    if var == depth + '_DO_mg_L':
                        
                        plot_df_ = all_stats_filt[(all_stats_filt['stat'] == stat) & (all_stats_filt['season'] == season) & (all_stats_filt['var'] == depth +'_DO_sol') & (all_stats_filt['site'].isin(long_site_list)) & (all_stats_filt['deep_DO_q'] == deep_DO_q)]
                        
                        plot_df_ = plot_df_.sort_values(by=['site'])
                        
                        plot_df_['slope_datetime_cent'] = plot_df_['slope_datetime']*100
                        
                        plot_df_['slope_datetime_cent_95hi'] = plot_df_['slope_datetime_s_hi']*100
                        
                        plot_df_['slope_datetime_cent_95lo'] = plot_df_['slope_datetime_s_lo']*100
                        
                        plot_df.loc[plot_df['site_type'] == 'Main Basin', 'site_type_label'] = 'Main Basin Trend'
                        
                        plot_df.loc[plot_df['site_type'] == 'Sub-Basins', 'site_type_label'] = 'Sub-Basins Trend'
        
                        
                        
                        marker_ = "$\circ$"
                
                        
                        # sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_sol'], x= 'site_num', y = 'slope_datetime_cent_95hi', color = '#dd9404', marker=marker, ax = ax, s= 20, legend=False)
                
                        # sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_sol'], x= 'site_num', y = 'slope_datetime_cent_95lo', color = '#dd9404', marker=marker, ax = ax, s= 20, legend=False)
                
                        if (depth == 'deep') & (season == 'loDO'):
                
                            sns.scatterplot(data = plot_df_, x= 'site_num', y = 'slope_datetime_cent', color = '#dd9404', marker=marker_, ax = ax, s =150, label= 'Expected DO Trend')

                            
                            sns.scatterplot(data = plot_df, x= 'site_num', y = 'slope_datetime_cent', hue = 'site_type_label', palette=palette_, marker=marker, ax = ax, s =50)
                        
                        else:
                            
                            sns.scatterplot(data = plot_df_, x= 'site_num', y = 'slope_datetime_cent', color = '#dd9404', marker=marker_, ax = ax, s =150, label= 'Expected DO Trend', legend=False)

                            
                            sns.scatterplot(data = plot_df, x= 'site_num', y = 'slope_datetime_cent', hue = 'site_type_label', palette=palette_, marker=marker, ax = ax, s =50, legend=False)
                        
                        
                    else:
            
                        sns.scatterplot(data = plot_df, x= 'site_num', y = 'slope_datetime_cent', hue ='site_type', palette = palette, marker=marker, ax = ax, s =50, legend=False)
                    
                    for idx in plot_df.index:
                        
                        if plot_df.loc[idx,'site_type'] == 'Main Basin':
                            
                            ax.plot([plot_df.loc[idx,'site_num'], plot_df.loc[idx,'site_num']],[plot_df.loc[idx,'slope_datetime_cent_95lo'], plot_df.loc[idx,'slope_datetime_cent_95hi']], color=palette['Main Basin'], alpha =0.7, zorder = -5, linewidth=1)
        
                        else:
                            
                            ax.plot([plot_df.loc[idx,'site_num'], plot_df.loc[idx,'site_num']],[plot_df.loc[idx,'slope_datetime_cent_95lo'], plot_df.loc[idx,'slope_datetime_cent_95hi']], color=palette['Sub-Basins'], alpha =0.7, zorder = -4, linewidth=1)
                    
                    
                    label = label_var #label_depth + ' ' + label_var
                                    
                    ax.text(0.05,0.05, label, transform=ax.transAxes, verticalalignment='bottom', fontweight = 'bold', color='k')
                        
                    # if (var == depth + '_CT') & (season == 'loDO'):
                        
                    #     ax.text(0.05,0.95, label_depth, transform=ax.transAxes, verticalalignment='top', fontweight = 'bold', color='k', style='italic')
                        
                        
                    ax.set_xticks([1,2,3,4,5],['PJ', 'NS', 'SP', 'CI', 'LC'])
                    
                    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
                                            
                    ax.axhline(0, color='gray', linestyle = '--', zorder = -5)
                    
                    ax.set_ylabel(unit, wrap=True)
                    
                    ax.set_xlabel('')
                    
                    ax.set_ylim(ymin, ymax)
                    
                    if (var == depth + '_DO_mg_L') & (depth == 'deep') & (season == 'loDO'):
                        
                        ax.legend(loc = 'upper left')
                        
                        
                    
                    
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/paper_1_fig_s' +str(c)+'.png', dpi=500,transparent=True, bbox_inches='tight')
    
c+=1                
                        
                        
                                
                                
                        
        
        
        
        
        
        
        
    
    # mosaic = [['surf_CT', 'surf_SA', 'surf_DO_mg_L'],['deep_CT', 'deep_SA', 'deep_DO_mg_L']]

    # fig, axd = plt.subplot_mosaic(mosaic, figsize=(8.5,5), layout='constrained', sharex=True, gridspec_kw=dict(wspace=0.1))

    
    # if season == 'loDO':
        
    #     label_season = 'Aug-Nov'
    
    # elif season == 'winter':
        
    #     label_season = 'Dec-Mar'
        
    # elif season == 'grow':
        
    #     label_season = 'Apr-Jul'

    # for var in ['surf_CT', 'surf_SA', 'surf_DO_mg_L', 'deep_DO_mg_L', 'deep_CT', 'deep_SA']:
        
    #     for stat in ['mk_ts']:
                    
    #         for deep_DO_q in ['deep_DO_q50']:
                
    #             if 'surf' in var:
                    
    #                 label_depth = 'Surface'
                    
    #                 #color = '#E91E63'
                    
    #             else:
                    
    #                 label_depth = 'Deep'
                    
    #                 c#olor = '#673AB7'
    
                
    #             if 'DO' in var:
                    
    #                 label_var = 'c ' + label_season + ' ' + label_depth + ' [DO]'
                    
    #                 ymin = -2
                    
    #                 ymax = 2
                    
    #                 marker = 'o'
                    
    #                 unit = r'[mg/L]/century'
                
    #             elif 'CT' in var:
                    
    #                 label_var = 'a '  + label_season + ' ' + label_depth + ' Temperature'
                    
    #                 ymin = -2.5
                    
    #                 ymax = 2.5
                    
    #                 marker = 'D'
                    
    #                 unit = r'[$^{\circ}$C]/century'
                
    #             else:
                    
    #                 label_var = 'b '  + label_season + ' ' +  label_depth + ' Salinity'
                    
    #                 ymin = -1
                    
    #                 ymax = 1
                    
    #                 marker = 's'
                    
    #                 unit = r'[PSU]/century'
    
                    
                    
                
    #             ax = axd[var]
    
    #             plot_df = all_stats_filt[(all_stats_filt['stat'] == stat) & (all_stats_filt['var'] == var) & (all_stats_filt['site'].isin(long_site_list)) & (all_stats_filt['deep_DO_q'] == deep_DO_q) & (all_stats_filt['season'] == season)]
                
    #             plot_df = plot_df.sort_values(by=['site']).reset_index()
                
    #             plot_df['slope_datetime_cent'] = plot_df['slope_datetime']*100
                
    #             plot_df['slope_datetime_cent_95hi'] = plot_df['slope_datetime_s_hi']*100
                
    #             plot_df['slope_datetime_cent_95lo'] = plot_df['slope_datetime_s_lo']*100
                
                
    #             palette = {'Main Basin':'#e04256', 'Sub-Basins':'#4565e8'}
                
    #             palette_ = {'Main Basin Trend':'#e04256', 'Sub-Basins Trend':'#4565e8'}
    
    
                
    #             sns.scatterplot(data = plot_df, x= 'site_num', y = 'slope_datetime_cent_95hi', hue ='site_type', palette = palette, marker=marker, ax = ax, s= 10, legend=False)
        
    #             sns.scatterplot(data = plot_df, x= 'site_num', y = 'slope_datetime_cent_95lo', hue ='site_type', palette = palette, marker=marker, ax = ax, s= 10, legend=False)
                
    #             if var == 'deep_DO_mg_L':
                    
    #                 plot_df_ = all_stats_filt[(all_stats_filt['stat'] == stat) & (all_stats_filt['season'] == season) & (all_stats_filt['var'] == 'deep_DO_sol') & (all_stats_filt['site'].isin(long_site_list)) & (all_stats_filt['deep_DO_q'] == deep_DO_q)]
                    
    #                 plot_df_ = plot_df_.sort_values(by=['site'])
                    
    #                 plot_df_['slope_datetime_cent'] = plot_df_['slope_datetime']*100
                    
    #                 plot_df_['slope_datetime_cent_95hi'] = plot_df_['slope_datetime_s_hi']*100
                    
    #                 plot_df_['slope_datetime_cent_95lo'] = plot_df_['slope_datetime_s_lo']*100
                    
    #                 plot_df.loc[plot_df['site_type'] == 'Main Basin', 'site_type_label'] = 'Main Basin Trend'
                    
    #                 plot_df.loc[plot_df['site_type'] == 'Sub-Basins', 'site_type_label'] = 'Sub-Basins Trend'
    
                    
                    
    #                 marker_ = "$\circ$"
            
                    
    #                 # sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_sol'], x= 'site_num', y = 'slope_datetime_cent_95hi', color = '#dd9404', marker=marker, ax = ax, s= 20, legend=False)
            
    #                 # sns.scatterplot(data = plot_df[plot_df['var'] == 'deep_DO_sol'], x= 'site_num', y = 'slope_datetime_cent_95lo', color = '#dd9404', marker=marker, ax = ax, s= 20, legend=False)
            
    #                 sns.scatterplot(data = plot_df_, x= 'site_num', y = 'slope_datetime_cent', color = '#dd9404', marker=marker_, ax = ax, s =150, label= 'Expected DO Trend')
                    
                    
                    
    #                 sns.scatterplot(data = plot_df, x= 'site_num', y = 'slope_datetime_cent', hue = 'site_type_label', palette=palette_, marker=marker, ax = ax, s =50)
    
                    
    #             else:
        
    #                 sns.scatterplot(data = plot_df, x= 'site_num', y = 'slope_datetime_cent', hue ='site_type', palette = palette, marker=marker, ax = ax, s =50, legend=False)
                
    #             for idx in plot_df.index:
                    
    #                 if plot_df.loc[idx,'site_type'] == 'Main Basin':
                        
    #                     ax.plot([plot_df.loc[idx,'site_num'], plot_df.loc[idx,'site_num']],[plot_df.loc[idx,'slope_datetime_cent_95lo'], plot_df.loc[idx,'slope_datetime_cent_95hi']], color=palette['Main Basin'], alpha =0.7, zorder = -5, linewidth=1)
    
    #                 else:
                        
    #                     ax.plot([plot_df.loc[idx,'site_num'], plot_df.loc[idx,'site_num']],[plot_df.loc[idx,'slope_datetime_cent_95lo'], plot_df.loc[idx,'slope_datetime_cent_95hi']], color=palette['Sub-Basins'], alpha =0.7, zorder = -4, linewidth=1)
                
                
    #             label = label_var #label_depth + ' ' + label_var
                                
    #             ax.text(0.05,0.05, label, transform=ax.transAxes, verticalalignment='bottom', fontweight = 'bold', color='k')
                    
                    
                
    #             # ymin = -max(abs(plot_df['slope_datetime_cent']))*2.5
                 
    #             # ymax = max(abs(plot_df['slope_datetime_cent']))*2.5
                
    #             #ax.set_xticks(sorted(all_stats_filt['site_num'].unique().tolist()),site_labels, rotation=90) 
                
    #             ax.set_xticks([1,2,3,4,5],['PJ', 'NS', 'SP', 'CI', 'LC'])
                
    #             ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
                                        
    #             ax.axhline(0, color='gray', linestyle = '--', zorder = -5)
                
    #             ax.set_ylabel(unit, wrap=True)
                
    #             ax.set_xlabel('')
                
    #             ax.set_ylim(ymin, ymax)
                
    #             if var == 'deep_DO_mg_L':
                    
    #                 ax.legend(loc = 'upper left')
                
                
    # plt.savefig('/Users/dakotamascarenas/Desktop/pltz/supplement_fig_' + season +'.png', dpi=500,transparent=True, bbox_inches='tight')

# %%
 