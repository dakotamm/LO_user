#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:14:35 2024

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

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle'], year_list=np.arange(1930,2025))


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

# %%

# c=0

# all_stats_filt.loc[all_stats_filt['site'] == 'point_jefferson', 'site_label'] = 'PJ'

# all_stats_filt.loc[all_stats_filt['site'] == 'near_seattle_offshore', 'site_label'] = 'NS'

# all_stats_filt.loc[all_stats_filt['site'] == 'saratoga_passage_mid', 'site_label'] = 'SP'

# all_stats_filt.loc[all_stats_filt['site'] == 'carr_inlet_mid', 'site_label'] = 'CI'

# all_stats_filt.loc[all_stats_filt['site'] == 'lynch_cove_mid', 'site_label'] = 'LC'


# all_stats_filt.loc[all_stats_filt['site'] == 'point_jefferson', 'site_type'] = 'Main Basin'

# all_stats_filt.loc[all_stats_filt['site'] == 'near_seattle_offshore', 'site_type'] = 'Main Basin'

# all_stats_filt.loc[all_stats_filt['site'] == 'saratoga_passage_mid', 'site_type'] = 'Sub-Basins'

# all_stats_filt.loc[all_stats_filt['site'] == 'carr_inlet_mid', 'site_type'] = 'Sub-Basins'

# all_stats_filt.loc[all_stats_filt['site'] == 'lynch_cove_mid', 'site_type'] = 'Sub-Basins'


# all_stats_filt.loc[all_stats_filt['site'] == 'point_jefferson', 'site_num'] = 1

# all_stats_filt.loc[all_stats_filt['site'] == 'near_seattle_offshore', 'site_num'] = 2

# all_stats_filt.loc[all_stats_filt['site'] == 'saratoga_passage_mid', 'site_num'] = 3

# all_stats_filt.loc[all_stats_filt['site'] == 'carr_inlet_mid', 'site_num'] = 4

# all_stats_filt.loc[all_stats_filt['site'] == 'lynch_cove_mid', 'site_num'] = 5







# mosaic = [['deep_CT', 'deep_SA', 'deep_DO_mg_L']]

# fig, axd = plt.subplot_mosaic(mosaic, figsize=(8.5,2.5), layout='constrained', sharex=True, gridspec_kw=dict(wspace=0.1))

        
    
    

# for var in ['deep_DO_mg_L', 'deep_CT', 'deep_SA']:
    
#     for stat in ['mk_ts']:
                
#         for deep_DO_q in ['deep_DO_q50']:
            
#             if 'surf' in var:
                
#                 label_depth = 'Surface'
                
#                 color = '#E91E63'
                
#             else:
                
#                 label_depth = 'Deep'
                
#                 color = '#673AB7'

            
#             if 'DO' in var:
                
#                 label_var = 'c  [DO]'
                
#                 ymin = -2
                
#                 ymax = 2
                
#                 marker = 'o'
                
#                 unit = r'[mg/L]/century'
            
#             elif 'CT' in var:
                
#                 label_var = 'a  Temperature'
                
#                 ymin = -2.5
                
#                 ymax = 2.5
                
#                 marker = 'D'
                
#                 unit = r'[$^{\circ}$C]/century'
            
#             else:
                
#                 label_var = 'b  Salinity'
                
#                 ymin = -1
                
#                 ymax = 1
                
#                 marker = 's'
                
#                 unit = r'[PSU]/century'

                
                
            
#             ax = axd[var]

#             plot_df = all_stats_filt[(all_stats_filt['stat'] == stat) & (all_stats_filt['var'] == var) & (all_stats_filt['site'].isin(long_site_list)) & (all_stats_filt['deep_DO_q'] == deep_DO_q) & (all_stats_filt['season'] == 'loDO')]
            
#             plot_df = plot_df.sort_values(by=['site']).reset_index()
            
#             plot_df['slope_datetime_cent'] = plot_df['slope_datetime']*100
            
#             plot_df['slope_datetime_cent_95hi'] = plot_df['slope_datetime_s_hi']*100
            
#             plot_df['slope_datetime_cent_95lo'] = plot_df['slope_datetime_s_lo']*100
            
            
#             palette = {'Main Basin':'#e04256', 'Sub-Basins':'#4565e8'}

            
#             sns.scatterplot(data = plot_df, x= 'site_num', y = 'slope_datetime_cent_95hi', hue ='site_type', palette = palette, marker=marker, ax = ax, s= 10, legend=False)
    
#             sns.scatterplot(data = plot_df, x= 'site_num', y = 'slope_datetime_cent_95lo', hue ='site_type', palette = palette, marker=marker, ax = ax, s= 10, legend=False)
            
#             if var == 'deep_DO_mg_L':
                
#                 sns.scatterplot(data = plot_df, x= 'site_num', y = 'slope_datetime_cent', hue ='site_type', palette = palette, marker=marker, ax = ax, s =50, legend=True)

                
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
            
            
# plt.savefig('/Users/dakotamascarenas/Desktop/pltz/test__.png', dpi=500,transparent=True, bbox_inches='tight')   

# %%

odf_use = (odf_use
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

odf_calc_use = (odf_calc_use
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

alpha = 0.05

for site in site_list:

    for season in ['winter', 'grow', 'loDO']:
               
        mosaic = [['surf_DO_mg_L', 'deep_DO_mg_L'], ['surf_CT', 'deep_CT'], ['surf_SA', 'deep_SA'], ['', 'strat_sigma']]
        
        fig, ax = plt.subplot_mosaic(mosaic, layout='constrained', figsize = (10,10), sharex=True)
        
        for var in ['DO_mg_L', 'CT', 'SA']:
                
            if var =='SA':
                        
                marker = 's'
                
                ymin = 25
                
                ymax = 35
                
                label = 'Salinity [PSU]'
                        
            elif var == 'CT':
                
                marker = 'D'
                
                ymin = 6
                
                ymax = 20
                
                label = 'Temperature [deg C]'
                
            else:
                
                marker = 'o'
                
                ymin = 0
                
                ymax = 18
                
                color = 'black'
                
                label = 'DO [mg/L]'
                             
                colors = {'deep':'#673AB7', 'surf':'#E91E63'}
                
                         
            for depth in ['surf', 'deep']:
                 
                 ax_name = depth + '_' + var
                 
                 var_name = ax_name
                 
                 plot_df = odf_use[(odf_use['site'] == site) & (odf_use['var'] == var) & (odf_use['season'] == season) & (odf_use['surf_deep'] == depth)]
                 
                 stat_df = all_stats_filt[(all_stats_filt['site'] == site) & (all_stats_filt['var'] == var_name) & (all_stats_filt['season'] == season)]
                 
                 if not stat_df.empty:

                         
                     sns.scatterplot(data=plot_df, x='datetime', y ='val', ax=ax[ax_name], alpha=0.7, legend = False, marker=marker, color = colors[depth]) 
                     
                        
                     x = plot_df['date_ordinal']
                     
                     x_plot = plot_df['datetime']
                     
                     y = plot_df['val']
                     
                 
                    
                     reject_null = stat_df['p'].unique()[0] < alpha
                     
                     B0 = stat_df['B0'].unique()[0]
                     
                     B1 = stat_df['B1'].unique()[0]
                     
                     slope_datetime = stat_df['slope_datetime'].unique()[0]
                     
                     slope_datetime_s_hi = stat_df['slope_datetime_s_hi'].unique()[0]
                     
                     slope_datetime_s_lo = stat_df['slope_datetime_s_lo'].unique()[0]
    
    
                     
    
                     
                     if reject_null == True:
                         
                         ax[ax_name].plot([x_plot.min(), x_plot.max()], [B0 + B1*x.min(), B0 + B1*x.max()], color=colors[depth], alpha =0.7, linestyle='-', linewidth=2)
                         
                         ax[ax_name].text(0.99,0.9, str(np.round(slope_datetime*100,2)) + '/cent. (' + str(np.round(slope_datetime_s_lo*100,2)) + ' - ' + str(np.round(slope_datetime_s_hi*100,2)) + ')', horizontalalignment='right', verticalalignment='top', transform=ax[ax_name].transAxes, color=colors[depth], bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
                     
                     else:
                         
                         ax[ax_name].text(0.99,0.9, str(np.round(slope_datetime*100,2)) + '/cent. (' + str(np.round(slope_datetime_s_lo*100,2)) + ' - ' + str(np.round(slope_datetime_s_hi*100,2)) + ')', horizontalalignment='right', verticalalignment='top', transform=ax[ax_name].transAxes, color='k', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
        
                         
                     ax[ax_name].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
                     
                     ax[ax_name].set_ylabel(str.capitalize(depth) + ' ' + label)
                     
                     ax[ax_name].set_ylim(ymin,ymax)
                     
                 
                     if var == 'DO_mg_L':
                         
                         ax[ax_name].axhspan(0,2, color = 'lightgray', alpha = 0.2)
                         
                     ax[ax_name].set_xlim([datetime.date(1930,1,1), datetime.date(2024,12,31)])
                     
                     ax[ax_name].set_xlabel('Year')
                 
                 
                 
            for var in ['strat_sigma']: #just deep for now!!!
             
                ax_name = var
                
                if var == 'DO_sol':
                
                    var_name = 'deep_' + ax_name
                    
                else:
                    
                    var_name = ax_name
             
                plot_df = odf_calc_use[(odf_calc_use['site'] == site) & (odf_calc_use['season'] == season) & (odf_calc_use['var'] == var)]
                
                stat_df = all_stats_filt[(all_stats_filt['site'] == site) & (all_stats_filt['var'] == var_name) & (all_stats_filt['season'] == season)]
                
                if not stat_df.empty:
                    
             
                    if 'DO_sol' in var:
                        
                        color = '#ff7f0e'
                        
                    else:
                        
                        color = '#1f77b4'
                            
                    sns.scatterplot(data=plot_df, x='datetime', y ='val', ax=ax[ax_name], color = color, alpha=0.7)
                    
                    x = plot_df['date_ordinal']
                    
                    x_plot = plot_df['datetime']
                    
                    y = plot_df['val']
                    
                
                   
                    reject_null = stat_df['p'].unique()[0] < alpha
                    
                    B0 = stat_df['B0'].unique()[0]
                    
                    B1 = stat_df['B1'].unique()[0]
                    
                    slope_datetime = stat_df['slope_datetime'].unique()[0]
                    
                    slope_datetime_s_hi = stat_df['slope_datetime_s_hi'].unique()[0]
                    
                    slope_datetime_s_lo = stat_df['slope_datetime_s_lo'].unique()[0]
                    
                    if reject_null:
                        
                        ax[ax_name].plot([x_plot.min(), x_plot.max()], [B0 + B1*x.min(), B0 + B1*x.max()], color=color, alpha =0.7, linestyle = '-', linewidth=2)
                    
                        ax[ax_name].text(0.99,0.9, str(np.round(slope_datetime*100,2)) + '/cent. (' + str(np.round(slope_datetime_s_lo*100,2)) + ' - ' + str(np.round(slope_datetime_s_hi*100,2)) + ')', horizontalalignment='right', verticalalignment='top', transform=ax[ax_name].transAxes, color=colors[depth], bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
                    
                    else:
                        
                        ax[ax_name].text(0.99,0.9, str(np.round(slope_datetime*100,2)) + '/cent. (' + str(np.round(slope_datetime_s_lo*100,2)) + ' - ' + str(np.round(slope_datetime_s_hi*100,2)) + ')', horizontalalignment='right', verticalalignment='top', transform=ax[ax_name].transAxes, color='k', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
       
                    
                    
                    ax[ax_name].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
                    
                    ax[ax_name].set_ylabel(label)
                    
                    ax[ax_name].set_ylim(ymin,ymax)
                        
                    ax[ax_name].set_xlim([datetime.date(1930,1,1), datetime.date(2024,12,31)])
                    
                    ax[ax_name].set_xlabel('Year')
                    
                    if 'DO_sol' in var:
                        
                        ax[ax_name].set_ylabel('DO Saturation [mg/L]')
                        
                        ax[ax_name].set_ylim(8,12)
                        
                    else:
                        
                        ax[ax_name].set_ylabel(r'Strat~Deep-Surf [$\sigma$]')
                        
                        ax[ax_name].set_ylim(0,10)
                     
        plt.suptitle(site + ' ' + season)
             
             
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_' + season + '_test_all-NOCTD.png', bbox_inches='tight', dpi=500, transparent=False)


# ended 7/19/2024 - MAKE CHART PLOT, like for each site and each configuration, where are we finding the significant slopes?