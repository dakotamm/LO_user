#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:16:40 2024

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


# %%

Ldir = Lfun.Lstart(gridname='cas7')

# %%

# poly_list = ['ps', 'admiralty_sill', 'budd_inlet', 'carr_inlet_mid', 'dana_passage', 'hat_island', 'hazel_point', 'hood_canal_mouth', 'lynch_cove_mid', 'near_seattle_offshore', 'near_edmonds', 'port_susan_mid', 'saratoga_passage_north', 'saratoga_passage_mid']

# odf_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))

# # %%

# basin_list = list(odf_dict.keys())

# var_list = ['SA', 'CT', 'DO_mg_L'] #'NO3 (uM)', 'Chl (mg m-3)', 'NO2 (uM)', 'SiO4 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO_mg_L', 'DO (uM)']

# # %%

# for key in odf_dict.keys():
    
#     odf_dict[key] = (odf_dict[key]
#                      .assign(
#                          datetime=(lambda x: pd.to_datetime(x['time'])),
#                          year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
#                          month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
#                          season=(lambda x: pd.cut(x['month'],
#                                                   bins=[0,3,6,9,12],
#                                                   labels=['winter', 'spring', 'summer', 'fall'])),
#                          DO_mg_L=(lambda x: x['DO (uM)']*32/1000),
#                          date_ordinal=(lambda x: x['datetime'].apply(lambda x: x.toordinal())),
#                          segment=(lambda x: key),
#                          decade=(lambda x: pd.cut(x['year'],
#                                                   bins=[1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2019, 2029],
#                                                   labels=['1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'], right=True))
#                              )
#                      )
    
#     for var in var_list:
        
#         if var not in odf_dict[key].columns:
            
#             odf_dict[key][var] = np.nan
            
#     odf_dict[key] = pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'source', 'otype', 'decade'],
#                                          value_vars=var_list, var_name='var', value_name = 'val')
    
# # %%

# odf = pd.concat(odf_dict.values(), ignore_index=True)

# # %%

# odf['source_type'] = odf['source'] + '_' + odf['otype']

# # %%

# lc_exclude = odf[(odf['segment'] == 'lynch_cove_mid') & (odf['z'] < -45)]

# odf = odf[~odf['cid'].isin(lc_exclude['cid'].unique())]

# # %%

# temp = odf[(odf['segment'] == 'near_seattle_offshore')].groupby('cid').agg({'z':'min'}).reset_index()

# sea_offshore_exclude = temp[temp['z'] >-70]

# odf = odf[~odf['cid'].isin(sea_offshore_exclude['cid'].unique())]

# # %%

# odf = odf.dropna()

# # %%

# min_depths = odf.groupby(['segment','season','cid']).agg({'z':'min'}).reset_index().dropna()

# min_depths_avg_sd = min_depths.groupby(['segment', 'season']).agg({'z':['mean', 'std']})

# min_depths_avg_sd.columns = min_depths_avg_sd.columns.to_flat_index().map('_'.join)

# min_depths_avg_sd = min_depths_avg_sd.reset_index()


# # %%

# odf_new = pd.merge(odf, min_depths_avg_sd, how='left', on=['segment', 'season'])

# # %%

# odf_filt_bottom = odf_new[(odf_new['z'] < odf_new['z_mean'] + odf_new['z_std']) & (odf_new['z'] > odf_new['z_mean'] - odf_new['z_std'])]

# # %%

# odf_bottom = odf_filt_bottom.groupby(['season','segment','cid','var']).agg({'val':['mean','std'], 'datetime':'mean', 'date_ordinal':'mean'})

# odf_bottom.columns = odf_bottom.columns.to_flat_index().map('_'.join)

# odf_bottom = odf_bottom.reset_index().dropna()

# # %%

# # for basin in basin_list:
    
# #     for season in ['winter','spring','summer', 'fall']:
        
# #         for var in var_list:
            
# #             fig, ax = plt.subplots(figsize=(24,8))
                    
# #             plt.rc('font', size=14)
            
# #             plot_df = odf_bottom[(odf_bottom['season_'] == season) & (odf_bottom['var_'] == var) & (odf_bottom['segment_'] == basin)]
            
# #             sns.scatterplot(data=plot_df, x='datetime_mean', y ='val_mean', ax=ax)
                        
# #             ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
# #             if var == 'DO_mg_L':
                
# #                 ax.axhspan(0,2, color = 'lightgray', alpha = 0.2)
                
# #             ax.set_title(basin + ' bottom ' + var + ' ' + season)
                        
# #             fig.tight_layout()
            
# #             plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + var+ '_' + season +'_bottom.png', bbox_inches='tight', dpi=500)
            
# # %%

# # for basin in basin_list:
    
# #     for season in ['winter','spring','summer', 'fall']:
        
# #         for var in var_list:
            
# #             fig, ax = plt.subplots(figsize=(24,8))
                    
# #             plt.rc('font', size=14)
            
# #             plot_df = odf_bottom[(odf_bottom['season'] == season) & (odf_bottom['var'] == var) & (odf_bottom['segment'] == basin)]
            
# #             plot_df_mean = plot_df.set_index('datetime_mean').sort_index()
            
# #             rolling_mean = plot_df_mean['val_mean'].rolling(window='3650D', min_periods=1).mean()
            
# #             sns.scatterplot(data=plot_df, x='datetime_mean', y ='val_mean', ax=ax)
            
# #             rolling_mean.plot(label='10-Year Rolling Mean')
                        
# #             ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
# #             if var == 'DO_mg_L':
                
# #                 ax.axhspan(0,2, color = 'lightgray', alpha = 0.2)
                
# #             ax.set_title(basin + ' bottom ' + var + ' ' + season)
            
# #             fig.tight_layout()
            
# #             plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + var+ '_' + season +'_bottom_decadal_mean.png', bbox_inches='tight', dpi=500)
            

# # %%

# for basin in basin_list:
    
#     for season in ['winter','spring','summer', 'fall']:
        
                    
#         fig, ax = plt.subplots(figsize=(8,16), nrows=3, sharex=True)
                
#         plt.rc('font', size=14)
        
#         c=0
        
        
#         for var in var_list:
            
#             if var =='SA':
                
#                 marker = 's'
                
#                 ymin = 22
                
#                 ymax = 32
                
#                 label = 'Salinity [PSU]'
                
#                 color = 'blue'
                
#                 color_less = '#393070'
                
#                 color_more = '#ABCF43'
                        
#             elif var == 'CT':
                
#                 marker = '^'
                
#                 ymin = 7
                
#                 ymax = 17
                
#                 label = 'Temperature [deg C]'
                
#                 color = 'red'
                
#                 color_less = '#466EA9'
                
#                 color_more = '#9C3C49'
                
#             else:
                
#                 marker = 'o'
                
#                 ymin = 0
                
#                 ymax = 12
                
#                 color = 'black'
                
#                 label = 'DO [mg/L]'
                
#                 color_more = '#3E1F95'
                
#                 color_less = '#EAB63A'
    
#             plot_df = odf_bottom[(odf_bottom['season'] == season) & (odf_bottom['var'] == var) & (odf_bottom['segment'] == basin)]
            
#             plot_df_mean = plot_df.set_index('datetime_mean').sort_index()
            
#             rolling_mean = plot_df_mean['val_mean'].rolling(window='3650D', min_periods=1).mean()
            
#             sns.scatterplot(data=plot_df, x='datetime_mean', y ='val_mean', ax=ax[c], color = color)
            
#             rolling_mean.plot(label='10-Year Rolling Mean', ax=ax[c], color = color)
                    
#             ax[c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
#             ax[c].set_ylabel(label)
            
#             ax[c].set_ylim(ymin,ymax)
        
#             if var == 'DO_mg_L':
                
#                 ax[c].axhspan(0,2, color = 'lightgray', alpha = 0.2)
                
#             c+=1
            
#         ax[0].set_title(basin + ' bottom ' + season)
        
#         ax[-1].set_xlabel('Date')
        
#         fig.tight_layout()
        
#         plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + season +'_bottom_decadal_mean.png', bbox_inches='tight', dpi=500)






# %%




            
# %%

poly_list = ['ps', 'mb', 'wb', 'ss', 'hc']

odf_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish'], otype_list=['bottle', 'ctd'], year_list=np.arange(1999,2025))

# %%

basin_list = list(odf_dict.keys())

var_list = ['SA', 'CT', 'DO_mg_L'] #'NO3 (uM)', 'Chl (mg m-3)', 'NO2 (uM)', 'SiO4 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO_mg_L', 'DO (uM)']

# %%

for key in odf_dict.keys():
    
    odf_dict[key] = (odf_dict[key]
                      .assign(
                          datetime=(lambda x: pd.to_datetime(x['time'])),
                          year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                          month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                          season=(lambda x: pd.cut(x['month'],
                                                  bins=[0,3,6,9,12],
                                                  labels=['winter', 'spring', 'summer', 'fall'])),
                          DO_mg_L=(lambda x: x['DO (uM)']*32/1000),
                          date_ordinal=(lambda x: x['datetime'].apply(lambda x: x.toordinal())),
                          segment=(lambda x: key),
                          decade=(lambda x: pd.cut(x['year'],
                                                  bins=[1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2019, 2029],
                                                  labels=['1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'], right=True))
                              )
                      )
    
    for var in var_list:
        
        if var not in odf_dict[key].columns:
            
            odf_dict[key][var] = np.nan
            
    odf_dict[key] = pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'source', 'otype', 'decade'],
                                          value_vars=var_list, var_name='var', value_name = 'val')
    
# %%

odf = pd.concat(odf_dict.values(), ignore_index=True)

# %%

odf['source_type'] = odf['source'] + '_' + odf['otype']

# %%

lc_exclude = odf[(odf['segment'] == 'lynch_cove_mid') & (odf['z'] < -45)]

odf = odf[~odf['cid'].isin(lc_exclude['cid'].unique())]

# %%

temp = odf[(odf['segment'] == 'near_seattle_offshore')].groupby('cid').agg({'z':'min'}).reset_index()

sea_offshore_exclude = temp[temp['z'] >-70]

odf = odf[~odf['cid'].isin(sea_offshore_exclude['cid'].unique())]

# %%

odf = odf.dropna()

# %%

min_depths = odf.groupby(['segment','season','cid']).agg({'z':'min'}).reset_index().dropna()

min_depths_avg_sd = min_depths.groupby(['segment', 'season']).agg({'z':['mean', 'std']})

min_depths_avg_sd.columns = min_depths_avg_sd.columns.to_flat_index().map('_'.join)

min_depths_avg_sd = min_depths_avg_sd.reset_index()


# %%

odf_new = pd.merge(odf, min_depths_avg_sd, how='left', on=['segment', 'season'])

# %%

odf_filt_bottom = odf_new[(odf_new['z'] < odf_new['z_mean'] + odf_new['z_std']) & (odf_new['z'] > odf_new['z_mean'] - odf_new['z_std'])]

# %%

odf_bottom = odf_filt_bottom.groupby(['season','segment','cid','var']).agg({'val':['mean','std'], 'datetime':'mean', 'date_ordinal':'mean'})

odf_bottom.columns = odf_bottom.columns.to_flat_index().map('_'.join)

odf_bottom = odf_bottom.reset_index().dropna()

            
# %%

odf_bottom_modern = odf_bottom[odf_bottom['datetime_mean'].dt.year < 2020]

for basin in basin_list:
    
    for season in ['winter','spring','summer', 'fall']:
        
        fig, ax = plt.subplots(figsize=(8,16), nrows=3, sharex=True)
                
        plt.rc('font', size=14)
        
        c=0
        
        for var in var_list:
            
            if var =='SA':
                
                marker = 's'
                
                ymin = 22
                
                ymax = 32
                
                label = 'Salinity [PSU]'
                
                color = 'blue'
                
                color_less = '#393070'
                
                color_more = '#ABCF43'
                        
            elif var == 'CT':
                
                marker = '^'
                
                ymin = 7
                
                ymax = 17
                
                label = 'Temperature [deg C]'
                
                color = 'red'
                
                color_less = '#466EA9'
                
                color_more = '#9C3C49'
                
            else:
                
                marker = 'o'
                
                ymin = 0
                
                ymax = 12
                
                color = 'black'
                
                label = 'DO [mg/L]'
                
                color_more = '#3E1F95'
                
                color_less = '#EAB63A'
    
            plot_df = odf_bottom_modern[(odf_bottom_modern['season'] == season) & (odf_bottom_modern['var'] == var) & (odf_bottom_modern['segment'] == basin)]
            
            plot_df_mean = plot_df.set_index('datetime_mean').sort_index()
            
            rolling_mean = plot_df_mean['val_mean'].rolling(window='365D', min_periods=1).mean()
            
            sns.scatterplot(data=plot_df, x='datetime_mean', y ='val_mean', ax=ax[c], color=color)
            
            rolling_mean.plot(label='Annual Rolling Mean', ax=ax[c], color = color)
                        
            ax[c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            ax[c].set_ylabel(label)
            
            ax[c].set_ylim(ymin,ymax)
        
            if var == 'DO_mg_L':
                
                ax[c].axhspan(0,2, color = 'lightgray', alpha = 0.2)
                
            c+=1
            
        ax[0].set_title(basin + ' bottom ' + season)
        
        ax[-1].set_xlabel('Date')
                        
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + season +'_bottom_annual_mean.png', bbox_inches='tight', dpi=500)
