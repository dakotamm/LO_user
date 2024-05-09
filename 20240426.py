#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:05:02 2024

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

poly_list = ['ps', 'admiralty_sill', 'budd_inlet', 'carr_inlet_mid', 'dana_passage', 'hat_island', 'hazel_point', 'hood_canal_mouth', 'lynch_cove_mid', 'near_seattle_offshore', 'near_edmonds', 'port_susan_mid', 'saratoga_passage_north', 'saratoga_passage_mid']

odf_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))

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

# for decade in odf['decade'].unique():
    
#     for basin in basin_list:

#         fig, ax = plt.subplots(figsize=(24,8), ncols=3)
        
#         plt.rc('font', size=14)
        
#         plot_df = odf[(odf['season'] == 'fall') & (odf['segment'] == basin) & (odf['decade'] == decade) & (odf['var'] == 'DO_mg_L') & (odf['val'] > 0) & (odf['val'] < 50)].groupby('cid').min().reset_index()
        
#         ax[0].hist2d(plot_df['val'], plot_df['z'], bins=100, cmap='inferno', cmin=1)
        
#         #plt.colorbar()
        
#         ax[0].set_xlabel('DO cast minima [mg/L]')
        
#         ax[0].set_ylabel('z cast minima [m]')
        
#         ax[0].axvspan(0,2, color = 'lightgray', alpha = 0.2) 
        
#         ax[0].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
#         ax[0].set_title(basin + ' ' + decade)


        
#         plot_df = odf[(odf['season'] == 'fall') & (odf['segment'] == basin) & (odf['decade'] == decade) & (odf['var'] == 'DO_mg_L') & (odf['var'] == 'DO_mg_L') & (odf['val'] > 0) & (odf['val'] < 50)].groupby('cid').agg({
#             'z': 'min',  # Find minimum value of 'Value1'
#             'val': lambda x: x.loc[x.idxmin()]  # Find value of 'Value2' at the minimum index of 'Value1'
#         }).reset_index()
        
#         ax[1].hist2d(plot_df['val'], plot_df['z'], bins=100, cmap='inferno', cmin=1)
        
#         #plt.colorbar()
        
#         ax[1].set_xlabel('DO at minimum depth [mg/L]')
        
#         ax[1].set_ylabel('z cast minima [m]')
        
#         ax[1].axvspan(0,2, color = 'lightgray', alpha = 0.2)
        
#         ax[1].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
#         #ax.set_title(basin + ' ' + decade)

        
#         plot_df = odf[(odf['season'] == 'fall') & (odf['segment'] == basin) & (odf['decade'] == decade) & (odf['var'] == 'DO_mg_L') & (odf['var'] == 'DO_mg_L') & (odf['val'] > 0) & (odf['val'] < 50)].groupby('cid').agg({
#             'val': 'min',  # Find minimum value of 'Value1'
#             'z': lambda x: x.loc[x.idxmin()]  # Find value of 'Value2' at the minimum index of 'Value1'
#         }).reset_index()
        
#         ax[2].hist2d(plot_df['val'], plot_df['z'], bins=100, cmap='inferno', cmin=1)
        
#         #plt.colorbar()
        
#         ax[2].set_xlabel('DO cast minima [mg/L]')
        
#         ax[2].set_ylabel('z at minimum DO [m]')
        
#         ax[2].axvspan(0,2, color = 'lightgray', alpha = 0.2)
        
#         ax[2].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
#         #ax.set_title(basin + ' ' + decade)

        
        
#         plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + decade +'s_hist_fall_DO_mins_vs_z_mins_all.png', bbox_inches='tight', dpi=500)

# # %%

# for decade in odf['decade'].unique():
    
#     for basin in basin_list:

#         fig, ax = plt.subplots(figsize=(24,8), ncols=3)
        
#         plt.rc('font', size=14)
        
#         plot_df = odf[(odf['season'] == 'fall') & (odf['segment'] == basin) & (odf['decade'] == decade) & (odf['var'] == 'SA')].groupby('cid').agg({'z':'min', 'val':'max'}).reset_index()
        
#         ax[0].hist2d(plot_df['val'], plot_df['z'], bins=100, cmap='inferno', cmin=1)
        
#         #plt.colorbar()
        
#         ax[0].set_xlabel('SA cast maxima [psu]')
        
#         ax[0].set_ylabel('z cast minima [m]')
        
#         ax[0].axvspan(0,2, color = 'lightgray', alpha = 0.2) 
        
#         ax[0].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
#         ax[0].set_title(basin + ' ' + decade)


        
#         plot_df = odf[(odf['season'] == 'fall') & (odf['segment'] == basin) & (odf['decade'] == decade) & (odf['var'] == 'SA')].groupby('cid').agg({
#             'z': 'min',  # Find minimum value of 'Value1'
#             'val': lambda x: x.loc[x.idxmin()]  # Find value of 'Value2' at the minimum index of 'Value1'
#         }).reset_index()
        
#         ax[1].hist2d(plot_df['val'], plot_df['z'], bins=100, cmap='inferno', cmin=1)
        
#         #plt.colorbar()
        
#         ax[1].set_xlabel('SA at minimum depth [psu]')
        
#         ax[1].set_ylabel('z cast minima [m]')
        
#         ax[1].axvspan(0,2, color = 'lightgray', alpha = 0.2)
        
#         ax[1].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
#         #ax.set_title(basin + ' ' + decade)

        
#         plot_df = odf[(odf['season'] == 'fall') & (odf['segment'] == basin) & (odf['decade'] == decade) & (odf['var'] == 'SA')].groupby('cid').agg({
#             'val': 'max',  # Find maximum value of 'Value1'
#             'z': lambda x: x.loc[x.idxmax()]  # Find value of 'Value2' at the maximum index of 'Value1'
#         }).reset_index()
        
#         ax[2].hist2d(plot_df['val'], plot_df['z'], bins=100, cmap='inferno', cmin=1)
        
#         #plt.colorbar()
        
#         ax[2].set_xlabel('SA cast maxima [mg/L]')
        
#         ax[2].set_ylabel('z at maximum SA [m]')
        
#         ax[2].axvspan(0,2, color = 'lightgray', alpha = 0.2)
        
#         ax[2].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
#         #ax.set_title(basin + ' ' + decade)

        
        
#         plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + decade +'s_hist_fall_SA_maxs_vs_z_mins_all.png', bbox_inches='tight', dpi=500)
        
# # %%

# for decade in odf['decade'].unique():
    
#     for basin in basin_list:

#         fig, ax = plt.subplots(figsize=(24,8), ncols=3)
        
#         plt.rc('font', size=14)
        
#         plot_df = odf[(odf['season'] == 'fall') & (odf['segment'] == basin) & (odf['decade'] == decade) & (odf['var'] == 'CT')].groupby('cid').agg({'z':'min', 'val':'min'}).reset_index()
        
#         ax[0].hist2d(plot_df['val'], plot_df['z'], bins=100, cmap='inferno', cmin=1)
        
#         #plt.colorbar()
        
#         ax[0].set_xlabel('CT cast minima [deg C]')
        
#         ax[0].set_ylabel('z cast minima [m]')
        
#         ax[0].axvspan(0,2, color = 'lightgray', alpha = 0.2) 
        
#         ax[0].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
#         ax[0].set_title(basin + ' ' + decade)


        
#         plot_df = odf[(odf['season'] == 'fall') & (odf['segment'] == basin) & (odf['decade'] == decade) & (odf['var'] == 'CT')].groupby('cid').agg({
#             'z': 'min',  # Find minimum value of 'Value1'
#             'val': lambda x: x.loc[x.idxmin()]  # Find value of 'Value2' at the minimum index of 'Value1'
#         }).reset_index()
        
#         ax[1].hist2d(plot_df['val'], plot_df['z'], bins=100, cmap='inferno', cmin=1)
        
#         #plt.colorbar()
        
#         ax[1].set_xlabel('CT at minimum depth [deg C]')
        
#         ax[1].set_ylabel('z cast minima [m]')
        
#         ax[1].axvspan(0,2, color = 'lightgray', alpha = 0.2)
        
#         ax[1].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
#         #ax.set_title(basin + ' ' + decade)

        
#         plot_df = odf[(odf['season'] == 'fall') & (odf['segment'] == basin) & (odf['decade'] == decade) & (odf['var'] == 'CT')].groupby('cid').agg({
#             'val': 'min',  # Find maximum value of 'Value1'
#             'z': lambda x: x.loc[x.idxmin()]  # Find value of 'Value2' at the maximum index of 'Value1'
#         }).reset_index()
        
#         ax[2].hist2d(plot_df['val'], plot_df['z'], bins=100, cmap='inferno', cmin=1)
        
#         #plt.colorbar()
        
#         ax[2].set_xlabel('CT cast minima [deg C]')
        
#         ax[2].set_ylabel('z at minimum CT [m]')
        
#         ax[2].axvspan(0,2, color = 'lightgray', alpha = 0.2)
        
#         ax[2].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
#         #ax.set_title(basin + ' ' + decade)

        
        
#         plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + decade +'s_hist_fall_CT_mins_vs_z_mins_all.png', bbox_inches='tight', dpi=500)
        
        
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

odf_bottom = odf_filt_bottom.groupby(['season','segment','cid','var']).agg({'val':['mean','std'], 'datetime':'mean', 'date_ordinal':'mean'}).reset_index()

odf_bottom.columns = odf_bottom.columns.to_flat_index().map('_'.join)

odf_bottom = odf_bottom.reset_index()

# %%

for basin in basin_list:
    
    for season in ['winter','spring','summer', 'fall']:
        
        for var in var_list:
            
            fig, ax = plt.subplots(figsize=(24,8))
                    
            plt.rc('font', size=14)
            
            plot_df = odf_bottom[(odf_bottom['season_'] == season) & (odf_bottom['var_'] == var) & (odf_bottom['segment_'] == basin)]
            
            sns.scatterplot(data=plot_df, x='datetime_mean', y ='val_mean', ax=ax)
            
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            if var == 'DO_mg_L':
                
                ax.axhspan(0,2, color = 'lightgray', alpha = 0.2)
                
            ax.set_title(basin + ' bottom ' + var + ' ' + season)
            
            fig.tight_layout()
            
            plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + var+ '_' + season +'_bottom.png', bbox_inches='tight', dpi=500)
                


