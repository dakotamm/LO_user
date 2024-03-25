#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:14:49 2024

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

poly_list = ['hc_wo_lc']

odf_dict = dfun.getPolyData(Ldir, poly_list, source_list=['ecology', 'nceiSalish', 'collias'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2020))

# %%

basin_list = list(odf_dict.keys())

var_list = ['SA', 'CT', 'NO3 (uM)', 'Chl (mg m-3)', 'NO2 (uM)', 'SiO4 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO_mg_L', 'DO (uM)']

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

odf = odf[odf['var'].isin(['DO_mg_L', 'CT', 'SA'])]

odf = odf[(odf['val'] >= 0) & (odf['val'] <50)]

odf[(odf['source'] == 'ecology') & (odf['var'] == 'DO_mg_L') & (odf['otype'] == 'bottle')] = np.nan

# %%

odf = (odf
            .assign(
               # datetime=(lambda x: pd.to_datetime(x['time'])),
                 depth_range=(lambda x: pd.cut(x['z'], 
                                               bins=[-700, -355, -275, -205, -165, -135, -105, -80, -55, -37.5, -27.5, -17.5, -7.5, 0],
                                               labels= ['>355m', '275m-355m', '205m-275m', '165-205m','135m-165m','105m-135m', '80m-105m', '65m-80m','55m-80m','27.5m-37.5m', '17.5m-27.5m', '7.5m-17.5m', '<7.5m'])),
                 lat_range=(lambda x: pd.cut(x['lat'],
                                          bins=[47, 47.4, 47.46, 47.51, 47.58, 47.65, 47.68, 47.7, 47.75, 47.82, 47.88, 48],
                                          labels=['47-47.4', '47.4-47.46', '47.46-47.51', '47.51-47.58', '47.58-47.65', '47.65-47.68', '47.68-47.7', '47.7-47.75', '47.75-47.82', '47.82-47.88', '47.88-48']))
              
                 
                 )
            )# make less manual


# %%

for basin in basin_list:

    for decade in ['1930', '2010']:
        
        fig, ax = plt.subplots(figsize=(6,6))
        
        plt.rc('font', size=14)
        
        plot_df = odf[(odf['segment'] == basin) & (odf['var'] == 'DO_mg_L')].groupby('cid').first().reset_index()

        sns.scatterplot(data=plot_df, x='lon', y='lat', color = 'white', alpha=0, legend=False)
        
        ax.autoscale(enable=False)
    
        pfun.add_coast(ax)
    
        pfun.dar(ax)
             
        plot_df = odf[(odf['segment'] == basin) & (odf['var'] == 'DO_mg_L') & (odf['decade'] == decade)].groupby('cid').first().reset_index()
        
        if decade == '1930':
    
            sns.scatterplot(data=plot_df, x='lon', y='lat', color = '#8ad6cc', legend=False)
            
        else:
            
            sns.scatterplot(data=plot_df, x='lon', y='lat', color = '#f97171', legend=False)
        
        #ax.set_title('Hood Canal ' + decade + 's DO sampling locations')

        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + decade + '_fall_DO_sampling_locations.png', bbox_inches='tight', dpi=500, transparent=True)
   
        
# %%

lat_counts = (odf
                     .dropna()
                     #.set_index('datetime')
                     .groupby(['lat_range','decade', 'season', 'segment', 'depth_range', 'var']).agg({'cid' :lambda x: x.nunique()})
                     .reset_index()
                     .rename(columns={'cid':'cid_count'})
                     )


# %%

lat_avgs_df = (odf#drop(columns=['segment', 'source'])
                  .groupby(['lat_range','decade','season', 'segment', 'depth_range', 'var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})
                  #.drop(columns =['lat','lon','cid', 'year', 'month'])
                  )


lat_avgs_df.columns = lat_avgs_df.columns.to_flat_index().map('_'.join)

# %%

lat_avgs_df = (lat_avgs_df
                  # .drop(columns=['date_ordinal_std'])
                  .rename(columns={'date_ordinal_mean':'date_ordinal'})
                  .reset_index() 
                  .dropna()
                  .assign(
                          #segment=(lambda x: key),
                          # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                          # month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                          # season=(lambda x: pd.cut(x['month'],
                          #                          bins=[0,3,6,9,12],
                          #                          labels=['winter', 'spring', 'summer', 'fall'])),
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

lat_avgs_df = pd.merge(lat_avgs_df, lat_counts, how='left', on=['lat_range', 'decade', 'season', 'segment','depth_range', 'var'])

# %%

lat_avgs_df = lat_avgs_df[lat_avgs_df['cid_count'] >1]

lat_avgs_df['val_ci95hi'] = lat_avgs_df['val_mean'] + 1.96*lat_avgs_df['val_std']/np.sqrt(lat_avgs_df['cid_count'])

lat_avgs_df['val_ci95lo'] = lat_avgs_df['val_mean'] - 1.96*lat_avgs_df['val_std']/np.sqrt(lat_avgs_df['cid_count'])

# %%

for basin in basin_list:
    
    c = 0
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 8), squeeze=True, sharey=True)
    
    #ax = ax.flatten()
    
    plt.rc('font', size=14)
    
    for var in ['DO_mg_L']:
        
        if var =='SA':
            
            marker = 's'
            
            xmin = 22
            
            xmax = 32
        
        elif var == 'CT':
            
            marker = '^'
            
            xmin = 7
            
            xmax = 17
            
        else:
            
            marker = 'o'
            
            xmin = 0
            
            xmax = 12
                    
        for season in ['fall']:
            
            plot_df = lat_avgs_df[(lat_avgs_df['segment'] == basin) & (lat_avgs_df['var'] == var) & (lat_avgs_df['season'] == season) & (lat_avgs_df['decade'] == '2010')]
    
            if not plot_df.empty:
                
                if c==10:
                
                    sns.lineplot(data = plot_df, x='val_mean', y ='z_mean', hue='lat_range', palette='rocket_r', ax=ax, orient='y')
                
                else:
                    
                    sns.lineplot(data = plot_df, x='val_mean', y ='z_mean', hue='lat_range', palette='rocket_r', ax=ax, orient='y', legend=False)
                
                
                lat_range = '47-47.4'
                
                color = '#E7C3A3'
                
                                                
                ax.fill_betweenx(plot_df[plot_df['lat_range'] == lat_range]['z_mean'], plot_df[plot_df['lat_range'] == lat_range]['val_ci95lo'], plot_df[plot_df['lat_range'] == lat_range]['val_ci95hi'], zorder=-4, alpha=0.4, color=color)
                
                lat_range = '47.88-48'
                
                color = '#1C152A'
                                                
                ax.fill_betweenx(plot_df[plot_df['lat_range'] == lat_range]['z_mean'], plot_df[plot_df['lat_range'] == lat_range]['val_ci95lo'], plot_df[plot_df['lat_range'] == lat_range]['val_ci95hi'], zorder=-4, alpha=0.4, color=color)
                
        
                ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
                #ax[c].set_title(var + ' ' + season)
                
                ax.set_xlim(xmin, xmax)
                    
                
                ax.set_xlabel('DO [mg/L]')
            
            c+=1
    
    ax.set_ylabel('z [m]')
    
    # ax[4].set_ylabel('z [m]')
    
    # ax[8].set_ylabel('z [m]')

    #fig.suptitle('2010 ' + basin +' average casts')
    
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_average_casts_lat_season_CI_2010.png', dpi=500, transparent=True)
       
    