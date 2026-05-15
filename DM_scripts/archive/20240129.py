#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:38:03 2024

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

poly_list = ['hazel_point', 'hc_mouth']

odf_dict = dfun.getPolyData(Ldir, poly_list, source_list=['ecology', 'nceiSalish', 'collias'], otype_list=['bottle'])

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

# make df??? not until after counts...honestly doesn't really matter i guess...

odf = pd.concat(odf_dict.values(), ignore_index=True)

odf = odf[odf['var'].isin(['DO_mg_L', 'CT', 'SA'])]

odf = odf[(odf['val'] >= 0) & (odf['val'] <50)]
    
    
    
# %%

# plot 2 - per decade average profiles, smoothed (bigger bins) - on same plot


odf = (odf
            .assign(
               # datetime=(lambda x: pd.to_datetime(x['time'])),
                 depth_range=(lambda x: pd.cut(x['z'], 
                                               bins=[-400, -355, -275, -205, -165, -135, -105, -80, -55, -37.5, -27.5, -17.5, -7.5, 0],
                                               labels= ['>355m', '275m-355m', '205m-275m', '165-205m','135m-165m','105m-135m', '80m-105m', '65m-80m','55m-80m','27.5m-37.5m', '17.5m-27.5m', '7.5m-17.5m', '<7.5m']))
                 # decade=(lambda x: pd.cut(x['year'],
                 #                          bins=[1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030],
                 #                          labels=['1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020']))
                 
                 
                 )
            )# make less manual

# %%

decade_counts = (odf
                     .dropna()
                     #.set_index('datetime')
                     .groupby(['decade','season', 'segment', 'depth_range', 'var', 'otype']).agg({'cid' :lambda x: x.nunique()})
                     .reset_index()
                     .rename(columns={'cid':'cid_count'})
                     )

# %%

decade_avgs_df = (odf#drop(columns=['segment', 'source'])
                  .groupby(['decade','season', 'segment', 'depth_range', 'var', 'otype']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})
                  #.drop(columns =['lat','lon','cid', 'year', 'month'])
                  )


decade_avgs_df.columns = decade_avgs_df.columns.to_flat_index().map('_'.join)

# %%

decade_avgs_df = (decade_avgs_df
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

decade_avgs_df = pd.merge(decade_avgs_df, decade_counts, how='left', on=['decade', 'season', 'segment','depth_range', 'var', 'otype'])

# %%

decade_avgs_df = decade_avgs_df[decade_avgs_df['cid_count'] >1]

decade_avgs_df['val_ci95hi'] = decade_avgs_df['val_mean'] + 1.96*decade_avgs_df['val_std']/np.sqrt(decade_avgs_df['cid_count'])

decade_avgs_df['val_ci95lo'] = decade_avgs_df['val_mean'] - 1.96*decade_avgs_df['val_std']/np.sqrt(decade_avgs_df['cid_count'])

# %%


for basin in basin_list:
    
    for var in ['SA', 'CT', 'DO_mg_L']:
        
        if var =='SA':
            
            marker = 's'
        
        elif var == 'CT':
            
            marker = '^'
            
        else:
            
            marker = 'o'
            
        plot_df = odf[(odf['segment'] == basin) & (odf['var'] == var) & (odf['otype'] == 'bottle')]
            
        sns.relplot(data=plot_df, x='val', y ='z', col='season', row = 'decade', markers=marker, alpha=0.5, hue='season')
        
        #fig.set_title(basin + ' ' + var)
        
       # fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + var + '_bottle_sampling_depths_decade_season.png', dpi=500)



# %%

# average seasonal/decadal plots

for basin in basin_list:
    
    c = 0
    
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20, 15), squeeze=True, sharey=True)
    
    ax = ax.flatten()
    
    plt.rc('font', size=14)
    
    for var in ['SA', 'CT', 'DO_mg_L']:
        
        if var =='SA':
            
            marker = 's'
        
        elif var == 'CT':
            
            marker = '^'
            
        else:
            
            marker = 'o'
                    
        for season in ['winter', 'spring', 'summer', 'fall']:
            
            plot_df = decade_avgs_df[(decade_avgs_df['segment'] == basin) & (decade_avgs_df['otype'] == 'bottle') & (decade_avgs_df['var'] == var) & (decade_avgs_df['season'] == season)]
    
            if not plot_df.empty:
                                
                sns.lineplot(data = plot_df, x='val_mean', y ='z_mean', hue='decade', palette='crest', ax=ax[c], orient='y')
                
               # ax[c].fill_betweenx(plot_df['z_mean'], plot_df['val_ci95lo'], plot_df['val_ci95hi'], zorder=-4, color='gray', alpha=0.7)
                
                # for idx in plot_df_avgs_use.index:
                    
                #     ax[c].hlines(plot_df_avgs_use.loc[idx, 'z_mean'], plot_df_avgs_use.loc[idx, 'val_ci95lo'], plot_df_avgs_use.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
    
                #also CI!!!
                
                # ax[c].set_xlabel('Date')
        
                # ax[c].set_ylabel('DO [mg/L]')
        
                ax[c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
                ax[c].set_title(var + ' ' + season)
                    
                # if basin == 'lc':
                    
                #     ax[c].set_ylim([-50,0])
                
                ax[c].set_xlabel(var)
            
            c+=1
    
    ax[0].set_ylabel('z [m]')
    
    ax[4].set_ylabel('z [m]')
    
    ax[8].set_ylabel('z [m]')

    fig.suptitle(basin +' average casts')
    
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_bottle_average_casts_decade_season.png', dpi=500)
    
# %%

for basin in basin_list:
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    plt.rc('font', size=14)
     
    plot_df = odf[(odf['segment'] == basin) & (odf['otype'] == 'bottle') & (odf['var'] == 'DO_mg_L')].groupby('cid').first().reset_index()
    
   # plot_df_mean = odf[(odf['segment'] == basin) & (odf['otype'] == 'bottle') & (odf['var'] == 'DO_mg_L')].groupby('decade').agg({'lat':'mean', 'lon':'mean'}).reset_index()
    
    sns.scatterplot(data=plot_df, x='lon', y='lat', hue='decade', palette='crest') #, alpha=0.5)
    
    #sns.scatterplot(data=plot_df_mean, x='lon', y='lat', hue='decade', palette='Set2', marker='s', sizes=20)
    
    ax.autoscale(enable=False)
    
    pfun.add_coast(ax)
    
    pfun.dar(ax)
    
    ax.set_xlim(-123.2, -122.5)
    
    ax.set_ylim(47.3,48)
    
    ax.set_title(basin + ' bottle DO sampling locations')

    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_bottle_DO_sampling_locations_by_decade.png', bbox_inches='tight', dpi=500)
    
# %%
    
for basin in basin_list:
    
    c = 0
    
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20, 15), squeeze=True, sharey=True)
    
    ax = ax.flatten()
    
    plt.rc('font', size=14)
    
    for var in ['SA', 'CT', 'DO_mg_L']:
        
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
                    
        for season in ['winter', 'spring', 'summer', 'fall']:
            
            plot_df = decade_avgs_df[(decade_avgs_df['segment'] == basin) & (decade_avgs_df['otype'] == 'bottle') & (decade_avgs_df['var'] == var) & (decade_avgs_df['season'] == season)]
    
            if not plot_df.empty:
                
                if c==0:
                
                    sns.lineplot(data = plot_df, x='val_mean', y ='z_mean', hue='decade', palette='crest', ax=ax[c], orient='y')
                
                else:
                    
                    sns.lineplot(data = plot_df, x='val_mean', y ='z_mean', hue='decade', palette='crest', ax=ax[c], orient='y', legend=False)
                
                # decade = '1940'
                
                # color = '#7FA38D'
                
                # decade = '1950'
                
                # color = '#6E9B86'
                
                decade = '1930'
                
                color = '#93B688'
                
                                                
                ax[c].fill_betweenx(plot_df[plot_df['decade'] == decade]['z_mean'], plot_df[plot_df['decade'] == decade]['val_ci95lo'], plot_df[plot_df['decade'] == decade]['val_ci95hi'], zorder=-4, alpha=0.4, color=color)
                
                decade = '2010'
                
                color = '#335180'
                                                
                ax[c].fill_betweenx(plot_df[plot_df['decade'] == decade]['z_mean'], plot_df[plot_df['decade'] == decade]['val_ci95lo'], plot_df[plot_df['decade'] == decade]['val_ci95hi'], zorder=-4, alpha=0.4, color=color)
                
                
                
                # for idx in plot_df_avgs_use.index:
                    
                #     ax[c].hlines(plot_df_avgs_use.loc[idx, 'z_mean'], plot_df_avgs_use.loc[idx, 'val_ci95lo'], plot_df_avgs_use.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
    
                #also CI!!!
                
                # ax[c].set_xlabel('Date')
        
                # ax[c].set_ylabel('DO [mg/L]')
        
                ax[c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
                ax[c].set_title(var + ' ' + season)
                
                ax[c].set_xlim(xmin, xmax)
                    
                # if basin == 'lc':
                    
                #     ax[c].set_ylim([-50,0])
                
                ax[c].set_xlabel(var)
            
            c+=1
    
    ax[0].set_ylabel('z [m]')
    
    ax[4].set_ylabel('z [m]')
    
    ax[8].set_ylabel('z [m]')

    fig.suptitle(basin +' average casts')
    
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_bottle_average_casts_decade_season_CI_1930_2010.png', dpi=500)