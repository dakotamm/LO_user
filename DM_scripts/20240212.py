#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 13:24:52 2024

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

poly_list = ['admiralty_sill', 'budd_inlet', 'carr_inlet_mid', 'dana_passage', 'hat_island', 'hazel_point', 'hood_canal_mouth', 'lynch_cove_shallow', 'near_alki', 'near_edmonds', 'port_susan_mid', 'saratoga_passage_north']

odf_dict = dfun.getPolyData(Ldir, poly_list, source_list=['ecology', 'nceiSalish', 'collias'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2023))

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

#odf = odf[(odf['val'] >= 0) & (odf['val'] <50)]
    
# %%

lc_exclude = odf[(odf['segment'] == 'lynch_cove_shallow') & (odf['z'] < -45)]

# %%

odf = odf[~odf['cid'].isin(lc_exclude['cid'].unique())]
    
# %%

# toward per decade average profiles, smoothed (bigger bins) - on same plot


odf = (odf
            .assign(
               # datetime=(lambda x: pd.to_datetime(x['time'])),
                 depth_range=(lambda x: pd.cut(x['z'], 
                                               bins=[-700, -355, -275, -205, -165, -135, -105, -80, -55, -37.5, -27.5, -17.5, -7.5, 0],
                                               labels= ['>355m', '275m-355m', '205m-275m', '165-205m','135m-165m','105m-135m', '80m-105m', '65m-80m','55m-80m','27.5m-37.5m', '17.5m-27.5m', '7.5m-17.5m', '<7.5m']))
                 # decade=(lambda x: pd.cut(x['year'],
                 #                          bins=[1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030],
                 #                          labels=['1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020']))
                 
                 
                 )
            )# make less manual

# %%

decade_std_temp = (odf#drop(columns=['segment', 'source'])
                  .groupby(['decade','season', 'segment', 'depth_range', 'var']).agg({'val':['mean', 'std']}) #, 'z':['mean'], 'date_ordinal':['mean']})
                  #.drop(columns =['lat','lon','cid', 'year', 'month'])
                  )

decade_std_temp.columns = decade_std_temp.columns.to_flat_index().map('_'.join)

decade_std_temp = decade_std_temp.reset_index()


# %%

odf_temp = pd.merge(odf, decade_std_temp, how='left', on=['decade', 'season', 'segment','depth_range', 'var'])

# %%

odf_decade = odf_temp[(odf_temp['val'] >= odf_temp['val_mean'] - 2*odf_temp['val_std']) & (odf_temp['val'] <= odf_temp['val_mean'] + 2*odf_temp['val_std'])]

odf_decade = odf_decade.drop(columns = ['val_mean', 'val_std'])

# %%

decade_counts = (odf_decade
                     .dropna()
                     #.set_index('datetime')
                     .groupby(['decade','season', 'segment', 'depth_range', 'var']).agg({'cid' :lambda x: x.nunique()})
                     .reset_index()
                     .rename(columns={'cid':'cid_count'})
                     )

# %%


decade_avgs_df = (odf_decade#drop(columns=['segment', 'source'])
                  .groupby(['decade','season', 'segment', 'depth_range', 'var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})
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

decade_avgs_df = pd.merge(decade_avgs_df, decade_counts, how='left', on=['decade', 'season', 'segment','depth_range', 'var'])

# %%

decade_avgs_df = decade_avgs_df[decade_avgs_df['cid_count'] >1]

decade_avgs_df['val_ci95hi'] = decade_avgs_df['val_mean'] + 1.96*decade_avgs_df['val_std']/np.sqrt(decade_avgs_df['cid_count'])

decade_avgs_df['val_ci95lo'] = decade_avgs_df['val_mean'] - 1.96*decade_avgs_df['val_std']/np.sqrt(decade_avgs_df['cid_count'])

# %%

# big map plot

fig, ax = plt.subplots()

plt.gcf().set_size_inches(45, 45)
 
#plt.rc('font', size=24)
 
#plot_df = odf_decade[(odf_decade['segment'] == basin) & (odf_decade['otype'] == 'bottle') & (odf_decade['var'] == 'DO_mg_L')].groupby('cid').first().reset_index()
  
#sns.scatterplot(data=plot_df, x='lon', y='lat', hue='decade', palette='plasma') #, alpha=0.5)

#sns.scatterplot(data=plot_df_mean, x='lon', y='lat', hue='decade', palette='Set2', marker='s', sizes=20)

#ax.autoscale(enable=False)

pfun.add_coast(ax)

pfun.dar(ax)

ax.set_xlim(-130, -121.5)

ax.set_ylim(45,51)

#ax.set_title(basin + ' bottle DO sampling locations')

fig.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/bigmap.svg', bbox_inches='tight', dpi=500)


# %%


for basin in ['hazel_point','dana_passage', 'hat_island', 'near_edmonds']:
    
    fig, ax = plt.subplots(nrows=3, ncols=1, squeeze=True, sharey=True)

    plt.gcf().set_size_inches(4, 10)
    
    c = 0
    
    #fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20, 15), squeeze=True, sharey=True)
    
    ax = ax.flatten()
    
    #plt.rc('font', size=12)
    
    for var in ['SA', 'CT', 'DO_mg_L']:
        
        if var =='SA':
            
            marker = 's'
            
            xmin = 22
            
            xmax = 32
            
            label = 'Salinity [PSU]'
        
        elif var == 'CT':
            
            marker = '^'
            
            xmin = 7
            
            xmax = 17
            
            label = 'Temperature [deg C]'
            
        else:
            
            marker = 'o'
            
            xmin = 0
            
            xmax = 12
            
            label = 'DO [mg/L]'
                    
        for season in ['fall']:
            
            plot_df = decade_avgs_df[(decade_avgs_df['segment'] == basin) & (decade_avgs_df['var'] == var) & (decade_avgs_df['season'] == season)]
    
            if not plot_df.empty:
                
                if c==12:
                
                    sns.lineplot(data = plot_df, x='val_mean', y ='z_mean', hue='decade', ax=ax[c], orient='y')
                
                else:
                    
                    sns.lineplot(data = plot_df, x='val_mean', y ='z_mean', hue='decade', palette='Blues', ax=ax[c], orient='y', legend=False)
                
                # decade = '1940'
                
                # color = '#7FA38D'
                
                decade = '1950'
                
                color = '#76189B'
                
                #sns.set_palette(palette)
                
                # decade = '1960'
                
                # color = '#83318B'
                
                # decade = '1930'
                
                # color = '#380996'
                
                                                
                ax[c].fill_betweenx(plot_df[plot_df['decade'] == decade]['z_mean'], plot_df[plot_df['decade'] == decade]['val_ci95lo'], plot_df[plot_df['decade'] == decade]['val_ci95hi'], zorder=-4, alpha=0.4, color=color)
                
                decade = '2010'
                
                color = '#F1B14E'
                                                
                ax[c].fill_betweenx(plot_df[plot_df['decade'] == decade]['z_mean'], plot_df[plot_df['decade'] == decade]['val_ci95lo'], plot_df[plot_df['decade'] == decade]['val_ci95hi'], zorder=-4, alpha=0.4, color=color)
                
                
                
                # for idx in plot_df_avgs_use.index:
                    
                #     ax[c].hlines(plot_df_avgs_use.loc[idx, 'z_mean'], plot_df_avgs_use.loc[idx, 'val_ci95lo'], plot_df_avgs_use.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
    
                #also CI!!!
                
                # ax[c].set_xlabel('Date')
        
                # ax[c].set_ylabel('DO [mg/L]')
        
                ax[c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
                #ax[c].set_title(var + ' ' + season)
                
                ax[c].set_xlim(xmin, xmax)
                    
                # if basin == 'lc':
                    
                #     ax[c].set_ylim([-50,0])
                
                ax[c].set_xlabel(label)
                
                ax[c].set_ylabel('z [m]')
                
            
            c+=1

    #fig.suptitle(basin +' average casts')
    
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_bottle_average_casts_decade_season_CI_1950_2010_FALL.png', dpi=500, transparent=False)
    
# %%

for basin in basin_list:
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    plt.rc('font', size=14)
     
    plot_df = odf_decade[(odf_decade['segment'] == basin) & (odf_decade['otype'] == 'bottle') & (odf_decade['var'] == 'DO_mg_L')].groupby('cid').first().reset_index()
    
   # plot_df_mean = odf[(odf['segment'] == basin) & (odf['otype'] == 'bottle') & (odf['var'] == 'DO_mg_L')].groupby('decade').agg({'lat':'mean', 'lon':'mean'}).reset_index()
    
    sns.scatterplot(data=plot_df, x='lon', y='lat', hue='decade', palette='plasma') #, alpha=0.5)
    
    #sns.scatterplot(data=plot_df_mean, x='lon', y='lat', hue='decade', palette='Set2', marker='s', sizes=20)
    
    ax.autoscale(enable=False)
    
    pfun.add_coast(ax)
    
    pfun.dar(ax)
    
    ax.set_xlim(-123.2, -122.1)
    
    ax.set_ylim(47,48.5)
    
    ax.set_title(basin + ' bottle DO sampling locations')

    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_bottle_DO_sampling_locations_by_decade.png', bbox_inches='tight', dpi=500)
