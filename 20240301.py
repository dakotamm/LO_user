#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:09:33 2024

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

poly_list = ['admiralty_sill', 'budd_inlet', 'carr_inlet_mid', 'dana_passage', 'hat_island', 'hazel_point', 'hood_canal_mouth', 'lynch_cove_shallow', 'near_alki', 'near_edmonds', 'port_susan_mid', 'saratoga_passage_north', 'ps', 'hc_wo_lc']

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

odf[(odf['source'] == 'ecology') & (odf['var'] == 'DO_mg_L') & (odf['otype'] == 'bottle')] = np.nan

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

fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True, sharex=True)

plt.gcf().set_size_inches(9, 3)


for var in ['DO_mg_L']:
    
    if var =='SA':
        
        marker = 's'
        
        xmin = 26
        
        xmax = 32
        
        label = 'Salinity [PSU]'
        
        color = ''
    
    elif var == 'CT':
        
        marker = '^'
        
        xmin = 7
        
        xmax = 14
        
        label = 'Temperature [deg C]'
        
    else:
        
        marker = 'o'
        
        xmin = 0
        
        xmax = 10
        
        label = 'DO [mg/L]'
        
        
        

    sns.scatterplot(data=odf_decade[(odf_decade['segment'] == 'ps') & (odf_decade['season'] == 'fall') & (odf_decade['var'] == var)], x='datetime', y='z', ax= ax, color='gray', alpha=0.2, s=5, edgecolor=None)
    
    sns.scatterplot(data=odf_decade[(odf_decade['segment'] == 'near_edmonds') & (odf_decade['season'] == 'fall') & (odf_decade['var'] == var)], x='datetime', y='z', ax= ax, color='b', s=15)
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
    ax.set_ylabel('Depth [m]')
    
    ax.set_xlabel('Year')
    
    ax.set_xlim(datetime.datetime.strptime('01/01/1930', '%m/%d/%Y'),datetime.datetime.strptime('12/31/2025', '%m/%d/%Y'))

   # ax[c].set_ylim(xmin, xmax)
    
   # ax[c].set_ylabel(label)
    
    #ax[c].patch.set_facecolor('#dbeaf5')
    
    fig.tight_layout()


    
    
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/fall_DO_depths_w_near_edmonds.png', dpi=500, transparent=True)

# %%

fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True, sharex=True)

plt.gcf().set_size_inches(9, 3)

ax.axvspan(datetime.datetime.strptime('01/01/1950', '%m/%d/%Y'),datetime.datetime.strptime('12/31/1959', '%m/%d/%Y'), color = '#8ad6cc', alpha = 0.5)

ax.axvspan(datetime.datetime.strptime('01/01/2010', '%m/%d/%Y'),datetime.datetime.strptime('12/31/2019', '%m/%d/%Y'), color = '#f97171', alpha = 0.5)

ax.axvspan(datetime.datetime.strptime('01/01/1930', '%m/%d/%Y'),datetime.datetime.strptime('12/31/1939', '%m/%d/%Y'), color = 'lightgray', alpha = 0.5)



sns.scatterplot(data=odf_decade[(odf_decade['segment'] == 'near_edmonds') & (odf_decade['season'] == 'fall') & (odf_decade['var'] == var)], x='datetime', y='val', ax= ax, color='b')

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)



if var == 'DO_mg_L':
    
    ax.axhspan(0,2,color = 'lightgray', alpha = 0.4)

ax.set_ylabel('DO [mg/L]')

ax.set_ylim(xmin, xmax)

    
ax.set_xlabel('Year')

ax.set_xlim(datetime.datetime.strptime('01/01/1930', '%m/%d/%Y'),datetime.datetime.strptime('12/31/2025', '%m/%d/%Y'))


fig.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/near_edmonds_DO_time_series_w_1930.png', dpi=500, transparent=True)

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

for basin in ['near_edmonds', 'dana_passage']:
    
    c = 0
    
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 12), squeeze=True, sharey=True)
    
    ax = ax.flatten()
    
    plt.rc('font', size=12)
    
    for var in ['SA', 'CT', 'DO_mg_L']:
        
        if var =='SA':
            
            marker = 's'
            
            xmin = 22
            
            xmax = 32
            
            var_label = 'salinity [psu]'
        
        elif var == 'CT':
            
            marker = '^'
            
            xmin = 7
            
            xmax = 17
            
            var_label = 'temperature [deg C]'
            
        else:
            
            marker = 'o'
            
            xmin = 0
            
            xmax = 12
            
            var_label = 'DO [mg/L]'
                    
        for season in ['winter', 'spring', 'summer', 'fall']:
            
            plot_df = decade_avgs_df[(decade_avgs_df['segment'] == basin) & (decade_avgs_df['var'] == var) & (decade_avgs_df['season'] == season)]
    
            if not plot_df.empty:
                
                if c==0:
                
                    sns.lineplot(data = plot_df, x='val_mean', y ='z_mean', hue='decade', palette='plasma', ax=ax[c], orient='y')
                
                else:
                    
                    sns.lineplot(data = plot_df, x='val_mean', y ='z_mean', hue='decade', palette='plasma', ax=ax[c], orient='y', legend=False)
                
                # decade = '1940'
                
                # color = '#7FA38D'
                
                # decade = '1950'
                
                # color = '#6E9B86'
                
                # decade = '1960'
                
                # color = '#83318B'
                
                                                
                # ax[c].fill_betweenx(plot_df[plot_df['decade'] == decade]['z_mean'], plot_df[plot_df['decade'] == decade]['val_ci95lo'], plot_df[plot_df['decade'] == decade]['val_ci95hi'], zorder=-4, alpha=0.4, color=color)
                
                # decade = '2010'
                
                # color = '#F1B14E'
                                                
                # ax[c].fill_betweenx(plot_df[plot_df['decade'] == decade]['z_mean'], plot_df[plot_df['decade'] == decade]['val_ci95lo'], plot_df[plot_df['decade'] == decade]['val_ci95hi'], zorder=-4, alpha=0.4, color=color)
                
                
                
                # for idx in plot_df_avgs_use.index:
                    
                #     ax[c].hlines(plot_df_avgs_use.loc[idx, 'z_mean'], plot_df_avgs_use.loc[idx, 'val_ci95lo'], plot_df_avgs_use.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
    
                #also CI!!!
                
                # ax[c].set_xlabel('Date')
        
                # ax[c].set_ylabel('DO [mg/L]')
        
                ax[c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
                ax[c].set_title(season)
                
                ax[c].set_xlim(xmin, xmax)
                
                ax[c].set_ylim(top=0)
                    
                # if basin == 'lc':
                    
                #     ax[c].set_ylim([-50,0])
                
                ax[c].set_xlabel(var_label)
            
            c+=1
    
    ax[0].set_ylabel('z [m]')
    
    ax[4].set_ylabel('z [m]')
    
    ax[8].set_ylabel('z [m]')

    #fig.suptitle(basin +' average casts')
    
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_average_casts_decade_season.png', dpi=500, transparent=True)
    
# %%

for basin in ['near_edmonds']:
    
    
    fig, ax = plt.subplots(nrows=3, ncols=1, squeeze=True, sharey=True)

    plt.gcf().set_size_inches(4,9)
    
    c=0
    
    
    for var in ['SA', 'CT', 'DO_mg_L']:
        
        if var =='SA':
            
            marker = 's'
            
            xmin = 22
            
            xmax = 32
            
            label = 'Salinity [PSU]'
            
            color_less = '#393070'
            
            color_more = '#ABCF43'
                    
        elif var == 'CT':
            
            marker = '^'
            
            xmin = 7
            
            xmax = 17
            
            label = 'Temperature [deg C]'
            
            color_less = '#466EA9'
            
            color_more = '#9C3C49'
            
        else:
            
            marker = 'o'
            
            xmin = 0
            
            xmax = 12
            
            label = 'DO [mg/L]'
            
            color_more = '#3E1F95'
            
            color_less = '#EAB63A'
            
            
                    
        for season in ['fall']:
            
            plot_df = decade_avgs_df[(decade_avgs_df['segment'] == basin) & (decade_avgs_df['var'] == var) & (decade_avgs_df['season'] == season)]
            
            decade0 = '1950'
            
            color0 = '#8ad6cc'
            
            decade1 = '2010'
            
            color1 = '#f97171'
            
            decade2 = '1930'
            
            color2 = 'lightgray'
    
            if not plot_df.empty:
                    
                sns.lineplot(data = plot_df[plot_df['decade'] == decade0], x='val_mean', y ='z_mean', color = color0, ax=ax[c], orient='y', legend=False)
                
                sns.lineplot(data = plot_df[plot_df['decade'] == decade1], x='val_mean', y ='z_mean', color = color1, ax=ax[c], orient='y', legend=False)

                sns.lineplot(data = plot_df[plot_df['decade'] == decade2], x='val_mean', y ='z_mean', color = color2, ax=ax[c], orient='y', legend=False)

                                
                ax[c].fill_betweenx(plot_df[plot_df['decade'] == decade0]['z_mean'], plot_df[plot_df['decade'] == decade0]['val_ci95lo'], plot_df[plot_df['decade'] == decade0]['val_ci95hi'],
                                 zorder=-4, alpha=0.5, color=color0)
                
                ax[c].fill_betweenx(plot_df[plot_df['decade'] == decade1]['z_mean'], plot_df[plot_df['decade'] == decade1]['val_ci95lo'], plot_df[plot_df['decade'] == decade1]['val_ci95hi'],
                                 zorder=-4, alpha=0.5, color=color1)
                
                ax[c].fill_betweenx(plot_df[plot_df['decade'] == decade2]['z_mean'], plot_df[plot_df['decade'] == decade2]['val_ci95lo'], plot_df[plot_df['decade'] == decade2]['val_ci95hi'],
                                 zorder=-4, alpha=0.5, color=color2)
                
                if var == 'DO_mg_L':
                    
                    ax[c].axvspan(0,2, color = 'lightgray', alpha = 0.2)
        
                ax[c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
                        
                ax[c].set_xlim(xmin, xmax)
                    
                
                ax[c].set_xlabel(label)
                
                ax[c].set_ylabel('z [m]')
                

                
            
        c+=1

    #fig.suptitle(basin +' average casts')
    
    fig.tight_layout()
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin +'_average_casts_decade_season_CI_1930_1950_2010_FALL.png', dpi=500, transparent=True)
