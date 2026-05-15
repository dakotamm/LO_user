#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:53:35 2024

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


# %%

temp_df = pd.read_csv('/Users/dakotamascarenas/Desktop/ps_longterm/ptools_data/ncdc/1527887.csv')


temp_df['datetime'] = pd.to_datetime(temp_df['DATE'])

temp_df['year'] = pd.DatetimeIndex(temp_df['datetime']).year


temp_df['month'] = pd.DatetimeIndex(temp_df['datetime']).month


temp_df['year_month'] = temp_df['year'].astype(str) + '_' + temp_df['month'].astype(str).apply(lambda x: x.zfill(2))

temp_df['date_ordinal'] = temp_df['datetime'].apply(lambda x: x.toordinal())


# %%

temp_monthly_avg_df = temp_df[['datetime', 'date_ordinal', 'year', 'month', 'year_month', 'TMAX', 'TMIN']].groupby(['year_month']).mean().reset_index().dropna()

# %%

# using all monthly averages

# using grow, loDO, and winter averages

stats_df = pd.DataFrame()

stat = 'mk_ts'

alpha = 0.05

    
for season in ['allyear', 'grow', 'loDO', 'winter']:
            
    if season == 'allyear':
                
        plot_df = temp_monthly_avg_df
        
    elif season == 'grow':
        
        plot_df = temp_monthly_avg_df[temp_monthly_avg_df['month'].isin([4,5,6,7])]
        
    elif season == 'loDO':
        
        plot_df = temp_monthly_avg_df[temp_monthly_avg_df['month'].isin([8,9,10,11])]
        
    elif season == 'winter':
        
        plot_df = temp_monthly_avg_df[temp_monthly_avg_df['month'].isin([12,1,2,3])]
        
    x = plot_df['date_ordinal']
    
    x_plot = plot_df['datetime']

    for var in ['TMAX', 'TMIN']:
        
        y = plot_df[var]
        
        plot_df['stat'] = stat
                
        reject_null, p_value, Z = dfun.mann_kendall(y, alpha) #dfun
                            
        plot_df['p'] = p_value
        
        result = stats.theilslopes(y,x,alpha=alpha)
        
        B1 = result.slope
        
        B0 = result.intercept
                
        plot_df['B1'] = B1

        plot_df['B0'] = B0
        
        plot_df['hi_sB1'] = result.high_slope
        
        plot_df['lo_sB1']  = result.low_slope
        
        slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_plot.max().year - x_plot.min().year)

        plot_df['slope_datetime'] = slope_datetime #per year
        
        slope_datetime_s_hi = (B0 + result.high_slope*x.max() - (B0 + result.high_slope*x.min()))/(x_plot.max().year - x_plot.min().year)
        
        slope_datetime_s_lo = (B0 + result.low_slope*x.max() - (B0 + result.low_slope*x.min()))/(x_plot.max().year - x_plot.min().year)
        
        plot_df['slope_datetime_s_hi'] = slope_datetime_s_hi #per year
        
        plot_df['slope_datetime_s_lo'] = slope_datetime_s_lo #per year
        
        plot_df['season'] = season
        
        plot_df['var'] = var
                                                                                        
        plot_df_concat = plot_df[['season','var', 'p', 'hi_sB1', 'lo_sB1', 'B1', 'B0', 'slope_datetime', 'slope_datetime_s_hi', 'slope_datetime_s_lo']].head(1) #slope_datetime_unc_cent, slope_datetime_s
    
        stats_df = pd.concat([stats_df, plot_df_concat])
        
# %%

fig, ax = plt.subplots(figsize=(8,4), sharex=True, sharey=True)

plot_df = stats_df.copy()

#plot_df = plot_df[plot_df['season'].isin(['yearlong','loDO'])]


plot_df['season_minmax'] = plot_df['season'] + '_' + plot_df['var']

plot_df = plot_df.sort_values(by='season_minmax').reset_index()

plot_df['slope_datetime_cent_95hi'] = plot_df['slope_datetime_s_hi']*100

plot_df['slope_datetime_cent_95lo'] = plot_df['slope_datetime_s_lo']*100

plot_df['slope_datetime_cent'] = plot_df['slope_datetime']*100




sns.scatterplot(data = plot_df, x= 'season_minmax', y = 'slope_datetime_cent_95hi', hue='var', hue_order = ['TMIN', 'TMAX'], style = 'season', style_order = ['allyear', 'winter', 'grow', 'loDO'], ax = ax, s= 50, legend=False)

sns.scatterplot(data = plot_df, x= 'season_minmax', y = 'slope_datetime_cent_95lo', hue='var', hue_order = ['TMIN', 'TMAX'], style = 'season', ax = ax, s= 50, legend=False)

sns.scatterplot(data = plot_df, x= 'season_minmax', y = 'slope_datetime_cent', hue='var', hue_order = ['TMIN', 'TMAX'], style = 'season', ax = ax, s =250, legend=False)

for idx in plot_df.index:
    
    if plot_df.loc[idx,'var'] == 'TMIN': 
        
        ax.plot([plot_df.loc[idx,'season_minmax'], plot_df.loc[idx,'season_minmax']],[plot_df.loc[idx,'slope_datetime_cent_95lo'], plot_df.loc[idx,'slope_datetime_cent_95hi']], color='#1f77b4', alpha =0.7, zorder = -5, linewidth=1)

    else:
        
        
        
        ax.plot([plot_df.loc[idx,'season_minmax'], plot_df.loc[idx,'season_minmax']],[plot_df.loc[idx,'slope_datetime_cent_95lo'], plot_df.loc[idx,'slope_datetime_cent_95hi']], color='#ff7f0e', alpha =0.7, zorder = -4, linewidth=1)




ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)

ax.axhline(0, color='gray', linestyle = '--', zorder = -5)


ax.tick_params(axis='x', rotation=45)

ax.set_ylabel(r'[$^{\circ}$C]/century', wrap=True)

ax.set_xlabel('')



plt.savefig('/Users/dakotamascarenas/Desktop/pltz/testy.png', dpi=500,transparent=False, bbox_inches='tight')   

    