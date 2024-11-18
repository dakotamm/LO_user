#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:02:45 2024

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

# %%

monthly_skagit_df = pd.read_csv('/Users/dakotamascarenas/Desktop/skagit_monthly.txt',sep='\t',header=(35), skiprows=(36,36))


# %%



monthly_skagit_df['day'] = 1

monthly_skagit_df['datetime'] = pd.to_datetime(dict(year=monthly_skagit_df['year_nu'], month=monthly_skagit_df['month_nu'], day=monthly_skagit_df['day']))

monthly_skagit_df.loc[monthly_skagit_df['month_nu'].isin([12,1,2,3]), 'season'] = 'winter'

monthly_skagit_df.loc[monthly_skagit_df['month_nu'].isin([4,5,6,7]), 'season'] = 'grow'

monthly_skagit_df.loc[monthly_skagit_df['month_nu'].isin([8,9,10,11]), 'season'] = 'loDO'

monthly_skagit_df = monthly_skagit_df.assign(
                    decade=(lambda x: pd.cut(x['year_nu'],
                         bins=[1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2019, 2029],
                         labels=['1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'], right=True)),
                    date_ordinal=(lambda x: x['datetime'].apply(lambda x: x.toordinal())))

monthly_skagit_df['yearday'] = monthly_skagit_df['datetime'].dt.dayofyear

for i in [61, 92, 122, 153, 183, 214, 245, 275, 306, 336]:
    
    monthly_skagit_df.loc[monthly_skagit_df['yearday'] == i, 'yearday'] = i-1




# %%

annual_counts_skagit = (monthly_skagit_df
                      .dropna()
                      .groupby(['year_nu','season']).agg({'month_nu' :lambda x: x.nunique()})
                      .reset_index()
                      .rename(columns={'month_nu':'month_count'})
                      )


skagit_means = monthly_skagit_df.groupby(['year_nu', 'season']).agg({'mean_va':['mean', 'std'], 'date_ordinal':['mean']})

skagit_means.columns = skagit_means.columns.to_flat_index().map('_'.join)

skagit_means = skagit_means.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!


skagit_means = (skagit_means
                  .rename(columns={'date_ordinal_mean':'date_ordinal'})
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )


skagit_means = pd.merge(skagit_means, annual_counts_skagit, how='left', on=['year_nu','season'])


skagit_means = skagit_means[skagit_means['month_count'] >1] #redundant but fine (see note line 234)

skagit_means['mean_va_ci95hi'] = skagit_means['mean_va_mean'] + 1.96*skagit_means['mean_va_std']/np.sqrt(skagit_means['month_count'])

skagit_means['mean_va_ci95lo'] = skagit_means['mean_va_mean'] - 1.96*skagit_means['mean_va_std']/np.sqrt(skagit_means['month_count'])

# %%

skagit_means['year'] = skagit_means['year_nu']

skagit_means = skagit_means[['year', 'season', 'mean_va_mean']].pivot(index='year', columns = 'season', values = 'mean_va_mean').reset_index()

# %%

monthly_skagit_df['val'] = monthly_skagit_df['mean_va']

# %%

alpha = 0.05

x = monthly_skagit_df['date_ordinal']

y = monthly_skagit_df['val']


reject_null, p_value_all, Z = dfun.mann_kendall(y, alpha) #dfun
            

all_result = stats.theilslopes(y,x,alpha=alpha)



x = monthly_skagit_df[monthly_skagit_df['season'] == 'winter']['date_ordinal']

y = monthly_skagit_df[monthly_skagit_df['season'] == 'winter']['val']

reject_null, p_value_winter, Z = dfun.mann_kendall(y, alpha) #dfun



winter_result = stats.theilslopes(y,x,alpha=alpha)



x = monthly_skagit_df[monthly_skagit_df['season'] == 'grow']['date_ordinal']

y = monthly_skagit_df[monthly_skagit_df['season'] == 'grow']['val']

reject_null, p_value_grow, Z = dfun.mann_kendall(y, alpha) #dfun



grow_result = stats.theilslopes(y,x,alpha=alpha)


x = monthly_skagit_df[monthly_skagit_df['season'] == 'loDO']['date_ordinal']

y = monthly_skagit_df[monthly_skagit_df['season'] == 'loDO']['val']

reject_null, p_value_loDO, Z = dfun.mann_kendall(y, alpha) #dfun



loDO_result = stats.theilslopes(y,x,alpha=alpha)

# %%

decadal_yearday_mean = monthly_skagit_df.groupby(['decade', 'yearday']).mean(numeric_only=True).reset_index()

decadal_yearday_mean = (decadal_yearday_mean
                  #.rename(columns={'date_ordinal_mean':'date_ordinal'})
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

# %%

fig, ax = plt.subplots(nrows=2, figsize = (8,6))


ax[0].plot(monthly_skagit_df['datetime'], monthly_skagit_df['val'], color= 'gray', label = 'Full')

x_plot = monthly_skagit_df['datetime']

x = monthly_skagit_df['date_ordinal']

y = monthly_skagit_df['val']

if p_value_all < 0.05:
    
    ax[0].plot([x_plot.min(), x_plot.max()], [all_result.intercept + all_result.slope*x.min(), all_result.intercept + all_result.slope*x.max()], alpha =0.7, linestyle = 'dashed', linewidth=2, label='all')


x = monthly_skagit_df[monthly_skagit_df['season'] == 'winter']['date_ordinal']

y = monthly_skagit_df[monthly_skagit_df['season'] == 'winter']['val']

x_plot = monthly_skagit_df[monthly_skagit_df['season'] == 'winter']['datetime']


if p_value_winter < 0.05:
    
    ax[0].plot([x_plot.min(), x_plot.max()], [winter_result.intercept + winter_result.slope*x.min(), winter_result.intercept + winter_result.slope*x.max()], alpha =0.7, linestyle = 'dashed', linewidth=2, label='winter')


x = monthly_skagit_df[monthly_skagit_df['season'] == 'grow']['date_ordinal']

y = monthly_skagit_df[monthly_skagit_df['season'] == 'grow']['val']

x_plot = monthly_skagit_df[monthly_skagit_df['season'] == 'grow']['datetime']


if p_value_grow < 0.05:
    
    ax[0].plot([x_plot.min(), x_plot.max()], [grow_result.intercept + grow_result.slope*x.min(), grow_result.intercept + grow_result.slope*x.max()], alpha =0.7, linestyle = 'dashed', linewidth=2, label='grow')



x = monthly_skagit_df[monthly_skagit_df['season'] == 'loDO']['date_ordinal']

y = monthly_skagit_df[monthly_skagit_df['season'] == 'loDO']['val']

x_plot = monthly_skagit_df[monthly_skagit_df['season'] == 'loDO']['datetime']


if p_value_loDO < 0.05:
    
    ax[0].plot([x_plot.min(), x_plot.max()], [loDO_result.intercept + loDO_result.slope*x.min(), loDO_result.intercept + loDO_result.slope*x.max()], alpha =0.7, linestyle = 'dashed', linewidth=2, label='loDO')


ax[0].legend()



sns.lineplot(data = decadal_yearday_mean, x='yearday', y = 'val', hue='decade', ax=ax[1], palette='plasma_r')



ax[0].set_ylabel('Monthly Mean Discharge [cfs]')

ax[1].set_ylabel('Decadal Mean Discharge [cfs]')


ax[0].set_xlabel('')

ax[1].set_xlabel('Yearday')




ax[0].grid(color = 'lightgray', linestyle = '--', alpha=0.5)

ax[1].grid(color = 'lightgray', linestyle = '--', alpha=0.5)




plt.tight_layout()


plt.savefig('/Users/dakotamascarenas/Desktop/pltz/skagit_trends_decadalhydrographs.png', bbox_inches='tight', dpi=500, transparent=False)




