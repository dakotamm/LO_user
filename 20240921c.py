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

monthly_skagit_df.loc[monthly_skagit_df['month_nu'].isin([12,1,2,3]), 'season'] = 'Winter (Dec-Mar)'

monthly_skagit_df.loc[monthly_skagit_df['month_nu'].isin([4,5,6,7]), 'season'] = 'Grow Season (Apr-Jul)'

monthly_skagit_df.loc[monthly_skagit_df['month_nu'].isin([8,9,10,11]), 'season'] = 'Low-DO Season (Aug-Nov)'

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



x = monthly_skagit_df[monthly_skagit_df['season'] == 'Winter (Dec-Mar)']['date_ordinal']

y = monthly_skagit_df[monthly_skagit_df['season'] == 'Winter (Dec-Mar)']['val']

reject_null, p_value_winter, Z = dfun.mann_kendall(y, alpha) #dfun



winter_result = stats.theilslopes(y,x,alpha=alpha)



x = monthly_skagit_df[monthly_skagit_df['season'] == 'Grow Season (Apr-Jul)']['date_ordinal']

y = monthly_skagit_df[monthly_skagit_df['season'] == 'Grow Season (Apr-Jul)']['val']

reject_null, p_value_grow, Z = dfun.mann_kendall(y, alpha) #dfun



grow_result = stats.theilslopes(y,x,alpha=alpha)


x = monthly_skagit_df[monthly_skagit_df['season'] == 'Low-DO Season (Aug-Nov)']['date_ordinal']

y = monthly_skagit_df[monthly_skagit_df['season'] == 'Low-DO Season (Aug-Nov)']['val']

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

fig, ax = plt.subplots(figsize = (6,4))

palette = {'Winter (Dec-Mar)': '#4565e8', 'Low-DO Season (Aug-Nov)':'#e04256', 'Grow Season (Apr-Jul)': '#6cbb3c'}

sns.scatterplot(data=monthly_skagit_df, x = 'datetime', y = 'val', hue = 'season', palette=palette, alpha=0.3)


#ax[0].plot(monthly_skagit_df['datetime'], monthly_skagit_df['val'], color= 'gray', label = 'Full')

x_plot = monthly_skagit_df['datetime']

x = monthly_skagit_df['date_ordinal']

y = monthly_skagit_df['val']


    
ax.plot([x_plot.min(), x_plot.max()], [all_result.intercept + all_result.slope*x.min(), all_result.intercept + all_result.slope*x.max()], linestyle = ':', linewidth=3, color = 'gray', label='All Time Series Trend (NOT SIGNIFICANT)')






x = monthly_skagit_df[monthly_skagit_df['season'] == 'Winter (Dec-Mar)']['date_ordinal']

y = monthly_skagit_df[monthly_skagit_df['season'] == 'Winter (Dec-Mar)']['val']

x_plot = monthly_skagit_df[monthly_skagit_df['season'] == 'Winter (Dec-Mar)']['datetime']


    
ax.plot([x_plot.min(), x_plot.max()], [winter_result.intercept + winter_result.slope*x.min(), winter_result.intercept + winter_result.slope*x.max()], color = palette['Winter (Dec-Mar)'], linestyle = 'dashed', linewidth=2, label='Winter Trend')


x = monthly_skagit_df[monthly_skagit_df['season'] == 'Grow Season (Apr-Jul)']['date_ordinal']

y = monthly_skagit_df[monthly_skagit_df['season'] == 'Grow Season (Apr-Jul)']['val']

x_plot = monthly_skagit_df[monthly_skagit_df['season'] == 'Grow Season (Apr-Jul)']['datetime']


    
ax.plot([x_plot.min(), x_plot.max()], [grow_result.intercept + grow_result.slope*x.min(), grow_result.intercept + grow_result.slope*x.max()], color = palette['Grow Season (Apr-Jul)'], linestyle = 'dashed', linewidth=2, label='Grow Season Trend')



x = monthly_skagit_df[monthly_skagit_df['season'] == 'Low-DO Season (Aug-Nov)']['date_ordinal']

y = monthly_skagit_df[monthly_skagit_df['season'] == 'Low-DO Season (Aug-Nov)']['val']

x_plot = monthly_skagit_df[monthly_skagit_df['season'] == 'Low-DO Season (Aug-Nov)']['datetime']


    
ax.plot([x_plot.min(), x_plot.max()], [loDO_result.intercept + loDO_result.slope*x.min(), loDO_result.intercept + loDO_result.slope*x.max()], color = 'gray', linestyle = 'dashed', linewidth=2, label='Low-DO Season Trend (NOT SIGNFICANT)')


ax.legend(loc='upper left')



#sns.lineplot(data = decadal_yearday_mean, x='yearday', y = 'val', hue='decade', ax=ax[1], palette='plasma_r')



ax.set_ylabel('Monthly Mean Discharge [cfs]')

#ax[1].set_ylabel('Decadal Mean Discharge [cfs]')


ax.set_xlabel('')

#ax[1].set_xlabel('Yearday')




ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)

#ax[1].grid(color = 'lightgray', linestyle = '--', alpha=0.5)




plt.tight_layout()


plt.savefig('/Users/dakotamascarenas/Desktop/pltz/skagit_trends_PRESENT.png', bbox_inches='tight', dpi=500, transparent=True)




