#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:54:44 2024

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
                         bins=[1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2019, 2029], #removed 30s*************
                         labels=['1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s'], right=True)),
                    date_ordinal=(lambda x: x['datetime'].apply(lambda x: x.toordinal())))

monthly_skagit_df['yearday'] = monthly_skagit_df['datetime'].dt.dayofyear

for i in [61, 92, 122, 153, 183, 214, 245, 275, 306, 336]:
    
    monthly_skagit_df.loc[monthly_skagit_df['yearday'] == i, 'yearday'] = i-1

monthly_skagit_df['mean_va'] = monthly_skagit_df['mean_va']*0.028316832 #cfs to m^3/s


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

monthly_skagit_df = monthly_skagit_df.assign(
    datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x)))))

# %%

stats_df = pd.DataFrame()

stat = 'mk_ts'

alpha = 0.05


    
for season in ['allyear', 'grow', 'loDO', 'winter']:
            
    if season == 'allyear':
                
        plot_df = monthly_skagit_df.copy()
        
    else:
        
        plot_df = monthly_skagit_df[monthly_skagit_df['season'] == season].copy()
        
    x = plot_df['date_ordinal']
    
    x_plot = plot_df['datetime']
    
    y = plot_df['val']
    
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
    
    #plot_df['var'] = var
    
    #  'var'
                                                                                    
    plot_df_concat = plot_df[['season', 'p', 'hi_sB1', 'lo_sB1', 'B1', 'B0', 'slope_datetime', 'slope_datetime_s_hi', 'slope_datetime_s_lo']].head(1) #slope_datetime_unc_cent, slope_datetime_s

    stats_df = pd.concat([stats_df, plot_df_concat])
    

stats_df.loc[stats_df['season'] == 'allyear', 'season_label'] = 'Full-Year*'

stats_df.loc[stats_df['season'] == 'grow', 'season_label'] = 'Apr-Jul'

stats_df.loc[stats_df['season'] == 'loDO', 'season_label'] = 'Aug-Nov*'

stats_df.loc[stats_df['season'] == 'winter', 'season_label'] = 'Dec-Mar'


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

for decade in decadal_yearday_mean['decade'].unique():
    
    temp = decadal_yearday_mean[(decadal_yearday_mean['decade'] == decade) & (decadal_yearday_mean['yearday'] == 1)]
    
    temp['yearday'] = 366
    
    decadal_yearday_mean = pd.concat([decadal_yearday_mean, temp])

# %%

fig, ax = plt.subplots(nrows=2, figsize = (9,7))

plt.subplots_adjust(hspace=0.3)



ax[0].plot(monthly_skagit_df['datetime'], monthly_skagit_df['val'], color= 'lightgray', label = 'Skagit Flow', alpha=0.7)

x_plot = monthly_skagit_df['datetime']

x = monthly_skagit_df['date_ordinal']
 
y = monthly_skagit_df['val']

for season in stats_df['season'].unique(): 
    
    plot_df = stats_df[stats_df['season'] == season]
    
    if season == 'allyear':
        
        color = 'black'
        
    elif season == 'grow':
        
        color = '#dd9404'
        
    elif season == 'loDO':
        
        color = '#e04256'
        
    elif season == 'winter':
        
        color = '#4565e8'
    
    
    
    if plot_df['p'].iloc[0] < alpha:
        
        linestyle = '-'
        
        label = plot_df['season_label'].iloc[0] #+ ' Trend\n[' + "{:.1f}".format(plot_df['slope_datetime'].iloc[0]*100) + r' $m^3/s$]'
        
    else:
        
        linestyle = '--' 
        
        label = plot_df['season_label'].iloc[0] #+ ' Trend\n[not significant]'

    
    ax[0].plot([x_plot.min(), x_plot.max()], [plot_df['B0'].iloc[0] + plot_df['B1'].iloc[0]*x.min(), plot_df['B0'].iloc[0] + plot_df['B1'].iloc[0]*x.max()], color = color, linestyle=linestyle, linewidth=2, label=label)



  
sns.lineplot(data = decadal_yearday_mean, x='yearday', y = 'val', hue='decade', ax=ax[1], palette='plasma_r')

ax[1].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')

ax[1].set_xlim([1,365])

ax[1].set_ylim(0,1800)

ax[1].axvline(90, color = 'gray', linestyle = '--')
 
ax[1].axvline(212, color = 'gray', linestyle = '--')

ax[1].axvline(334, color = 'gray', linestyle = '--')
 
ax[1].text(30,1350, 'Dec-Mar', horizontalalignment='center', verticalalignment='center', color='gray', fontweight = 'bold')

ax[1].text(151,1350, 'Apr-Jul', horizontalalignment='center', verticalalignment='center', color='gray', fontweight = 'bold')

ax[1].text(273,1350, 'Aug-Nov', horizontalalignment='center', verticalalignment='center', color='gray', fontweight = 'bold')
 


 

ax[0].text(0.025,0.05, 'a', transform=ax[0].transAxes, verticalalignment='bottom', fontweight = 'bold', color='k')

ax[1].text(0.025,0.05, 'b', transform=ax[1].transAxes, verticalalignment='bottom', fontweight = 'bold', color='k')


  



ax[0].set_ylabel(r'Discharge [$m^3/s$]')

ax[1].set_ylabel(r'Discharge [$m^3/s$]')
 

ax[0].set_xlabel('Year') 

ax[1].set_xlabel('Yearday') 

ax[0].set_ylim(0,1800) 
 


 
ax[1].grid(color = 'lightgray', linestyle = '--', alpha=0.5) 

ax[0].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
  
ax[0].legend(loc='upper center', ncol = 5)  
 
#ax[1].legend(loc='upper center', ncol = 9)

h, l = ax[1].get_legend_handles_labels() 

h.insert(0, plt.Line2D([0], [0], color='white', lw=0, marker='o', markersize=0))  # Empty entry

#h.insert(len(h), plt.Line2D([0], [0], color='white', lw=0, marker='o', markersize=0))  # Empty entry

#l.insert(0, "1940s")  # Title as an entry

ax[1].legend(h, ['1940s','','','','','','','','','     2020s      '], loc='upper center', ncol=11, handlelength=1)



#ax[0].legend(bbox_to_anchor=(1.05, 0.5), loc='upper left')

#ax[0].legend(bbox_to_anchor=(0.5, 0), loc='upper center')




#plt.tight_layout()


plt.savefig('/Users/dakotamascarenas/Desktop/pltz/paper_1_fig_4.png', bbox_inches='tight', dpi=500, transparent=True)