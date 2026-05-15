#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 15:35:01 2025

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




poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson', 'mb', 'hc', 'ss', 'wb'] # 5 sites + 4 basins

#poly_list = ['ps']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)

# %%

all_stats_filt = dfun.buildStatsDF(odf_depth_mean, site_list, odf_calc_use=odf_calc_long, odf_depth_mean_deep_DO_percentiles=odf_depth_mean_deep_DO_percentiles)

# %%

monthly_skagit_df = pd.read_csv('/Users/dakotamascarenas/Desktop/skagit_monthly.txt',sep='\t',header=(35), skiprows=(36,36))




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

monthly_skagit_df['val'] = monthly_skagit_df['mean_va']


monthly_skagit_df = monthly_skagit_df.assign(
    datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x)))))

skagit_stats_df = pd.DataFrame()

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

    skagit_stats_df = pd.concat([skagit_stats_df, plot_df_concat])


# %%

annual_seasonal_counts_skagit = (monthly_skagit_df
                      .dropna()
                      .groupby(['year_nu','season']).agg({'month_nu' :lambda x: x.nunique()})
                      .reset_index()
                      .rename(columns={'month_nu':'month_count'})
                      )


annual_seasonal_skagit_means = monthly_skagit_df.groupby(['year_nu', 'season']).agg({'mean_va':['mean', 'std'], 'date_ordinal':['mean']})

annual_seasonal_skagit_means.columns = annual_seasonal_skagit_means.columns.to_flat_index().map('_'.join)

annual_seasonal_skagit_means = annual_seasonal_skagit_means.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!


annual_seasonal_skagit_means = (annual_seasonal_skagit_means
                  .rename(columns={'date_ordinal_mean':'date_ordinal'})
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )


annual_seasonal_skagit_means = pd.merge(annual_seasonal_skagit_means, annual_seasonal_counts_skagit, how='left', on=['year_nu','season'])


annual_seasonal_skagit_means = annual_seasonal_skagit_means[annual_seasonal_skagit_means['month_count'] >1] #redundant but fine (see note line 234)

annual_seasonal_skagit_means['mean_va_ci95hi'] = annual_seasonal_skagit_means['mean_va_mean'] + 1.96*annual_seasonal_skagit_means['mean_va_std']/np.sqrt(annual_seasonal_skagit_means['month_count'])

annual_seasonal_skagit_means['mean_va_ci95lo'] = annual_seasonal_skagit_means['mean_va_mean'] - 1.96*annual_seasonal_skagit_means['mean_va_std']/np.sqrt(annual_seasonal_skagit_means['month_count'])


annual_seasonal_skagit_means['year'] = annual_seasonal_skagit_means['year_nu']

annual_seasonal_skagit_means = annual_seasonal_skagit_means[['year', 'season', 'mean_va_mean']].pivot(index='year', columns = 'season', values = 'mean_va_mean').reset_index()    



# %%

odf_use_SA = odf_depth_mean[odf_depth_mean['var'] == 'SA']

odf_use_SA = pd.merge(odf_use_SA, annual_seasonal_skagit_means[['year', 'grow', 'loDO', 'winter']], how='left', on = ['year'])


# %%

for season in ['grow', 'loDO', 'winter']:
    
    fig, ax = plt.subplots()
    
    plot_df = annual_seasonal_skagit_means.copy()
    
    plot_df['rank'] = plot_df[season].rank()
    
    sns.scatterplot(data = plot_df, x='rank', y=season, hue='year', palette='plasma_r')
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + season + '_ranked_skagit.png', dpi=500)
    
# %%

for season in ['grow', 'loDO', 'winter']:
    
    fig, ax = plt.subplots()
    
    plot_df = annual_seasonal_skagit_means.copy()
    
    plot_df = plot_df[plot_df['year'] >= 1998]
    
    plot_df['rank'] = plot_df[season].rank()
    
    sns.scatterplot(data = plot_df, x='rank', y=season, hue='year', palette='plasma_r')
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + season + '_ranked_skagit_shortonly.png', dpi=500)
    
# %%

for site in site_list:
    
    for season_SA in ['grow', 'loDO', 'winter']:
    
        mosaic = [['surf_grow', 'deep_grow'], ['surf_loDO', 'deep_loDO'], ['surf_winter', 'deep_winter']]
    
        fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=False, layout='constrained', figsize=(10,10))

        for depth in ['surf', 'deep']:
                    
            for season_skagit in ['grow', 'loDO', 'winter']:
                
                ax_name = depth + '_' + season_skagit
                
                ax = axd[ax_name]
                
                plot_df = odf_use_SA[(odf_use_SA['site'] == site) & (odf_use_SA['season'] == season_SA) & (odf_use_SA['surf_deep'] == depth)]
                
                if site in long_site_list:
                
                    plot_df_skagit = annual_seasonal_skagit_means[['year', season_skagit]]
                    
                else:
                    
                    plot_df_skagit = annual_seasonal_skagit_means[annual_seasonal_skagit_means['year'] >=1998][['year', season_skagit]]
                                                              
                plot_df_skagit['rank'] = plot_df_skagit[season_skagit].rank()
                
                plot_df = pd.merge(plot_df, plot_df_skagit, how='left', on = ['year'])

                sns.scatterplot(data = plot_df, x='rank', y='val', hue='year', palette='plasma_r', ax=ax, legend = False)
            
                ax.set_title(ax_name)
                
                ax.set_ylabel('SA [g/kg]')
                
                ax.set_xlabel('Skagit Seasonal Flow Rank')
                
                ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)

                
        fig.suptitle(season_SA + '_SA')
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' +site + '_' + season_SA + '_SA_vs_ranked_skagit.png', dpi=500)
        
# %%

for site in site_list:
    
    for season_SA in ['grow', 'loDO', 'winter']:
    
        mosaic = [['surf_grow', 'deep_grow'], ['surf_loDO', 'deep_loDO'], ['surf_winter', 'deep_winter']]
    
        fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=False, layout='constrained', figsize=(10,10))

        for depth in ['surf', 'deep']:
                    
            for season_skagit in ['grow', 'loDO', 'winter']:
                
                ax_name = depth + '_' + season_skagit
                
                ax = axd[ax_name]
                
                plot_df = odf_use_SA[(odf_use_SA['site'] == site) & (odf_use_SA['season'] == season_SA) & (odf_use_SA['surf_deep'] == depth)]
                                                              
                sns.scatterplot(data = plot_df, x=season_skagit, y='val', hue='year', palette='plasma_r', ax=ax, legend = False)
            
                ax.set_title(ax_name)
                
                ax.set_ylabel('SA [g/kg]')
                
                ax.set_xlabel('Skagit Seasonal Average [m^3/s]')
                
                ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
                
        fig.suptitle(season_SA + '_SA')
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' +site + '_' + season_SA + '_SA_vs_seasonalavg_skagit.png', dpi=500)


        
                
                
        
        
