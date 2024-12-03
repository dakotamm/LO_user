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

poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson'] # 5 sites + 4 basins

#poly_list = ['ps']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)

# %%

site_list =  odf['site'].unique()




odf_use = odf_depth_mean.copy()

odf_calc_use = odf_calc_long.copy()

all_stats_filt = dfun.buildStatsDF(odf_use, site_list, odf_calc_use=odf_calc_use, odf_depth_mean_deep_DO_percentiles=odf_depth_mean_deep_DO_percentiles)



# %%

c=0

all_stats_filt.loc[all_stats_filt['site'] == 'point_jefferson', 'site_label'] = 'PJ'

all_stats_filt.loc[all_stats_filt['site'] == 'near_seattle_offshore', 'site_label'] = 'NS'

all_stats_filt.loc[all_stats_filt['site'] == 'saratoga_passage_mid', 'site_label'] = 'SP'

all_stats_filt.loc[all_stats_filt['site'] == 'carr_inlet_mid', 'site_label'] = 'CI'

all_stats_filt.loc[all_stats_filt['site'] == 'lynch_cove_mid', 'site_label'] = 'LC'


all_stats_filt.loc[all_stats_filt['site'] == 'point_jefferson', 'site_type'] = 'Main Basin'

all_stats_filt.loc[all_stats_filt['site'] == 'near_seattle_offshore', 'site_type'] = 'Main Basin'

all_stats_filt.loc[all_stats_filt['site'] == 'saratoga_passage_mid', 'site_type'] = 'Sub-Basins'

all_stats_filt.loc[all_stats_filt['site'] == 'carr_inlet_mid', 'site_type'] = 'Sub-Basins'

all_stats_filt.loc[all_stats_filt['site'] == 'lynch_cove_mid', 'site_type'] = 'Sub-Basins'


all_stats_filt.loc[all_stats_filt['site'] == 'point_jefferson', 'site_num'] = 1

all_stats_filt.loc[all_stats_filt['site'] == 'near_seattle_offshore', 'site_num'] = 2

all_stats_filt.loc[all_stats_filt['site'] == 'saratoga_passage_mid', 'site_num'] = 3

all_stats_filt.loc[all_stats_filt['site'] == 'carr_inlet_mid', 'site_num'] = 4

all_stats_filt.loc[all_stats_filt['site'] == 'lynch_cove_mid', 'site_num'] = 5



# %%

temp_df = pd.read_csv('/Users/dakotamascarenas/Desktop/ps_longterm/ptools_data/ncdc/1527887.csv')


temp_df['datetime'] = pd.to_datetime(temp_df['DATE'])

temp_df['year'] = pd.DatetimeIndex(temp_df['datetime']).year


temp_df['month'] = pd.DatetimeIndex(temp_df['datetime']).month


temp_df['year_month'] = temp_df['year'].astype(str) + '_' + temp_df['month'].astype(str).apply(lambda x: x.zfill(2))

temp_df['date_ordinal'] = temp_df['datetime'].apply(lambda x: x.toordinal())

temp_df.loc[temp_df['month'].isin([4,5,6,7]), 'season'] = 'grow'

temp_df.loc[temp_df['month'].isin([8,9,10,11]), 'season'] = 'loDO'

temp_df.loc[temp_df['month'].isin([12,1,2,3]), 'season'] = 'winter'


temp_df['TAVG'] = temp_df[['TMAX', 'TMIN']].mean(axis=1)

temp_df['year_season'] = temp_df['year'].astype(str) + '_' + temp_df['season']


temp_df = pd.melt(temp_df, id_vars =['STATION', 'NAME', 'DATE', 'datetime', 'year', 'month', 'year_month', 'date_ordinal', 'season', 'year_season'], value_vars=['PRCP', 'TMAX', 'TMIN', 'TSUN', 'TAVG'], var_name='var', value_name='val')

# %%

monthly_counts = (temp_df
                      .dropna()
                      .groupby(['year_month', 'var']).agg({'val' :lambda x: x.nunique()})
                      .reset_index()
                      .rename(columns={'val':'val_count'})
                      )

temp_monthly_avg_df = temp_df[['datetime', 'date_ordinal', 'year', 'month', 'year_month', 'season', 'year_season', 'var', 'val']].groupby(['year','month','year_month','season', 'year_season', 'var']).agg({'val':['mean', 'std'], 'date_ordinal':['mean']})

temp_monthly_avg_df.columns = temp_monthly_avg_df.columns.to_flat_index().map('_'.join)

temp_monthly_avg_df = temp_monthly_avg_df.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!


temp_monthly_avg_df = (temp_monthly_avg_df
                  .rename(columns={'date_ordinal_mean':'date_ordinal'})
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )


temp_monthly_avg_df = pd.merge(temp_monthly_avg_df, monthly_counts, how='left', on=['year_month', 'var'])


temp_monthly_avg_df = temp_monthly_avg_df[temp_monthly_avg_df['val_count'] >1] #redundant but fine (see note line 234)

temp_monthly_avg_df['val_ci95hi'] = temp_monthly_avg_df['val_mean'] + 1.96*temp_monthly_avg_df['val_std']/np.sqrt(temp_monthly_avg_df['val_count'])

temp_monthly_avg_df['val_ci95lo'] = temp_monthly_avg_df['val_mean'] - 1.96*temp_monthly_avg_df['val_std']/np.sqrt(temp_monthly_avg_df['val_count'])

temp_monthly_avg_df['val'] = temp_monthly_avg_df['val_mean']

# %%

seasonal_counts = (temp_df
                      .dropna()
                      .groupby(['year_season', 'var']).agg({'val' :lambda x: x.nunique()})
                      .reset_index()
                      .rename(columns={'val':'val_count'})
                      )

temp_seasonal_avg_df = temp_df[['datetime', 'date_ordinal', 'year', 'season', 'year_season', 'var','val']].groupby(['year','season', 'year_season','var']).agg({'val':['mean', 'std'], 'date_ordinal':['mean']})

temp_seasonal_avg_df.columns = temp_seasonal_avg_df.columns.to_flat_index().map('_'.join)

temp_seasonal_avg_df = temp_seasonal_avg_df.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!


temp_seasonal_avg_df = (temp_seasonal_avg_df
                  .rename(columns={'date_ordinal_mean':'date_ordinal'})
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )


temp_seasonal_avg_df = pd.merge(temp_seasonal_avg_df, seasonal_counts, how='left', on=['year_season', 'var'])


temp_seasonal_avg_df = temp_seasonal_avg_df[temp_seasonal_avg_df['val_count'] >1] #redundant but fine (see note line 234)

temp_seasonal_avg_df['val_ci95hi'] = temp_seasonal_avg_df['val_mean'] + 1.96*temp_seasonal_avg_df['val_std']/np.sqrt(temp_seasonal_avg_df['val_count'])

temp_seasonal_avg_df['val_ci95lo'] = temp_seasonal_avg_df['val_mean'] - 1.96*temp_seasonal_avg_df['val_std']/np.sqrt(temp_seasonal_avg_df['val_count'])

temp_seasonal_avg_df['val'] = temp_seasonal_avg_df['val_mean']

# %%

annual_counts = (temp_df
                      .dropna()
                      .groupby(['year', 'var']).agg({'val' :lambda x: x.nunique()})
                      .reset_index()
                      .rename(columns={'val':'val_count'})
                      )

temp_annual_avg_df = temp_df[['datetime', 'date_ordinal', 'year','var','val']].groupby(['year', 'var']).agg({'val':['mean', 'std'], 'date_ordinal':['mean']})

temp_annual_avg_df.columns = temp_annual_avg_df.columns.to_flat_index().map('_'.join)

temp_annual_avg_df = temp_annual_avg_df.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!


temp_annual_avg_df = (temp_annual_avg_df
                  .rename(columns={'date_ordinal_mean':'date_ordinal'})
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )


temp_annual_avg_df = pd.merge(temp_annual_avg_df, annual_counts, how='left', on=['year', 'var'])


temp_annual_avg_df = temp_annual_avg_df[temp_annual_avg_df['val_count'] >1] #redundant but fine (see note line 234)

temp_annual_avg_df['val_ci95hi'] = temp_annual_avg_df['val_mean'] + 1.96*temp_annual_avg_df['val_std']/np.sqrt(temp_annual_avg_df['val_count'])

temp_annual_avg_df['val_ci95lo'] = temp_annual_avg_df['val_mean'] - 1.96*temp_annual_avg_df['val_std']/np.sqrt(temp_annual_avg_df['val_count'])

temp_annual_avg_df['val'] = temp_annual_avg_df['val_mean']


# %%

# odf_depth_mean = (odf_depth_mean
#                   #.rename(columns={'date_ordinal_mean':'date_ordinal'})
#                   .dropna()
#                   .assign(
#                           datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
#                           )
#                   )

#odf_depth_mean['year_month'] = odf_depth_mean['year'].astype(str) + '_' + odf_depth_mean['month'].astype(str).apply(lambda x: x.zfill(2))


# odf_depth_mean_monthly_avg_df = odf_depth_mean.groupby(['site', 'year_month', 'summer_non_summer','surf_deep', 'var']).mean().reset_index().dropna()


# %%

odf_annual_use, odf_annual_long_use = dfun.annualDepthAverageDF(odf_depth_mean, odf_calc_long)

# %%

# using all monthly averages

# using grow, loDO, and winter averages

stats_df = pd.DataFrame()

stat = 'mk_ts'

alpha = 0.05


    
for season in ['allyear', 'grow', 'loDO', 'winter']:
            
    if season == 'allyear':
                
        plot_df = temp_monthly_avg_df.copy()
        
    else:
        
        plot_df = temp_monthly_avg_df[temp_monthly_avg_df['season'] == season].copy()

    for var in ['TAVG']:
        
        x = plot_df[plot_df['var'] == var]['date_ordinal']
        
        x_plot = plot_df[plot_df['var'] == var]['datetime']
        
        y = plot_df[plot_df['var'] == var]['val']
        
        plot_df['stat'] = stat
                
        # reject_null, p_value, Z = dfun.mann_kendall(y, alpha) #dfun
                            
        #plot_df['p'] = p_value
        
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
        
        # 'p' for below
                                                                                        
        plot_df_concat = plot_df[['season','var', 'hi_sB1', 'lo_sB1', 'B1', 'B0', 'slope_datetime', 'slope_datetime_s_hi', 'slope_datetime_s_lo']].head(1) #slope_datetime_unc_cent, slope_datetime_s
    
        stats_df = pd.concat([stats_df, plot_df_concat])
        
# %%
# %%


mosaic = [['time_series', 'time_series', 'trends']]

fig, axd = plt.subplot_mosaic(mosaic, figsize=(10,4), layout='constrained', gridspec_kw=dict(wspace=0.1))

ax = axd['time_series']



plot_df = temp_annual_avg_df[temp_annual_avg_df['var'] == 'TAVG']

for idx in plot_df.index:

    ax.plot([plot_df.loc[idx,'datetime'], plot_df.loc[idx,'datetime']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color='gray', alpha =0.3, linewidth=1,zorder=-5)

sns.scatterplot(data=plot_df, x='datetime', y='val', ax=ax, alpha=0.5, color = 'gray', label='Sea-Tac Annual Average')



plot_df = temp_seasonal_avg_df[(temp_seasonal_avg_df['season'] == 'loDO') & (temp_seasonal_avg_df['var'] == 'TAVG')]

for idx in plot_df.index:

    ax.plot([plot_df.loc[idx,'datetime'], plot_df.loc[idx,'datetime']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color='gray', alpha =0.3, linewidth=1, zorder=-5)

sns.scatterplot(data=plot_df, x='datetime', y='val', ax=ax, alpha=0.5, color = 'black', label='Sea-Tac Annual August-November Average')
 


plot_df_surf_CT = odf_annual_use[(odf_annual_use['site'] == 'point_jefferson') & (odf_annual_use['surf_deep'] == 'surf') & (odf_annual_use['var'] == 'CT') & (odf_annual_use['summer_non_summer'] == 'summer')]

for idx in plot_df_surf_CT.index:

    ax.plot([plot_df_surf_CT.loc[idx,'datetime'], plot_df_surf_CT.loc[idx,'datetime']],[plot_df_surf_CT.loc[idx,'val_ci95lo'], plot_df_surf_CT.loc[idx,'val_ci95hi']], color='#4565e8', linewidth=1, zorder=-5, alpha=0.5)

sns.scatterplot(data=plot_df_surf_CT, x='datetime', y='val', ax=ax, color = '#4565e8', label = 'PJ Surface Annual August-November Average')



plot_df_deep_CT = odf_annual_use[(odf_annual_use['site'] == 'point_jefferson') & (odf_annual_use['surf_deep'] == 'deep') & (odf_annual_use['var'] == 'CT') & (odf_annual_use['summer_non_summer'] == 'summer')]

for idx in plot_df_deep_CT.index:

    ax.plot([plot_df_deep_CT.loc[idx,'datetime'], plot_df_deep_CT.loc[idx,'datetime']],[plot_df_deep_CT.loc[idx,'val_ci95lo'], plot_df_deep_CT.loc[idx,'val_ci95hi']], color='#e04256', linewidth=1, zorder=-5, alpha=0.5)

sns.scatterplot(data=plot_df_deep_CT, x='datetime', y='val', ax=ax, color = '#e04256', label = 'PJ Deep Annual August-November Average')



ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3, zorder=-5)

ax.legend(loc = 'upper left')

#ax.set_ylim(0,18)

ax.set_xlabel('')

ax.set_ylabel(r'Temperature [$^{\circ}$C]', wrap=True)

ax.set_ylim(7,20)

ax.text(0.05,0.05, 'a', transform=ax.transAxes, verticalalignment='bottom', fontweight = 'bold', color='k')



ax = axd['trends']

P4_trend = 0.0084*100 #degC per century

plot_df = stats_df.copy()

plot_df.loc[plot_df['season'] == 'loDO','season_label'] = 'Aug-Nov'

plot_df.loc[plot_df['season'] == 'loDO','season_number'] = 3


plot_df.loc[plot_df['season'] == 'grow','season_label'] = 'Apr-Jul'

plot_df.loc[plot_df['season'] == 'grow','season_number'] = 2


plot_df.loc[plot_df['season'] == 'winter','season_label'] = 'Dec-Mar'

plot_df.loc[plot_df['season'] == 'winter','season_number'] = 1


plot_df.loc[plot_df['season'] == 'allyear','season_label'] = 'Full-Year'

plot_df.loc[plot_df['season'] == 'allyear','season_number'] = 0





plot_df['slope_datetime_cent_95hi'] = plot_df['slope_datetime_s_hi']*100

plot_df['slope_datetime_cent_95lo'] = plot_df['slope_datetime_s_lo']*100

plot_df['slope_datetime_cent'] = plot_df['slope_datetime']*100


 

sns.scatterplot(data = plot_df, x= 'season_number', y = 'slope_datetime_cent_95hi', color = 'gray', ax = ax, s= 10, legend = False)

sns.scatterplot(data = plot_df, x= 'season_number', y = 'slope_datetime_cent_95lo', color = 'gray', ax = ax, s= 10, legend=False)

sns.scatterplot(data = plot_df, x= 'season_number', y = 'slope_datetime_cent', color = 'gray', ax = ax, s =50, label = 'Sea-Tac Monthly (Seasonal)')

for idx in plot_df.index:        
        
    ax.plot([plot_df.loc[idx,'season_number'], plot_df.loc[idx,'season_number']],[plot_df.loc[idx,'slope_datetime_cent_95lo'], plot_df.loc[idx,'slope_datetime_cent_95hi']], color='gray', alpha =0.7, zorder = -4, linewidth=1)


pj_surf = all_stats_filt[(all_stats_filt['site'] == 'point_jefferson') & (all_stats_filt['summer_non_summer'] =='summer') & (all_stats_filt['var'] == 'surf_CT')]['slope_datetime'].iloc[0]*100
 
pj_deep = all_stats_filt[(all_stats_filt['site'] == 'point_jefferson') & (all_stats_filt['summer_non_summer'] =='summer') & (all_stats_filt['var'] == 'deep_CT')]['slope_datetime'].iloc[0]*100



ax.axhline(pj_surf, color='#4565e8', label = 'PJ Surface August-November')

ax.axhline(pj_deep, color = '#e04256', label = 'PJ Deep August-November')


ax.axhline(P4_trend, color = 'black', linestyle = '--', label = 'Whitney et al. (2007) at P4')


ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3) 
 
#ax.axhline(0, color='gray', linestyle = '--', zorder = -5)


#ax.tick_params(axis='x', rotation=45)

ax.set_ylim(0,7)

ax.set_xticks([0,1,2,3],['Full-Year', 'Dec-Mar', 'Apr-Jul', 'Aug-Nov'])


ax.set_ylabel(r'[$^{\circ}$C]/century', wrap=True)

ax.set_xlabel('')

ax.legend(loc = 'upper left')

ax.text(0.05,0.05, 'b', transform=ax.transAxes, verticalalignment='bottom', fontweight = 'bold', color='k')




 
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/paper_1_fig_3.png', dpi=500,transparent=True, bbox_inches='tight')   



    