#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 11:07:08 2025

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

from pygam import LinearGAM, s

import matplotlib.patheffects as pe

# %%




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


# %%

monthly_skagit_working = monthly_skagit_df.copy()

monthly_skagit_working['year_fudge'] = monthly_skagit_working['year_nu'] # this is to make seasons chronological because winter overlaps calendar years!

monthly_skagit_working.loc[monthly_skagit_working['month_nu'] == 12, 'year_fudge'] = monthly_skagit_working['year_nu'] + 1

annual_seasonal_counts_skagit = (monthly_skagit_working
                      .dropna()
                      .groupby(['year_fudge', 'season']).agg({'month_nu' :lambda x: x.nunique()})
                      .reset_index()
                      .rename(columns={'month_nu':'month_count'})
                      )


annual_seasonal_skagit_means = monthly_skagit_working.groupby(['year_fudge', 'season']).agg({'mean_va':['mean', 'std'], 'date_ordinal':['mean']})

annual_seasonal_skagit_means.columns = annual_seasonal_skagit_means.columns.to_flat_index().map('_'.join)

annual_seasonal_skagit_means = annual_seasonal_skagit_means.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!


annual_seasonal_skagit_means = (annual_seasonal_skagit_means
                  .rename(columns={'date_ordinal_mean':'date_ordinal'})
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )


annual_seasonal_skagit_means = pd.merge(annual_seasonal_skagit_means, annual_seasonal_counts_skagit, how='left', on=['year_fudge','season'])


annual_seasonal_skagit_means = annual_seasonal_skagit_means[annual_seasonal_skagit_means['month_count'] >1] #redundant but fine (see note line 234)

annual_seasonal_skagit_means['mean_va_ci95hi'] = annual_seasonal_skagit_means['mean_va_mean'] + 1.96*annual_seasonal_skagit_means['mean_va_std']/np.sqrt(annual_seasonal_skagit_means['month_count'])

annual_seasonal_skagit_means['mean_va_ci95lo'] = annual_seasonal_skagit_means['mean_va_mean'] - 1.96*annual_seasonal_skagit_means['mean_va_std']/np.sqrt(annual_seasonal_skagit_means['month_count'])


annual_seasonal_skagit_means['year'] = annual_seasonal_skagit_means['year_fudge']

annual_seasonal_skagit_means = annual_seasonal_skagit_means[['year', 'season', 'mean_va_mean']].pivot(index='year', columns = 'season', values = 'mean_va_mean').reset_index()    

# %%

monthly_skagit_working = monthly_skagit_df.copy()

monthly_skagit_working['year_fudge'] = monthly_skagit_working['year_nu'] # this is to make seasons chronological because winter overlaps calendar years!

monthly_skagit_working.loc[monthly_skagit_working['month_nu'] == 12, 'year_fudge'] = monthly_skagit_working['year_nu'] + 1

annual_counts_skagit = (monthly_skagit_working
                      .dropna()
                      .groupby(['year_fudge']).agg({'month_nu' :lambda x: x.nunique()})
                      .reset_index()
                      .rename(columns={'month_nu':'month_count'})
                      )


annual_skagit_means = monthly_skagit_working.groupby(['year_fudge']).agg({'mean_va':['mean', 'std'], 'date_ordinal':['mean']})

annual_skagit_means.columns = annual_skagit_means.columns.to_flat_index().map('_'.join)

annual_skagit_means = annual_skagit_means.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!


annual_skagit_means = (annual_skagit_means
                  .rename(columns={'date_ordinal_mean':'date_ordinal'})
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )


annual_skagit_means = pd.merge(annual_skagit_means, annual_counts_skagit, how='left', on=['year_fudge'])


annual_skagit_means = annual_skagit_means[annual_skagit_means['month_count'] >1] #redundant but fine (see note line 234)

annual_skagit_means['mean_va_ci95hi'] = annual_skagit_means['mean_va_mean'] + 1.96*annual_skagit_means['mean_va_std']/np.sqrt(annual_skagit_means['month_count'])

annual_skagit_means['mean_va_ci95lo'] = annual_skagit_means['mean_va_mean'] - 1.96*annual_skagit_means['mean_va_std']/np.sqrt(annual_skagit_means['month_count'])


annual_skagit_means['year'] = annual_skagit_means['year_fudge']

annual_skagit_means['allyear'] = annual_skagit_means['mean_va_mean']


# %%

odf_use, odf_calc_use = dfun.seasonalDepthAverageDF(odf_depth_mean, odf_calc_long) #don't worry about filtering median because only using SA

# %%

odf_use_SA = odf_use[odf_use['var'] == 'SA']

odf_use_SA = pd.merge(odf_use_SA, annual_seasonal_skagit_means[['year', 'grow', 'loDO', 'winter']], how='left', on = ['year'])

odf_use_SA = pd.merge(odf_use_SA, annual_skagit_means[['year', 'allyear']], how='left', on = ['year'])


# %%

annual_seasonal_skagit_means_fudge = annual_seasonal_skagit_means.copy()

annual_seasonal_skagit_means_fudge['year'] = annual_seasonal_skagit_means_fudge['year'] + 1

annual_seasonal_skagit_means_fudge['grow_fudge'] = annual_seasonal_skagit_means_fudge['grow']

annual_seasonal_skagit_means_fudge['loDO_fudge'] = annual_seasonal_skagit_means_fudge['loDO']

odf_use_SA = pd.merge(odf_use_SA, annual_seasonal_skagit_means_fudge[['year', 'grow_fudge', 'loDO_fudge']], how='left', on = ['year'])

# %%

skagit_means = pd.merge(annual_skagit_means[['year', 'allyear']], annual_seasonal_skagit_means[['year', 'grow', 'loDO', 'winter']], how='left', on=['year'])
# %%

max_months = monthly_skagit_working.loc[monthly_skagit_working.groupby(['year_fudge'])['val'].idxmax()] #shift december to next year...

max_months['year'] = max_months['year_fudge']

max_months['max_month_fudge'] = max_months['month_nu']

odf_use_SA = pd.merge(odf_use_SA, max_months[['year', 'max_month_fudge']], how='left', on = ['year'])

# %%


odf_use_SA['grow_less_loDO'] = odf_use_SA['grow'] - odf_use_SA['loDO']

odf_use_SA['winter_less_grow'] = odf_use_SA['winter'] - odf_use_SA['grow']

odf_use_SA['loDO_fudge_less_winter'] = odf_use_SA['loDO_fudge'] - odf_use_SA['winter']


# %%

good_sites = ['SAR003','KSBP01', 'HCB007', 'CRR001', 'ADM003', 'carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'point_jefferson', 'saratoga_passage_mid']

for site in good_sites:      
    
    mosaic = [['surf_grow', 'deep_grow'], ['surf_loDO', 'deep_loDO'], ['surf_winter', 'deep_winter']]
        
    fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=False, layout='constrained', figsize=(10,10), gridspec_kw=dict(wspace=0.1, hspace=0.1))

    for depth in ['surf', 'deep']:
                    
        for season in ['grow', 'loDO', 'winter']:
            
            # if season == 'grow':
                
            #     season_skagit = 'allyear'
                
            # elif season == 'loDO':
                
            #     season_skagit = 'allyear'
            
            # elif season == 'winter':
                
            #     season_skagit = 'allyear' 
                
            ax_name = depth + '_' + season
            
            ax = axd[ax_name]
            
            plot_df = odf_use_SA[(odf_use_SA['site'] == site) & (odf_use_SA['season'] == season) & (odf_use_SA['surf_deep'] == depth)].dropna()
                        
            x = plot_df['max_month_fudge']
            
            y = plot_df['val']
            
            results = stats.linregress(x,y)
            
            ax.scatter(x, y)
            
            x_reg = np.linspace(x.min(), x.max(), 2) # make two x coordinates from min and max values of SLI_max
            
            y_reg = results.intercept + results.slope*x_reg
            
            ax.text(0.8, 0.9, "r^2 = {:.2f}".format(results.rvalue**2), transform=ax.transAxes)
            
            ax.plot(x_reg, y_reg, '-r')
                    
            ax.set_ylabel(depth + ' ' + season + ' SA [g/kg]')
            
            ax.set_xlabel('max flow month')
            
            ax.set_title('SA= ' + depth + '_' + season)
                                    
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
    
    fig.suptitle('annual max flow month vs ' + site + ' SA')

    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' +site + '_annualmaxflowmonth_vs_SA_select.png', dpi=500)

# %%

mosaic = [['time_series'], ['max_months'], ['grow_less_loDO'], ['winter_less_grow'], ['loDO_fudge_less_winter']]

fig, axd = plt.subplot_mosaic(mosaic, layout='constrained', figsize=(6,10), gridspec_kw=dict(wspace=0.1, hspace=0.1))

ax = axd['time_series']

ax.plot(monthly_skagit_df['datetime'], monthly_skagit_df['val'])

ax.set_ylabel('mo. avg. discharge [m^3/s]') 


ax = axd['max_months']

ax.scatter(max_months['year'], max_months['max_month_fudge'])

ax.set_ylabel('max discharge month')


ax = axd['grow_less_loDO']

ax.scatter(odf_use_SA['year'], odf_use_SA['grow_less_loDO'])

ax.set_ylabel('grow-loDO [m^3/s]')


ax = axd['winter_less_grow']

ax.scatter(odf_use_SA['year'], odf_use_SA['winter_less_grow'])

ax.set_ylabel('winter-grow [m^3/s]')


ax = axd['loDO_fudge_less_winter']

ax.scatter(odf_use_SA['year'], odf_use_SA['loDO_fudge_less_winter'])

ax.set_ylabel('loDO(last year)-winter [m^3/s]')



plt.savefig('/Users/dakotamascarenas/Desktop/pltz/skagittimeseries_maxmonths_diffs.png', dpi=500)




# %%


good_sites = ['SAR003','KSBP01', 'HCB007', 'CRR001', 'ADM003', 'carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'point_jefferson', 'saratoga_passage_mid']

for site in good_sites:      
    
    mosaic = [['surf_grow', 'deep_grow'], ['surf_loDO', 'deep_loDO'], ['surf_winter', 'deep_winter']]
    
    fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=False, layout='constrained', figsize=(10,10), gridspec_kw=dict(wspace=0.1, hspace=0.1))

    for depth in ['surf', 'deep']:
                    
        for season in ['grow', 'loDO', 'winter']:
            
            # if season == 'grow':
                
            #     season_skagit = 'allyear'
                
            # elif season == 'loDO':
                
            #     season_skagit = 'allyear'
            
            # elif season == 'winter':
                
            #     season_skagit = 'allyear' 
                
            ax_name = depth + '_' + season
            
            ax = axd[ax_name]
            
            plot_df = odf_use_SA[(odf_use_SA['site'] == site) & (odf_use_SA['season'] == season) & (odf_use_SA['surf_deep'] == depth)].dropna()
                        
            x = plot_df['grow_less_loDO']
            
            y = plot_df['val']
            
            results = stats.linregress(x,y)
            
            ax.scatter(x, y)
            
            x_reg = np.linspace(x.min(), x.max(), 2) # make two x coordinates from min and max values of SLI_max
            
            y_reg = results.intercept + results.slope*x_reg
            
            ax.text(0.8, 0.9, "r^2 = {:.2f}".format(results.rvalue**2), transform=ax.transAxes)
            
            ax.plot(x_reg, y_reg, '-r')
                    
            ax.set_ylabel(depth + ' ' + season + ' SA [g/kg]')
            
            ax.set_xlabel('grow-loDO diff. Skagit flow [m^3/s]')
            
            ax.set_title('SA= ' + depth + '_' + season)
                                    
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
    
    fig.suptitle('grow less loDO diff vs ' + site + ' SA')

    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' +site + '_growlessloDOdiff_vs_SA_select.png', dpi=500)

# %%


good_sites = ['SAR003','KSBP01', 'HCB007', 'CRR001', 'ADM003', 'carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'point_jefferson', 'saratoga_passage_mid']

for site in good_sites:      
    
    mosaic = [['surf_grow', 'deep_grow'], ['surf_loDO', 'deep_loDO'], ['surf_winter', 'deep_winter']]
    
    fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=False, layout='constrained', figsize=(10,10), gridspec_kw=dict(wspace=0.1, hspace=0.1))

    for depth in ['surf', 'deep']:
                    
        for season in ['grow', 'loDO', 'winter']:
            
            # if season == 'grow':
                
            #     season_skagit = 'allyear'
                
            # elif season == 'loDO':
                
            #     season_skagit = 'allyear'
            
            # elif season == 'winter':
                
            #     season_skagit = 'allyear' 
                
            ax_name = depth + '_' + season
            
            ax = axd[ax_name]
            
            plot_df = odf_use_SA[(odf_use_SA['site'] == site) & (odf_use_SA['season'] == season) & (odf_use_SA['surf_deep'] == depth)].dropna()
                        
            x = plot_df['winter_less_grow']
            
            y = plot_df['val']
            
            results = stats.linregress(x,y)
            
            ax.scatter(x, y)
            
            x_reg = np.linspace(x.min(), x.max(), 2) # make two x coordinates from min and max values of SLI_max
            
            y_reg = results.intercept + results.slope*x_reg
            
            ax.text(0.8, 0.9, "r^2 = {:.2f}".format(results.rvalue**2), transform=ax.transAxes)
            
            ax.plot(x_reg, y_reg, '-r')
                    
            ax.set_ylabel(depth + ' ' + season + ' SA [g/kg]')
            
            ax.set_xlabel('winter-grow diff. Skagit flow [m^3/s]')
            
            ax.set_title('SA= ' + depth + '_' + season)
                                    
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
    
    fig.suptitle('winter less grow diff vs ' + site + ' SA')

    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' +site + '_winterlessgrowdiff_vs_SA_select.png', dpi=500)

# %%


good_sites = ['SAR003','KSBP01', 'HCB007', 'CRR001', 'ADM003', 'carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'point_jefferson', 'saratoga_passage_mid']

for site in good_sites:      
    
    mosaic = [['surf_grow', 'deep_grow'], ['surf_loDO', 'deep_loDO'], ['surf_winter', 'deep_winter']]
    
    fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=False, layout='constrained', figsize=(10,10), gridspec_kw=dict(wspace=0.1, hspace=0.1))

    for depth in ['surf', 'deep']:
                    
        for season in ['grow', 'loDO', 'winter']:
            
            # if season == 'grow':
                
            #     season_skagit = 'allyear'
                
            # elif season == 'loDO':
                
            #     season_skagit = 'allyear'
            
            # elif season == 'winter':
                
            #     season_skagit = 'allyear' 
                
            ax_name = depth + '_' + season
            
            ax = axd[ax_name]
            
            plot_df = odf_use_SA[(odf_use_SA['site'] == site) & (odf_use_SA['season'] == season) & (odf_use_SA['surf_deep'] == depth)].dropna()
                        
            x = plot_df['loDO_fudge_less_winter']
            
            y = plot_df['val']
            
            results = stats.linregress(x,y)
            
            ax.scatter(x, y)
            
            x_reg = np.linspace(x.min(), x.max(), 2) # make two x coordinates from min and max values of SLI_max
            
            y_reg = results.intercept + results.slope*x_reg
            
            ax.text(0.8, 0.9, "r^2 = {:.2f}".format(results.rvalue**2), transform=ax.transAxes)
            
            ax.plot(x_reg, y_reg, '-r')
                    
            ax.set_ylabel(depth + ' ' + season + ' SA [g/kg]')
            
            ax.set_xlabel('loDO(lastyear)-winter diff. Skagit flow [m^3/s]')
            
            ax.set_title('SA= ' + depth + '_' + season)
                                    
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
    
    fig.suptitle('loDO (last year) less winter diff vs ' + site + ' SA')

    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' +site + '_loDOfudgelesswinterdiff_vs_SA_select.png', dpi=500)

# %%

fig, ax = plt.subplots()

sns.lineplot(data = monthly_skagit_df[monthly_skagit_df['year_nu'] >= 1999], x='yearday', y = 'val', hue='year_nu', palette='plasma_r') 

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/1999on_yearly_hydrograph.png', dpi=500)




            
        
            
            
            
            
            
            
            
            
            
        
    
    
    
    