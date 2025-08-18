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
                      .groupby(['year_fudge']).agg({'month_nu' :lambda x: x.nunique()})
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

good_sites = ['SAR003','KSBP01', 'HCB007', 'CRR001', 'ADM003', 'carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'point_jefferson', 'saratoga_passage_mid']

for site in good_sites:      
    
    mosaic = [['surf_grow', 'deep_grow'], ['surf_loDO', 'deep_loDO'], ['surf_winter', 'deep_winter']]
    
    fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=False, layout='constrained', figsize=(10,10), gridspec_kw=dict(wspace=0.1, hspace=0.1))

    for depth in ['surf', 'deep']:
                    
        for season in ['grow', 'loDO', 'winter']:
            
            if season == 'grow':
                
                season_skagit = 'allyear'
                
            elif season == 'loDO':
                
                season_skagit = 'allyear'
            
            elif season == 'winter':
                
                season_skagit = 'allyear' 
                
            ax_name = depth + '_' + season
            
            ax = axd[ax_name]
            
            plot_df = odf_use_SA[(odf_use_SA['site'] == site) & (odf_use_SA['season'] == season) & (odf_use_SA['surf_deep'] == depth)].dropna()
                        
            x = plot_df[season_skagit]
            
            y = plot_df['val']
            
            results = stats.linregress(x,y)
            
            ax.scatter(x, y)
            
            x_reg = np.linspace(x.min(), x.max(), 2) # make two x coordinates from min and max values of SLI_max
            
            y_reg = results.intercept + results.slope*x_reg
            
            ax.text(0.8, 0.9, "r^2 = {:.2f}".format(results.rvalue**2), transform=ax.transAxes)
            
            ax.plot(x_reg, y_reg, '-r')
                    
            ax.set_ylabel(depth + ' ' + season + ' SA [g/kg]')
            
            ax.set_xlabel(season_skagit + ' Skagit flow [m^3/s]')
            
            ax.set_title('SA= ' + depth + '_' + season)
                                    
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
    
    fig.suptitle('full-year skagit vs ' + site + ' SA')

    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' +site + '_allyearSkagit_vs_SA_select.png', dpi=500)


# %%

good_sites = ['SAR003','KSBP01', 'HCB007', 'CRR001', 'ADM003', 'carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'point_jefferson', 'saratoga_passage_mid']

for site in good_sites:      
    
    mosaic = [['surf_grow', 'deep_grow'], ['surf_loDO', 'deep_loDO'], ['surf_winter', 'deep_winter']]
    
    fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=False, layout='constrained', figsize=(10,10), gridspec_kw=dict(wspace=0.1, hspace=0.1))

    for depth in ['surf', 'deep']:
                    
        for season in ['grow', 'loDO', 'winter']:
            
            if season == 'grow':
                
                season_skagit = 'winter'
                
            elif season == 'loDO':
                
                season_skagit = 'grow'
            
            elif season == 'winter':
                
                season_skagit = 'loDO_fudge' #I think this solves the year wrap problem - basically brought last year's fall to the winter year to compare to, so it's the preceding season
                
            ax_name = depth + '_' + season
            
            ax = axd[ax_name]
            
            plot_df = odf_use_SA[(odf_use_SA['site'] == site) & (odf_use_SA['season'] == season) & (odf_use_SA['surf_deep'] == depth)].dropna()
                        
            x = plot_df[season_skagit]
            
            y = plot_df['val']
            
            results = stats.linregress(x,y)
            
            ax.scatter(x, y)
            
            x_reg = np.linspace(x.min(), x.max(), 2) # make two x coordinates from min and max values of SLI_max
            
            y_reg = results.intercept + results.slope*x_reg
            
            ax.text(0.8, 0.9, "r^2 = {:.2f}".format(results.rvalue**2), transform=ax.transAxes)
            
            ax.plot(x_reg, y_reg, '-r')
                    
            ax.set_ylabel(depth + ' ' + season + ' SA [g/kg]')
            
            ax.set_xlabel(season_skagit + ' Skagit flow [m^3/s]')
            
            ax.set_title('SA= ' + depth + '_' + season)
                                    
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
    
    fig.suptitle('last season skagit vs ' + site + ' SA')

    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' +site + '_lastseasonSkagit_vs_SA_select.png', dpi=500)

# %%

good_sites = ['SAR003','KSBP01', 'HCB007', 'CRR001', 'ADM003', 'carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'point_jefferson', 'saratoga_passage_mid']

for site in good_sites:
        
    mosaic = [['surf_grow', 'deep_grow'], ['surf_loDO', 'deep_loDO'], ['surf_winter', 'deep_winter']]
    
    fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=False, layout='constrained', figsize=(10,10), gridspec_kw=dict(wspace=0.1, hspace=0.1))

    for depth in ['surf', 'deep']:
                    
        for season in ['grow', 'loDO', 'winter']:
            
            season_skagit = season
                
            ax_name = depth + '_' + season
            
            ax = axd[ax_name]
            
            plot_df = odf_use_SA[(odf_use_SA['site'] == site) & (odf_use_SA['season'] == season) & (odf_use_SA['surf_deep'] == depth)].dropna()
                        
            x = plot_df[season_skagit]
            
            y = plot_df['val']
            
            results = stats.linregress(x,y)
            
            ax.scatter(x, y)
            
            x_reg = np.linspace(x.min(), x.max(), 2) # make two x coordinates from min and max values of SLI_max
            
            y_reg = results.intercept + results.slope*x_reg
            
            ax.text(0.8, 0.9, "r^2 = {:.2f}".format(results.rvalue**2), transform=ax.transAxes)
            
            ax.plot(x_reg, y_reg, '-r')
                    
            ax.set_ylabel(depth + ' ' + season + ' SA [g/kg]')
            
            ax.set_xlabel(season_skagit + ' Skagit flow [m^3/s]')
            
            ax.set_title('SA= ' + depth + '_' + season)
                                    
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
            
    fig.suptitle('same season skagit vs ' + site + ' SA')

    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' +site + '_sameseasonSkagit_vs_SA_select.png', dpi=500)
    
# %% 

good_sites = ['SAR003','KSBP01', 'HCB007', 'CRR001', 'ADM003', 'carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'point_jefferson', 'saratoga_passage_mid']

for site in good_sites:
        
    mosaic = [['surf_grow', 'deep_grow'], ['surf_loDO', 'deep_loDO'], ['surf_winter', 'deep_winter']]
    
    fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=False, layout='constrained', figsize=(10,10), gridspec_kw=dict(wspace=0.1, hspace=0.1))

    for depth in ['surf', 'deep']:
                    
        for season in ['grow', 'loDO', 'winter']:
            
            if season == 'grow':
                
                season_skagit = 'loDO_fudge'
                
            elif season == 'loDO':
                
                season_skagit = 'winter'
            
            elif season == 'winter':
                
                season_skagit = 'grow_fudge'
                
            ax_name = depth + '_' + season
            
            ax = axd[ax_name]
            
            plot_df = odf_use_SA[(odf_use_SA['site'] == site) & (odf_use_SA['season'] == season) & (odf_use_SA['surf_deep'] == depth)].dropna()
                        
            x = plot_df[season_skagit]
            
            y = plot_df['val']
            
            results = stats.linregress(x,y)
            
            ax.scatter(x, y)
            
            x_reg = np.linspace(x.min(), x.max(), 2) # make two x coordinates from min and max values of SLI_max
            
            y_reg = results.intercept + results.slope*x_reg
            
            ax.text(0.8, 0.9, "r^2 = {:.2f}".format(results.rvalue**2), transform=ax.transAxes)
            
            ax.plot(x_reg, y_reg, '-r')
                    
            ax.set_ylabel(depth + ' ' + season + ' SA [g/kg]')
            
            ax.set_xlabel(season_skagit + ' Skagit flow [m^3/s]')
            
            ax.set_title('SA= ' + depth + '_' + season)
                                    
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
    
    fig.suptitle('two seasons ago skagit vs ' + site + ' SA')

    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' +site + '_twoseasonsagoSkagit_vs_SA_select.png', dpi=500)

# %%    

palette = {'grow':'#dd9404', 'loDO':'#e04256', 'winter':'#4565e8'}


for site in site_list:
    
    mosaic = [['surf_grow', 'deep_grow'], ['surf_loDO', 'deep_loDO'], ['surf_winter', 'deep_winter']]
    
    fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=False, layout='constrained', figsize=(10,10))
        
    for depth in ['surf', 'deep']:
        
        for season in ['grow', 'loDO', 'winter']:
            
            ax = axd[depth + '_' + season]
        
            plot_df = odf_use_SA[(odf_use_SA['site'] == site) & (odf_use_SA['season'] == season) & (odf_use_SA['surf_deep'] == depth)].dropna()
            
            # Define the GAM model
            gam_SA = LinearGAM(s(0))
            gam_skagit_grow = LinearGAM(s(0))
            gam_skagit_loDO = LinearGAM(s(0))
            gam_skagit_winter = LinearGAM(s(0))

            x_pred = plot_df['year']
            
            # Fit the model to the data
            gam_SA.fit(plot_df['year'], plot_df['val'])
            y_pred_SA = gam_SA.predict(x_pred)
            
            gam_skagit_grow.fit(plot_df['year'], plot_df['grow'])
            y_pred_skagit_grow = gam_skagit_grow.predict(x_pred)
            
            gam_skagit_loDO.fit(plot_df['year'], plot_df['loDO'])
            y_pred_skagit_loDO = gam_skagit_loDO.predict(x_pred)
            
            gam_skagit_winter.fit(plot_df['year'], plot_df['winter'])
            y_pred_skagit_winter = gam_skagit_winter.predict(x_pred)
            
            
            ax0 = ax.twinx()
            
            ax.scatter(plot_df['year'], plot_df['val'], marker = 'o', color = palette[season])
            
            ax.plot(x_pred, y_pred_SA, color = palette[season])
            
            ax0.scatter(plot_df['year'], plot_df['grow'], marker = '.', alpha=0.1, color = palette['grow'])
            ax0.plot(x_pred, y_pred_skagit_grow, linestyle = '--', alpha = 0.5, color = palette['grow'])
            
            ax0.scatter(plot_df['year'], plot_df['loDO'], marker = '.', alpha=0.1, color = palette['loDO'])
            ax0.plot(x_pred, y_pred_skagit_loDO, linestyle = '--', alpha = 0.5, color = palette['loDO'])
            
            ax0.scatter(plot_df['year'], plot_df['winter'], marker = '.', alpha=0.1, color = palette['winter'])
            ax0.plot(x_pred, y_pred_skagit_winter, linestyle = '--', alpha = 0.5, color = palette['winter'])
            
            ax.set_ylabel(depth + season + ' average SA [g/kg]')
            
            ax0.set_ylabel('seasonal average Skagit flow [m^3/s]')
            
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
            
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_SA_vs_seasonalSkagit_GAM.png', dpi=500)
    
# %%

good_sites = ['SAR003','KSBP01', 'HCB007', 'CRR001', 'ADM003', 'carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'point_jefferson', 'saratoga_passage_mid']


short_site_coords = odf_depth_mean[odf_depth_mean['site'].isin(short_site_list)].groupby(['site']).first().reset_index()

fig, ax = plt.subplots(figsize=(6,9))


#ax = axd['map_source']
 
ax.set_xlim(X[i1],-121.4)#X[i2]) # Salish Sea
ax.set_ylim(Y[j1],Y[j2]) # Salish Sea
        
ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

sns.scatterplot(data=short_site_coords, x='lon', y='lat', ax = ax, color = 'black')

for idx in short_site_coords.index:
    
    if short_site_coords.iloc[idx]['site'] in good_sites:
        
        color = 'k'
        
    else:
        
        color = 'gray'
        
    ax.annotate(short_site_coords.iloc[idx]['site'], (short_site_coords.iloc[idx]['lon'], short_site_coords.iloc[idx]['lat']), color = color)
    


pfun.add_coast(ax)

pfun.dar(ax)

for site in ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'point_jefferson', 'saratoga_passage_mid']:
    
    path = path_dict[site]
        
    if site in ['near_seattle_offshore']:
        
        patch = patches.PathPatch(path, facecolor='#e04256', edgecolor='white', zorder=1, label='Main Basin', alpha=0.5)
    
    elif site in ['point_jefferson']:
            

        patch = patches.PathPatch(path, facecolor='#e04256', edgecolor='white', zorder=1, alpha=0.5)
                
    elif site in ['saratoga_passage_mid']:
        
        patch = patches.PathPatch(path, facecolor='#4565e8', edgecolor='white', zorder=1, label = 'Sub-Basins', alpha=0.5)
        
    else:
        
        patch = patches.PathPatch(path, facecolor='#4565e8', edgecolor='white', zorder=1, alpha=0.5)
         
    ax.add_patch(patch)
    
ax.text(0.57,0.5, 'PJ', transform=ax.transAxes, fontsize=18, color = '#e04256', path_effects=[pe.withStroke(linewidth=4, foreground="white")])

ax.text(0.54,0.32, 'NS', transform=ax.transAxes, fontsize=18, color = '#e04256', path_effects=[pe.withStroke(linewidth=4, foreground="white")])

    
ax.text(0.62,0.67, 'SP', transform=ax.transAxes, fontsize=18, color = '#4565e8', path_effects=[pe.withStroke(linewidth=4, foreground="white")])

ax.text(0.22,0.29, 'LC', transform=ax.transAxes, fontsize=18, color = '#4565e8', path_effects=[pe.withStroke(linewidth=4, foreground="white")])
 
ax.text(0.48,0.2, 'CI', transform=ax.transAxes, fontsize=18, color = '#4565e8', path_effects=[pe.withStroke(linewidth=4, foreground="white")])

# ax.text(0.15,0.81, 'Strait of\nJuan de Fuca', transform=ax.transAxes, fontsize = 8, color = 'black', ha='center', va='center', rotation = -30)

# ax.text(0.3,0.85, '^ to Strait\nof Georgia', transform=ax.transAxes, fontsize = 7, color = 'black', ha='center', va='center')

# ax.text(0.36,0.785, 'Admiralty\nInlet', transform=ax.transAxes, fontsize = 6, color = 'black', ha='center', va='center')


# ax.text(0.02,0.64 , 'Puget Sound', transform=ax.transAxes, fontsize = 12, color = 'black')

# ax.text(0.025,0.36, 'Hood Canal', transform=ax.transAxes, fontsize = 10, color = 'gray', rotation = 55)

# ax.text(0.57,0.1, 'South Sound', transform=ax.transAxes, fontsize = 10, color = 'gray')

# ax.text(0.77,0.5, 'Main Basin', transform=ax.transAxes, fontsize = 10, color = 'gray', rotation = 50)

# ax.text(0.82,0.73, 'Whidbey Basin', transform=ax.transAxes, fontsize = 10, color = 'gray', rotation = -70)
 
# ax.text(0.86,0.95, 'Skagit\nRiver', transform=ax.transAxes, fontsize = 6, color = 'black', ha='center', va='center')

 


 

#ax.text(0.05,0.025, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')


ax.plot([-122.65,-122.65],[48.35, 48.45], color = 'black', linestyle='--', linewidth=3)

ax.plot([-122.8,-122.7],[48.1, 48.2], color = 'black', linestyle='--', linewidth=3)



ax.plot([-122.75,-122.55],[47.95, 47.9], color = 'gray', linestyle='--', linewidth=2)

ax.plot([-122.61,-122.49],[47.37, 47.27], color = 'gray', linestyle='--', linewidth=2)

ax.plot([-122.61,-122.49],[47.37, 47.27], color = 'gray', linestyle='--', linewidth=2)

ax.plot([-122.40,-122.27],[47.95, 47.87], color = 'gray', linestyle='--', linewidth=2)



 
ax.legend(loc = 'upper left')

ax.set_xlim(-123.2, -122.1) 
 
ax.set_ylim(47,48.5)


ax.set_xlabel('')

ax.set_ylabel('')
 
#xlbl = ax.get_xticklabels()

ax.set_xticks([-123.0, -122.6, -122.2], ['-123.0','-122.6', '-122.2']) #['','-123.0', '', '-122.6', '', '-122.2'])

fig.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/long_short_sites.png', dpi=500)

# %%

good_sites = ['SAR003','KSBP01', 'HCB007', 'CRR001', 'ADM003', 'carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'point_jefferson', 'saratoga_passage_mid']


palette = {'grow':'#dd9404', 'loDO':'#e04256', 'winter':'#4565e8'}


for site in good_sites:
    
    mosaic = [['surf_grow', 'deep_grow'], ['surf_loDO', 'deep_loDO'], ['surf_winter', 'deep_winter']]
    
    fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=False, layout='constrained', figsize=(10,10), gridspec_kw=dict(wspace=0.1, hspace=0.1))
        
    for depth in ['surf', 'deep']:
        
        for season in ['grow', 'loDO', 'winter']:
            
            ax = axd[depth + '_' + season]
        
            plot_df = odf_use_SA[(odf_use_SA['site'] == site) & (odf_use_SA['season'] == season) & (odf_use_SA['surf_deep'] == depth)].dropna()
            
            # Define the GAM model
            gam_SA = LinearGAM(s(0))
            gam_skagit_grow = LinearGAM(s(0))
            gam_skagit_loDO = LinearGAM(s(0))
            gam_skagit_winter = LinearGAM(s(0))
            gam_skagit_allyear = LinearGAM(s(0))

            x_pred = plot_df['year']
            
            # Fit the model to the data
            gam_SA.fit(plot_df['year'], plot_df['val'])
            y_pred_SA = gam_SA.predict(x_pred)
            
            gam_skagit_grow.fit(plot_df['year'], plot_df['grow'])
            y_pred_skagit_grow = gam_skagit_grow.predict(x_pred)
            
            gam_skagit_loDO.fit(plot_df['year'], plot_df['loDO'])
            y_pred_skagit_loDO = gam_skagit_loDO.predict(x_pred)
            
            gam_skagit_winter.fit(plot_df['year'], plot_df['winter'])
            y_pred_skagit_winter = gam_skagit_winter.predict(x_pred)
            
            gam_skagit_allyear.fit(plot_df['year'], plot_df['allyear'])
            y_pred_skagit_allyear = gam_skagit_allyear.predict(x_pred)
            
            
            ax0 = ax.twinx()
            
            ax.scatter(plot_df['year'], plot_df['val'], marker = 'o', color = palette[season])
            
            ax.plot(x_pred, y_pred_SA, color = palette[season])
            
            ax0.scatter(plot_df['year'], plot_df['grow'], marker = '.', alpha=0.1, color = palette['grow'])
            ax0.plot(x_pred, y_pred_skagit_grow, linestyle = '--', alpha = 0.5, color = palette['grow'], label='grow')
            
            ax0.scatter(plot_df['year'], plot_df['loDO'], marker = '.', alpha=0.1, color = palette['loDO'])
            ax0.plot(x_pred, y_pred_skagit_loDO, linestyle = '--', alpha = 0.5, color = palette['loDO'], label = 'loDO')
            
            ax0.scatter(plot_df['year'], plot_df['winter'], marker = '.', alpha=0.1, color = palette['winter'])
            ax0.plot(x_pred, y_pred_skagit_winter, linestyle = '--', alpha = 0.5, color = palette['winter'], label = 'winter')
            
            ax0.scatter(plot_df['year'], plot_df['allyear'], marker = '.', alpha=0.1, color = 'k')
            ax0.plot(x_pred, y_pred_skagit_allyear, linestyle = '--', alpha = 0.5, color = 'k', label = 'allyear')
            
            ax.set_title('SA= ' + depth + '_' + season)
            
            ax.set_ylabel('seasonal average SA [g/kg]')
            
            ax0.set_ylabel('seasonal/annual average Skagit flow [m^3/s]')
            
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
            
    fig.suptitle(site)
            
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_SA_vs_seasonalSkagit_GAM_select.png', dpi=500)
    
# %%

fig, ax0 = plt.subplots(figsize=(6,4))

# Define the GAM model
gam_SA = LinearGAM(s(0))
gam_skagit_grow = LinearGAM(s(0))
gam_skagit_loDO = LinearGAM(s(0))
gam_skagit_winter = LinearGAM(s(0))
gam_skagit_allyear = LinearGAM(s(0))

plot_df = skagit_means

x_pred = plot_df['year']

gam_skagit_grow.fit(plot_df['year'], plot_df['grow'])
y_pred_skagit_grow = gam_skagit_grow.predict(x_pred)

gam_skagit_loDO.fit(plot_df['year'], plot_df['loDO'])
y_pred_skagit_loDO = gam_skagit_loDO.predict(x_pred)

gam_skagit_winter.fit(plot_df['year'], plot_df['winter'])
y_pred_skagit_winter = gam_skagit_winter.predict(x_pred)

gam_skagit_allyear.fit(plot_df['year'], plot_df['allyear'])
y_pred_skagit_allyear = gam_skagit_allyear.predict(x_pred)

ax0.scatter(plot_df['year'], plot_df['grow'], marker = '.', alpha=0.1, color = palette['grow'])
ax0.plot(x_pred, y_pred_skagit_grow, linestyle = '--', alpha = 1, color = palette['grow'], label='grow')

ax0.scatter(plot_df['year'], plot_df['loDO'], marker = '.', alpha=0.1, color = palette['loDO'])
ax0.plot(x_pred, y_pred_skagit_loDO, linestyle = '--', alpha = 1, color = palette['loDO'], label = 'loDO')

ax0.scatter(plot_df['year'], plot_df['winter'], marker = '.', alpha=0.1, color = palette['winter'])
ax0.plot(x_pred, y_pred_skagit_winter, linestyle = '--', alpha = 1, color = palette['winter'], label = 'winter')

ax0.scatter(plot_df['year'], plot_df['allyear'], marker = '.', alpha=0.1, color = 'k')
ax0.plot(x_pred, y_pred_skagit_allyear, linestyle = '--', alpha = 1, color = 'k', label = 'allyear')

ax0.set_ylabel('seasonal/annual average Skagit flow [m^3/s]')

ax0.grid(color = 'lightgray', linestyle = '--', alpha=0.3)

ax0.set_ylim(0,1500)

ax0.legend(loc = 'upper right')

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/skagit_allyearseasonal_GAM.png', dpi=500)

# %%

fig, ax0 = plt.subplots(figsize=(6,4))

# Define the GAM model
gam_SA = LinearGAM(s(0))
gam_skagit_grow = LinearGAM(s(0))
gam_skagit_loDO = LinearGAM(s(0))
gam_skagit_winter = LinearGAM(s(0))
gam_skagit_allyear = LinearGAM(s(0))

plot_df = monthly_skagit_df

x_pred = plot_df['datetime']

plot_df_grow = monthly_skagit_df[monthly_skagit_df['season'] == 'grow']

plot_df_loDO = monthly_skagit_df[monthly_skagit_df['season'] == 'loDO']

plot_df_winter = monthly_skagit_df[monthly_skagit_df['season'] == 'winter']



gam_skagit_grow.fit(plot_df_grow['datetime'], plot_df_grow['mean_va'])
y_pred_skagit_grow = gam_skagit_grow.predict(x_pred)

gam_skagit_loDO.fit(plot_df_loDO['datetime'], plot_df_loDO['mean_va'])
y_pred_skagit_loDO = gam_skagit_loDO.predict(x_pred)

gam_skagit_winter.fit(plot_df_winter['datetime'], plot_df_winter['mean_va'])
y_pred_skagit_winter = gam_skagit_winter.predict(x_pred)

gam_skagit_allyear.fit(plot_df['datetime'], plot_df['mean_va'])
y_pred_skagit_allyear = gam_skagit_allyear.predict(x_pred)

ax0.scatter(plot_df_grow['datetime'], plot_df_grow['mean_va'], marker = '.', alpha=0.1, color = palette['grow'])
ax0.plot(x_pred, y_pred_skagit_grow, linestyle = '--', alpha = 1, color = palette['grow'], label='grow')

ax0.scatter(plot_df_loDO['datetime'], plot_df_loDO['mean_va'], marker = '.', alpha=0.1, color = palette['loDO'])
ax0.plot(x_pred, y_pred_skagit_loDO, linestyle = '--', alpha = 1, color = palette['loDO'], label = 'loDO')

ax0.scatter(plot_df_winter['datetime'], plot_df_winter['mean_va'], marker = '.', alpha=0.1, color = palette['winter'])
ax0.plot(x_pred, y_pred_skagit_winter, linestyle = '--', alpha = 1, color = palette['winter'], label = 'winter')

#ax0.scatter(plot_df['datetime'], plot_df['mean_va'], marker = '.', alpha=0.1, color = 'k')
ax0.plot(x_pred, y_pred_skagit_allyear, linestyle = '--', alpha = 1, color = 'k', label = 'allyear')

ax0.set_ylabel('monthly average Skagit flow [m^3/s]')

ax0.grid(color = 'lightgray', linestyle = '--', alpha=0.3)

ax0.legend(loc = 'upper right')

ax0.set_ylim(0,1500)


plt.savefig('/Users/dakotamascarenas/Desktop/pltz/skagit_allyearseasonal_monthly_GAM.png', dpi=500)

# %%




            
        
            
            
            
            
            
            
            
            
            
        
    
    
    
    