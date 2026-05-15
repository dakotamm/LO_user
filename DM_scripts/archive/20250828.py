#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 14:51:40 2025

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

odf_use = (odf_use
                  .rename(columns={'date_ordinal_mean':'date_ordinal'})
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

odf_calc_use = (odf_calc_use
                  .rename(columns={'date_ordinal_mean':'date_ordinal'})
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

# %%


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

all_stats_filt.loc[all_stats_filt['site'] == 'saratoga_passage_mid', 'site_num'] = 4

all_stats_filt.loc[all_stats_filt['site'] == 'carr_inlet_mid', 'site_num'] = 3

all_stats_filt.loc[all_stats_filt['site'] == 'lynch_cove_mid', 'site_num'] = 5

# %%

odf_SA = odf[(odf['var'] == 'SA')]

    
# %%

mosaic = [['grow'], ['loDO'], ['winter']]

big_df = odf_use[odf_use['var'] == 'SA']

for site in long_site_list:
    
    fig, axd = plt.subplot_mosaic(mosaic, figsize=(6,12), layout='constrained', sharex=True, sharey=True, gridspec_kw=dict(wspace=0.1, hspace=0.1))
    
    for season in odf_use['season'].unique():
        
        if season == 'loDO':
            color = 'red'  
            label = 'Aug-Nov (Low DO)'
        elif season == 'winter':
            color = 'blue'
            label = 'Dec-Mar (Winter)'
        elif season == 'grow':
            color = 'gold'
            label = 'Apr-Jul (Spring Bloom)'
        
        ax = axd[season]
        
        plot_df = big_df[(big_df['surf_deep'] == 'surf') & (big_df['season'] == season) & (big_df['site'] == site)]
        
        ax.scatter(plot_df['datetime'], plot_df['val'], color = color, alpha=0.5)
        
        ax.text(0.05,0.95, season, transform=ax.transAxes, verticalalignment='top', fontweight = 'bold', color='k')
        
        ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3) 
        
        ax.set_xlabel('')
        
        ax.set_ylabel('SA [g/kg]')
        
        ax.set_ylim(15,35)
        
    plt.suptitle(site + ' surf')
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_surfSA_timeseries_smallerscale.png', dpi=500)
        

# %%

mosaic = [['1930', '1940'], ['1950', '1960'], ['1970', '1980'], ['1990', '2000'], ['2010', '2020']]

decade_colors = {
    "1930": "#fcd225",
    "1940": "#fdae32",
    "1950": "#f68d45",
    "1960": "#e76f5a",
    "1970": "#d5546e",
    "1980": "#c03a83",
    "1990": "#a62098",
    "2000": "#8606a6",
    "2010": "#6300a7",
    "2020": "#3e049c"
}

for site in long_site_list:


    for season in odf_SA['season'].unique():
        
        fig, axd = plt.subplot_mosaic(mosaic, figsize=(5,12), layout='constrained', sharex=True, sharey=True, gridspec_kw=dict(wspace=0.1, hspace=0.1))
        
        big_df = odf_SA[(odf_SA['surf_deep'] == 'surf') & (odf_SA['season'] == season) & (odf_SA['site'] == site)]
    
        for decade in odf_SA['decade'].unique():
            
            ax = axd[decade]
                    
            ax.scatter(big_df['val'], big_df['z'], color='gray', alpha=0.1, s=10)
            
            plot_df = big_df[big_df['decade'] == decade]
            
            bin_edges = [-5, -4, -3, -2, -1, 1]
            bin_labels = ['4-5m', '3-4m', '2-3m', '1-2m', '0-1m']
    
            # Create a 'Depth_Bin' column
            plot_df['depth_bin'] = pd.cut(plot_df['z'], bins=bin_edges, labels=bin_labels, right=False)
        
            # Calculate the average of 'Value1' and 'Value2' for each depth bin
            average_by_bin = plot_df.groupby('depth_bin')[['z','val']].mean().reset_index()
            
            average_by_bin['val_bin'] = average_by_bin['val']
            
            average_by_bin['z_bin'] = average_by_bin['z']
    
            
            plot_df = pd.merge(plot_df, average_by_bin[['depth_bin', 'val_bin', 'z_bin']], how='left', on=['depth_bin'])
    
    
            
            ax.scatter(plot_df['val'], plot_df['z'], color=decade_colors[decade], alpha=0.5)
                            
            ax.plot(average_by_bin['val_bin'].dropna(), average_by_bin['z_bin'].dropna(), color='k')
            
            ax.scatter(average_by_bin['val_bin'], average_by_bin['z_bin'], color='k', marker = 'P', s=50)
            
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
            
            ax.set_ylabel('z [m]')
            
            ax.set_xlabel('SA [g/kg]')
            
            ax.set_xlim(15,30)
            
            ax.text(0.05,0.95, decade+'s', transform=ax.transAxes, verticalalignment='top', fontweight = 'bold', color='k')
            
        plt.suptitle(site + ' ' + season + ' surf')
    
            
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_' + season + '_surfSA_bydecade_smallerscale.png', dpi=500)
        
# %%

mosaic = [['1930', '1940'], ['1950', '1960'], ['1970', '1980'], ['1990', '2000'], ['2010', '2020']]

decade_colors = {
    "1930": "#fcd225",
    "1940": "#fdae32",
    "1950": "#f68d45",
    "1960": "#e76f5a",
    "1970": "#d5546e",
    "1980": "#c03a83",
    "1990": "#a62098",
    "2000": "#8606a6",
    "2010": "#6300a7",
    "2020": "#3e049c"
}

for site in long_site_list:


    for season in odf_SA['season'].unique():
        
        fig, axd = plt.subplot_mosaic(mosaic, figsize=(5,12), layout='constrained', sharex=True, sharey=True, gridspec_kw=dict(wspace=0.1, hspace=0.1))
        
        big_df = odf_SA[(odf_SA['surf_deep'] == 'surf') & (odf_SA['season'] == season) & (odf_SA['site'] == site)]
        
        bin_edges = [-5, -4, -3, -2, -1, 1]
        bin_labels = ['4-5m', '3-4m', '2-3m', '1-2m', '0-1m']

        
        # Create a 'Depth_Bin' column
        big_df['depth_bin'] = pd.cut(big_df['z'], bins=bin_edges, labels=bin_labels, right=False)
    
        # Calculate the average of 'Value1' and 'Value2' for each depth bin
        average_by_bin_big = big_df.groupby('depth_bin')[['z','val']].mean().reset_index()
        
        average_by_bin_big['val_bin'] = average_by_bin_big['val']
        
        average_by_bin_big['z_bin'] = average_by_bin_big['z']
        
        big_df = pd.merge(big_df, average_by_bin_big[['depth_bin', 'val_bin', 'z_bin']], how='left', on=['depth_bin'])
    
    
    
        for decade in odf_SA['decade'].unique():
            
            ax = axd[decade]
                    
            ax.scatter(big_df['val'], big_df['z'], color='gray', alpha=0.1, s=10)
            
            plot_df = big_df[big_df['decade'] == decade]
            
            bin_edges = [-5, -4, -3, -2, -1, 1]
            bin_labels = ['4-5m', '3-4m', '2-3m', '1-2m', '0-1m']
    
            # Create a 'Depth_Bin' column
            plot_df['depth_bin'] = pd.cut(plot_df['z'], bins=bin_edges, labels=bin_labels, right=False)
        
            # Calculate the average of 'Value1' and 'Value2' for each depth bin
            average_by_bin = plot_df.groupby('depth_bin')[['z','val']].mean().reset_index()
            
            average_by_bin['val_bin'] = average_by_bin['val']
            
            average_by_bin['z_bin'] = average_by_bin['z']
    
            
            plot_df = pd.merge(plot_df, average_by_bin[['depth_bin', 'val_bin', 'z_bin']], how='left', on=['depth_bin'])
    
    
            
            ax.scatter(plot_df['val'], plot_df['z'], color=decade_colors[decade], alpha=0.5)
                            
            ax.plot(average_by_bin_big['val_bin'].dropna(), average_by_bin_big['z_bin'].dropna(), color='k')
            
            ax.scatter(average_by_bin['val_bin'], average_by_bin['z_bin'], color='k', marker = 'P', s=50)
            
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
            
            ax.set_ylabel('z [m]')
            
            ax.set_xlabel('SA [g/kg]')
            
            ax.set_xlim(15,30)
            
            ax.text(0.05,0.95, decade+'s', transform=ax.transAxes, verticalalignment='top', fontweight = 'bold', color='k')
            
        plt.suptitle(site + ' ' + season + ' surf')
    
            
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_' + season + '_surfSA_bydecade_smallerscale_meanline_decavg.png', dpi=500)
        

# %%

mosaic = [['1930', '1940'], ['1950', '1960'], ['1970', '1980'], ['1990', '2000'], ['2010', '2020']]

decade_colors = {
    "1930": "#fcd225",
    "1940": "#fdae32",
    "1950": "#f68d45",
    "1960": "#e76f5a",
    "1970": "#d5546e",
    "1980": "#c03a83",
    "1990": "#a62098",
    "2000": "#8606a6",
    "2010": "#6300a7",
    "2020": "#3e049c"
}

for site in long_site_list:


    for season in odf_SA['season'].unique():
        
        fig, axd = plt.subplot_mosaic(mosaic, figsize=(5,12), layout='constrained', sharex=True, sharey=True, gridspec_kw=dict(wspace=0.1, hspace=0.1))
        
        big_df = odf_SA[(odf_SA['surf_deep'] == 'surf') & (odf_SA['season'] == season) & (odf_SA['site'] == site)]
        
        bin_edges = [-5, -4, -3, -2, -1, 1]
        bin_labels = ['4-5m', '3-4m', '2-3m', '1-2m', '0-1m']

        
        # Create a 'Depth_Bin' column
        big_df['depth_bin'] = pd.cut(big_df['z'], bins=bin_edges, labels=bin_labels, right=False)
    
        # Calculate the average of 'Value1' and 'Value2' for each depth bin
        average_by_bin = big_df.groupby('depth_bin')[['z','val']].mean().reset_index()
        
        average_by_bin['val_bin'] = average_by_bin['val']
        
        average_by_bin['z_bin'] = average_by_bin['z']
        
        big_df = pd.merge(big_df, average_by_bin[['depth_bin', 'val_bin', 'z_bin']], how='left', on=['depth_bin'])
    
    
    
        for decade in odf_SA['decade'].unique():
            
            ax = axd[decade]
                    
            ax.scatter(big_df['val'], big_df['z'], color='gray', alpha=0.1, s=10)
            
            plot_df = big_df[big_df['decade'] == decade]
    
    
            
            ax.scatter(plot_df['val'], plot_df['z'], color=decade_colors[decade], alpha=0.5)
                            
            ax.plot(average_by_bin['val_bin'].dropna(), average_by_bin['z_bin'].dropna(), color='k')
            
            ax.scatter(average_by_bin['val_bin'], average_by_bin['z_bin'], color='k', marker = 'P', s=50)
            
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
            
            ax.set_ylabel('z [m]')
            
            ax.set_xlabel('SA [g/kg]')
            
            ax.set_xlim(15,30)
            
            ax.text(0.05,0.95, decade+'s', transform=ax.transAxes, verticalalignment='top', fontweight = 'bold', color='k')
            
        plt.suptitle(site + ' ' + season + ' surf')
    
            
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_' + season + '_surfSA_bydecade_smallerscale_meanline.png', dpi=500)
        
# %%

mosaic = [['grow'], ['loDO'], ['winter']]


for site in long_site_list:
    
    fig, axd = plt.subplot_mosaic(mosaic, figsize=(6,12), layout='constrained', sharex=True, sharey=True, gridspec_kw=dict(wspace=0.1, hspace=0.1))
    
    for season in odf['season'].unique():
        
        plot_df = odf_SA[(odf_SA['surf_deep'] == 'surf') & (odf_SA['season'] == season) & (odf_SA['site'] == site)]
        
        if season == 'loDO':
            color = 'red'  
            label = 'Aug-Nov (Low DO)'
        elif season == 'winter':
            color = 'blue'
            label = 'Dec-Mar (Winter)'
        elif season == 'grow':
            color = 'gold'
            label = 'Apr-Jul (Spring Bloom)'
        
        ax = axd[season]
        
        bin_edges = [-5, -4, -3, -2, -1, 1]
        bin_labels = ['4-5m', '3-4m', '2-3m', '1-2m', '0-1m']
        
        # Create a 'Depth_Bin' column
        plot_df['depth_bin'] = pd.cut(plot_df['z'], bins=bin_edges, labels=bin_labels, right=False)
    
        # Calculate the average of 'Value1' and 'Value2' for each depth bin
        average_by_bin_by_year = plot_df.groupby(['year','depth_bin'])[['z','val']].mean().reset_index()
        
        average_by_bin_by_year['val_bin'] = average_by_bin_by_year['val']
        
        average_by_bin_by_year['z_bin'] = average_by_bin_by_year['z']

        
        plot_df = pd.merge(plot_df, average_by_bin_by_year[['year', 'depth_bin', 'val_bin', 'z_bin']], how='left', on=['year', 'depth_bin'])
        
        for dbin in ['0-1m', '4-5m']:
            
            if dbin == '0-1m':
                
                color_ = color
                
                ycoord = 0.1
            
            else:
                
                color_ = 'k'
                
                ycoord = 0.05
            
        
            y = plot_df[plot_df['depth_bin'] == dbin]['val_bin']
            
            x = plot_df[plot_df['depth_bin'] == dbin]['year']
        
            result = stats.theilslopes(y,x,alpha=0.95)

            B1 = result.slope

            B0 = result.intercept
            
            high_sB1 = result.high_slope
            
            low_sB1 = result.low_slope
                
            ax.scatter(plot_df[plot_df['depth_bin'] == dbin]['year'], plot_df[plot_df['depth_bin'] == dbin]['val_bin'], color = color_, alpha=0.5, label = dbin)
        
            x_plot = np.linspace(plot_df['year'].min(), plot_df['year'].max(), 100) 
            
            y_plot = B1*x_plot + B0
            
            ax.plot(x_plot, y_plot, alpha=0.5, color = color_)
            
            ax.text(0.05, ycoord, dbin + ' slope (95% CI) = ' + str(round(B1*100,2)) + '[g/kg]/cent. (' + str(round(low_sB1*100,2)) + '-' + str(round(high_sB1*100,2)) + ')', transform=ax.transAxes, verticalalignment='top', color='k')

        ax.text(0.05,0.95, season, transform=ax.transAxes, verticalalignment='top', fontweight = 'bold', color='k')
        
        ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3) 
        
        ax.set_xlabel('')
        
        ax.set_ylabel('SA [g/kg]')
        
        ax.set_ylim(15,35)
        
        ax.legend()
        
    plt.suptitle(site + ' surf')
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_1m_5m_seasonalannualSA_timeseries_smallerscale.png', dpi=500)
    
# %%

mosaic = [['grow'], ['loDO'], ['winter']]

big_df = odf_use[odf_use['var'] == 'SA']

for site in long_site_list:
    
    fig, axd = plt.subplot_mosaic(mosaic, figsize=(6,12), layout='constrained', sharex=True, sharey=True, gridspec_kw=dict(wspace=0.1, hspace=0.1))
    
    for season in odf_use['season'].unique():
        
        if season == 'loDO':
            color = 'red'  
            label = 'Aug-Nov (Low DO)'
        elif season == 'winter':
            color = 'blue'
            label = 'Dec-Mar (Winter)'
        elif season == 'grow':
            color = 'gold'
            label = 'Apr-Jul (Spring Bloom)'
        
        ax = axd[season]
        
        plot_df = big_df[(big_df['surf_deep'] == 'surf') & (big_df['season'] == season) & (big_df['site'] == site)]
        
        for month in plot_df['month'].unique():
            
            little_df = plot_df[plot_df['month'] == month]
            
            x = little_df['date_ordinal']
            
            x_plot = little_df['datetime']
            
            y = little_df['val']
        
            result = stats.theilslopes(y,x,alpha=0.95)

            B1 = result.slope

            B0 = result.intercept
            
            high_sB1 = result.high_slope
            
            low_sB1 = result.low_slope
            
            slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_plot.max().year - x_plot.min().year)
            
            slope_datetime_s_hi = (B0 + high_sB1*x.max() - (B0 + high_sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
            
            slope_datetime_s_lo = (B0 + low_sB1*x.max() - (B0 + low_sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
                
            ax.scatter(little_df['datetime'], little_df['val'], alpha=0.5, label = 'month ' + str(month) + ' slope (95% CI) = ' + str(round(slope_datetime*100,2)) + '[g/kg]/cent. (' + str(round(slope_datetime_s_lo*100,2)) + '-' + str(round(slope_datetime_s_hi*100,2)) + ')')
                    
            ax.plot([x_plot.min(), x_plot.max()], [B0 + B1*x.min(), B0 + B1*x.max()], alpha=0.5)
            
           # ax.text(0.05, ycoord, dbin + ' slope (95% CI) = ' + str(round(B1*100,2)) + '[g/kg]/cent. s(' + str(round(low_sB1*100,2)) + '-' + str(round(high_sB1*100,2)) + ')', transform=ax.transAxes, verticalalignment='top', color='k')

            
            
        
       # ax.scatter(plot_df['datetime'], plot_df['val'], color = color, alpha=0.5)
        
        ax.text(0.05,0.95, season, transform=ax.transAxes, verticalalignment='top', fontweight = 'bold', color='k')
        
        ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3) 
        
        ax.set_xlabel('')
        
        ax.set_ylabel('SA [g/kg]')
        
        ax.set_ylim(15,35)
        
        ax.legend()
        
    plt.suptitle(site + ' surf')
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_surf_monthly_SA_timeseries_smallerscale.png', dpi=500)
    

# %%

mosaic = [['grow'], ['loDO'], ['winter']]

big_df = odf_use[odf_use['var'] == 'SA']

palette = {'loDO':'red', 'grow':'gold','winter':'blue'}

for site in long_site_list:
    
    #fig, axd = plt.subplot_mosaic(mosaic, figsize=(6,12), layout='constrained', sharex=True, sharey=True, gridspec_kw=dict(wspace=0.1, hspace=0.1))
    
    plot_df = big_df[(big_df['surf_deep'] == 'surf') & (big_df['site'] == site)]
    
    num_samples_per_month = plot_df[['year','season','month','val']].groupby(['year','season','month']).count().reset_index()
    
    num_samples_per_month['count'] = num_samples_per_month['val']
    
    sns.relplot(data = num_samples_per_month, x='year', y='count', hue='season', palette = palette, col='month', col_wrap=3, height=2)

    
    # for season in odf_use['season'].unique():
        
    #     if season == 'loDO':
    #         color = 'red'  
    #         label = 'Aug-Nov (Low DO)'
    #     elif season == 'winter':
    #         color = 'blue'
    #         label = 'Dec-Mar (Winter)'
    #     elif season == 'grow':
    #         color = 'gold'
    #         label = 'Apr-Jul (Spring Bloom)'
            
        
        #ax = axd[season]
        
        #plot_df = big_df[(big_df['surf_deep'] == 'surf') & (big_df['season'] == season) & (big_df['site'] == site)]
        
        # num_samples_per_month = plot_df[['year','month','val']].groupby(['year','month']).count().reset_index()
        
        # num_samples_per_month['count'] = num_samples_per_month['val']
        
        # for month in num_samples_per_month['month'].unique():
        
        #     ax.scatter(num_samples_per_month[num_samples_per_month['month'] == month]['year'], num_samples_per_month[num_samples_per_month['month'] == month]['count'], alpha=0.5, label = 'month ' + str(month))
        
        # ax.text(0.05,0.95, season, transform=ax.transAxes, verticalalignment='top', fontweight = 'bold', color='k')
        
        # ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3) 
        
        # ax.set_xlabel('')
        
        # ax.set_ylabel('num sample per month')
        
        # ax.legend()
                
    #plt.suptitle(site + ' surf')
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_surfSA_samplemonthcount.png', dpi=500)