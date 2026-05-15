#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:36:17 2024

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


# %%

poly_list = ['ps', 'mb', 'wb', 'ss', 'hc', 'admiralty_sill', 'budd_inlet', 'carr_inlet_mid', 'dana_passage', 'hat_island', 'hazel_point', 'hood_canal_mouth', 'lynch_cove_mid', 'near_seattle_offshore', 'near_edmonds', 'port_susan_mid', 'saratoga_passage_north', 'saratoga_passage_mid']

odf_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))

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

# %%

odf['source_type'] = odf['source'] + '_' + odf['otype']

# %%

lc_exclude = odf[(odf['segment'] == 'lynch_cove_mid') & (odf['z'] < -45)]

odf = odf[~odf['cid'].isin(lc_exclude['cid'].unique())]

# %%

temp = odf[(odf['segment'] == 'near_seattle_offshore')].groupby('cid').agg({'z':'min'}).reset_index()

sea_offshore_exclude = temp[temp['z'] >-60]

odf = odf[~odf['cid'].isin(sea_offshore_exclude['cid'].unique())]

# %%

odf = odf.dropna()

# %%

odf = odf.assign(
    ix=(lambda x: x['lon'].apply(lambda x: zfun.find_nearest_ind(lon_1D, x))),
    iy=(lambda x: x['lat'].apply(lambda x: zfun.find_nearest_ind(lat_1D, x)))
)

# %%

odf['h'] = odf.apply(lambda x: -depths[x['iy'], x['ix']], axis=1)

# %%

odf['yearday'] = odf['datetime'].dt.dayofyear

# %%

odf = odf[odf['val'] >0]

# %%

odf_grow = odf[(odf['yearday'] > 200) & (odf['yearday'] <= 300)]

# %%

for basin in basin_list:
    
    fig, ax = plt.subplots()
    
    plot_df = odf[(odf['segment'] == basin) & (odf['season'] == 'fall') & (odf['var'] == 'DO_mg_L')].groupby('cid').min().reset_index()
    
    sns.scatterplot(data = plot_df, x='datetime', y='h', hue='val')
    
    ax.set_title(basin + ' fall DO casts bath depth')
    
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_fall_DO_bath_depths.png', bbox_inches='tight', dpi=500)
    
# %%

# OKAY ACTUALLY, FIRST I WANT TO LOOK AT THE SEASONAL HYPOXIC MINIMA...2D histogram again like Aurora's model stuff

for basin in ['ps', 'mb', 'wb', 'hc', 'ss']:
    
    for decade in odf['decade'].unique():
    
        fig, ax = plt.subplots()
                
        #plt.rc('font', size=14)
        
        plot_df = odf[(odf['segment'] == basin) & (odf['var'] == 'DO_mg_L') & (odf['decade'] == decade)].groupby('cid').min().reset_index()
        
        plt.hist2d(plot_df['yearday'], plot_df['val'], bins=100, cmap='inferno', cmin=1)
        
       # plt.colorbar()
        
        #ax[0].set_xlabel('SA cast maxima [psu]')
        
        #ax[0].set_ylabel('z cast minima [m]')
        
        plt.axhspan(0,2, color = 'lightgray', alpha = 0.2) 
        
        plt.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        plt.title(basin + ' ' + decade + ' yearday vs. DO historgram')
        
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + decade + '_yearday_vs_DO_hist.png', bbox_inches='tight', dpi=500)


# %%

for basin in ['ps']:
    
    for decade in odf['decade'].unique():
        
        fig, ax = plt.subplots(figsize = (8,16))
        
        plt.rc('font', size=14)
        
        plot_df = odf[(odf['segment'] == basin) & (odf['decade'] == decade) & (odf['var'] == 'DO_mg_L')].groupby('cid').min().reset_index()
        
        sns.scatterplot(data = plot_df, x = 'lon', y = 'lat', hue = 'val')
        
        sns.scatterplot(data = plot_df[plot_df['val'] <2], x = 'lon', y = 'lat', color = 'r', alpha = 0.5)
        
        pfun.add_coast(ax)

        pfun.dar(ax)

        ax.set_xlim(-123.5, -122)

        ax.set_ylim(46.9,48.5)

        ax.set(xlabel=None)
         
        ax.set(ylabel=None)

        ax.tick_params(axis='x', labelrotation=45)
        
        ax.set_title(basin + ' ' + decade + 's DO mins')
        
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + decade + '_DO_min_map.png', bbox_inches='tight', dpi=500)

    
# %%

for basin in ['ps']:
    
    for decade in odf_grow['decade'].unique():
        
        fig, ax = plt.subplots(figsize = (8,16))
        
        plt.rc('font', size=14)
        
        plot_df = odf_grow[(odf['segment'] == basin) & (odf_grow['decade'] == decade) & (odf_grow['var'] == 'DO_mg_L')].groupby('cid').min().reset_index()
        
        sns.scatterplot(data = plot_df, x = 'lon', y = 'lat', hue = 'val')
        
        sns.scatterplot(data = plot_df[plot_df['val'] <2], x = 'lon', y = 'lat', color = 'r', alpha = 0.5)
        
        pfun.add_coast(ax)

        pfun.dar(ax)

        ax.set_xlim(-123.5, -122)

        ax.set_ylim(46.9,48.5)

        ax.set(xlabel=None)
         
        ax.set(ylabel=None)

        ax.tick_params(axis='x', labelrotation=45)
        
        ax.set_title(basin + ' ' + decade + 's DO mins GROW SEASON')
        
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + decade + '_DO_min_map_grow.png', bbox_inches='tight', dpi=500)



# %%

for basin in basin_list:
    
    fig, ax = plt.subplots()
    
    plot_df = odf_grow[(odf_grow['segment'] == basin) & (odf_grow['var'] == 'DO_mg_L')].groupby('cid').min().reset_index()
    
    sns.scatterplot(data = plot_df, x='datetime', y='h', hue='val')
    
    ax.set_title(basin + ' fall DO casts bath depth GROW SEASON')
    
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_fall_DO_bath_depths_grow.png', bbox_inches='tight', dpi=500)









