#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 12:29:22 2025

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




poly_list = ['mb', 'hc', 'ss', 'wb'] # 5 sites + 4 basins

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

odf_use = dfun.monthlyDepthAverageDF(odf_depth_mean) #don't worry about filtering median because only using SA, nor about long variable

good_sites = ['SAR003','KSBP01', 'HCB007', 'CRR001', 'ADM003']

odf_use = odf_use[odf_use['site'].isin(good_sites)]

odf_use = odf_use[odf_use['var'] == 'SA']

monthly_skagit_df['year'] = monthly_skagit_df['year_nu']

monthly_skagit_df['month'] = monthly_skagit_df['month_nu']

# %%

odf_use = pd.merge(odf_use, monthly_skagit_df[['year', 'month','mean_va']], how='left', on=['year', 'month']).dropna()

# %%

mosaic = [['skagit'], ['ADM003'], ['KSBP01'], ['CRR001'], ['SAR003'], ['HCB007']]

fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=True, layout='constrained', figsize=(6,10), gridspec_kw=dict(wspace=0.1, hspace=0.1))

ax = axd['skagit']

plot_df = odf_use.groupby(['year', 'month']).first().reset_index()

lags = np.arange(1,int(len(plot_df)/4))

R = np.empty(len(lags))

for i, k in enumerate(lags):
    R[i] = plot_df['mean_va'].autocorr(lag=lags[i])
    
ax.plot(lags, R,'-.', label='skagit', color = 'gray')

ax.legend()

ax.set_ylabel('Autocorrelation, R')
ax.set_xlabel('Lag, months')

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)



for site in good_sites:
    
    ax = axd[site]
    
    ax.set_title(site)
    
    for depth in ['surf', 'deep']:
        
        if depth == 'surf':
            
            color = 'blue'
            
        else:
            
            color = 'red'
        
        plot_df = odf_use[(odf_use['site'] == site) & (odf_use['surf_deep'] == depth)]
        
        lags = np.arange(1,int(len(plot_df)/4))
 
        R = np.empty(len(lags))

        for i, k in enumerate(lags):
            R[i] = plot_df['val'].autocorr(lag=lags[i])
            
        ax.plot(lags, R,'-.', label=depth, color = color)

    ax.legend()

    ax.set_ylabel('Autocorrelation, R')
    ax.set_xlabel('Lag, months')

    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
    
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/ecology_skagit_autocorr.png', dpi=500)

# %%

def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    
    https://stackoverflow.com/questions/33171413/cross-correlation-time-lag-correlation-with-pandas
    
    """
    return datax.corr(datay.shift(lag))

# %%

mosaic = [['skagit'], ['ADM003'], ['KSBP01'], ['CRR001'], ['SAR003'], ['HCB007']]

fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=True, layout='constrained', figsize=(6,10), gridspec_kw=dict(wspace=0.1, hspace=0.1))

ax = axd['skagit']

plot_df = odf_use.groupby(['year', 'month']).first().reset_index()

lags = np.arange(1,24)

R = np.empty(len(lags))

for i, k in enumerate(lags):
    R[i] = plot_df['mean_va'].autocorr(lag=lags[i])
    
ax.plot(lags, R,'-.', label='skagit', color = 'gray')

ax.legend()

ax.set_ylabel('Autocorrelation, R')
ax.set_xlabel('Lag, months')

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)





for site in good_sites:
    
    ax = axd[site]
    
    ax.set_title(site)
    
    for depth in ['surf', 'deep']:
        
        if depth == 'surf':
            
            color = 'blue'
            
        else:
            
            color = 'red'
        
        plot_df = odf_use[(odf_use['site'] == site) & (odf_use['surf_deep'] == depth)]
        
        lags = np.arange(1,24)
  
        R = np.empty(len(lags))

        for i, k in enumerate(lags):
            R[i] = crosscorr(plot_df['mean_va'], plot_df['val'], lag=lags[i])
            
        ax.plot(lags, R,'-.', label=depth, color = color)

    ax.legend()

    ax.set_ylabel('crosscorr (skagit-v-val), R')
    ax.set_xlabel('Lag, months')

    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
    
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/ecology_skagit_crosscorr.png', dpi=500)            

# %%

mosaic = [['skagit'], ['KSBP01']]

fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=True, layout='constrained', figsize=(6,6), gridspec_kw=dict(wspace=0.1, hspace=0.1))

ax = axd['skagit']

plot_df = odf_use.groupby(['year', 'month']).first().reset_index()

lags = np.arange(1,int(len(plot_df)/4))

R = np.empty(len(lags))

for i, k in enumerate(lags):
    R[i] = plot_df['mean_va'].autocorr(lag=lags[i])
    
ax.plot(lags, R,'-.', label='skagit', color = 'gray')

ax.legend()

ax.set_ylabel('Autocorrelation, R')
ax.set_xlabel('Lag, months')

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)



for site in ['KSBP01']:
    
    ax = axd[site]
    
    ax.set_title(site)
    
    for depth in ['surf', 'deep']:
        
        if depth == 'surf':
            
            color = 'blue'
            
        else:
            
            color = 'red'
        
        plot_df = odf_use[(odf_use['site'] == site) & (odf_use['surf_deep'] == depth)]
        
        lags = np.arange(1,int(len(plot_df)/4))
 
        R = np.empty(len(lags))

        for i, k in enumerate(lags):
            R[i] = plot_df['val'].autocorr(lag=lags[i])
            
        ax.plot(lags, R,'-.', label=depth, color = color)

    ax.legend()

    ax.set_ylabel('Autocorrelation, R')
    ax.set_xlabel('Lag, months')

    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
    
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/KSBP01_skagit_autocorr.png', dpi=500)
        
# %%

mosaic = [['skagit'], ['KSBP01']]

fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=True, layout='constrained', figsize=(6,6), gridspec_kw=dict(wspace=0.1, hspace=0.1))

ax = axd['skagit']

plot_df = odf_use.groupby(['year', 'month']).first().reset_index()

lags = np.arange(1,24)

R = np.empty(len(lags))

for i, k in enumerate(lags):
    R[i] = plot_df['mean_va'].autocorr(lag=lags[i])
    
ax.plot(lags, R,'-.', label='skagit', color = 'gray')

ax.legend()

ax.set_ylabel('Autocorrelation, R')
ax.set_xlabel('Lag, months')

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)





for site in ['KSBP01']:
    
    ax = axd[site]
    
    ax.set_title(site)
    
    for depth in ['surf', 'deep']:
        
        if depth == 'surf':
            
            color = 'blue'
            
        else:
            
            color = 'red'
        
        plot_df = odf_use[(odf_use['site'] == site) & (odf_use['surf_deep'] == depth)]
        
        lags = np.arange(1,24)
  
        R = np.empty(len(lags))

        for i, k in enumerate(lags):
            R[i] = crosscorr(plot_df['mean_va'], plot_df['val'], lag=lags[i])
            
        ax.plot(lags, R,'-.', label=depth, color = color)

    ax.legend()

    ax.set_ylabel('crosscorr (skagit-v-val), R')
    ax.set_xlabel('Lag, months')

    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)
    
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/KSBP01_skagit_crosscorr.png', dpi=500)    






