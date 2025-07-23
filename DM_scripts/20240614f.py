#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:04:50 2024

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

df_2022 = pd.read_pickle('/Users/dakotamascarenas/LPM_output/obsmod/multi_ctd_2022.p')

df_2023 = pd.read_pickle('/Users/dakotamascarenas/LPM_output/obsmod/multi_ctd_2023.p')

df_2024 = pd.read_pickle('/Users/dakotamascarenas/LPM_output/obsmod/multi_ctd_2024.p')

# %%

obs_df = df_2022['obs'].copy()

temp = df_2023['obs'].copy()

temp['cid'] = temp['cid'] + obs_df['cid'].max()

obs_df = pd.concat([obs_df, temp])

temp = df_2024['obs'].copy()

temp['cid'] = temp['cid'] + obs_df['cid'].max()

obs_df = pd.concat([obs_df, temp])

# %%

mod_df_0 = df_2022['cas6_v0_live'].copy()

temp = df_2023['cas6_v0_live'].copy()

temp['cid'] = temp['cid'] + mod_df_0['cid'].max()

mod_df_0 = pd.concat([mod_df_0, temp])

temp = df_2024['cas6_v0_live'].copy()

temp['cid'] = temp['cid'] + mod_df_0['cid'].max()

mod_df_0 = pd.concat([mod_df_0, temp])


# %%

mod_df_1 = df_2022['cas6_v1_live'].copy()

temp = df_2023['cas6_v1_live'].copy()

temp['cid'] = temp['cid'] + mod_df_1['cid'].max()

mod_df_1 = pd.concat([mod_df_1, temp])

temp = df_2024['cas6_v1_live'].copy()

temp['cid'] = temp['cid'] + mod_df_1['cid'].max()

mod_df_1 = pd.concat([mod_df_1, temp])

# %%

obs_df = obs_df.assign(
    ix=(lambda x: x['lon'].apply(lambda x: zfun.find_nearest_ind(lon_1D, x))),
    iy=(lambda x: x['lat'].apply(lambda x: zfun.find_nearest_ind(lat_1D, x)))
)

mod_df_0 = mod_df_0.assign(
    ix=(lambda x: x['lon'].apply(lambda x: zfun.find_nearest_ind(lon_1D, x))),
    iy=(lambda x: x['lat'].apply(lambda x: zfun.find_nearest_ind(lat_1D, x)))
)

mod_df_1 = mod_df_1.assign(
    ix=(lambda x: x['lon'].apply(lambda x: zfun.find_nearest_ind(lon_1D, x))),
    iy=(lambda x: x['lat'].apply(lambda x: zfun.find_nearest_ind(lat_1D, x)))
)


# %%

obs_df['h'] = obs_df.apply(lambda x: -depths[x['iy'], x['ix']], axis=1)

mod_df_0['h'] = mod_df_0.apply(lambda x: -depths[x['iy'], x['ix']], axis=1)

mod_df_1['h'] = mod_df_1.apply(lambda x: -depths[x['iy'], x['ix']], axis=1)


# %%

# obs_df['yearday'] = obs_df['datetime'].dt.dayofyear

# mod_df_0['yearday'] = mod_df_0['datetime'].dt.dayofyear

# mod_df_1['yearday'] = mod_df_1['datetime'].dt.dayofyear


# %%

# obs_df = obs_df[obs_df['val'] >0]

# mod_df_0 = mod_df_0[mod_df_0['val'] >0]

# mod_df_1 = mod_df_1[mod_df_1['val'] >0]

# %%

obs_df = (obs_df
                  .assign(
                      datetime=(lambda x: pd.to_datetime(x['time'], utc=True)),
                      # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                      # month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                      # season=(lambda x: pd.cut(x['month'],
                      #                         bins=[0,3,6,9,12],
                      #                         labels=['winter', 'spring', 'summer', 'fall'])),
                      DO_mg_L=(lambda x: x['DO (uM)']*32/1000),
                      date_ordinal=(lambda x: x['datetime'].apply(lambda x: x.toordinal())) #,
                      # segment=(lambda x: key),
                      # decade=(lambda x: pd.cut(x['year'],
                      #                         bins=[1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2019, 2029],
                      #                         labels=['1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'], right=True))
                          )
                  )

mod_df_0 = (mod_df_0
                  .assign(
                      datetime=(lambda x: pd.to_datetime(x['time'], utc=True)),
                      # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                      # month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                      # season=(lambda x: pd.cut(x['month'],
                      #                         bins=[0,3,6,9,12],
                      #                         labels=['winter', 'spring', 'summer', 'fall'])),
                      DO_mg_L=(lambda x: x['DO (uM)']*32/1000),
                      date_ordinal=(lambda x: x['datetime'].apply(lambda x: x.toordinal())) #,
                      # segment=(lambda x: key),
                      # decade=(lambda x: pd.cut(x['year'],
                      #                         bins=[1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2019, 2029],
                      #                         labels=['1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'], right=True))
                          )
                  )

mod_df_1 = (mod_df_1
                  .assign(
                      datetime=(lambda x: pd.to_datetime(x['time'], utc=True)),
                      # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                      # month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                      # season=(lambda x: pd.cut(x['month'],
                      #                         bins=[0,3,6,9,12],
                      #                         labels=['winter', 'spring', 'summer', 'fall'])),
                      DO_mg_L=(lambda x: x['DO (uM)']*32/1000),
                      date_ordinal=(lambda x: x['datetime'].apply(lambda x: x.toordinal())) #,
                      # segment=(lambda x: key),
                      # decade=(lambda x: pd.cut(x['year'],
                      #                         bins=[1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2019, 2029],
                      #                         labels=['1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'], right=True))
                          )
                  )


# %%

obs_df_depth_mean = obs_df[obs_df['z'] < obs_df['h']*.08].groupby(['cid', 'name']).mean(numeric_only=True).reset_index()

mod_df_0_depth_mean = mod_df_0[mod_df_0['z'] < mod_df_0['h']*.08].groupby(['cid', 'name']).mean(numeric_only=True).reset_index()

mod_df_1_depth_mean = mod_df_1[mod_df_1['z'] < mod_df_1['h']*.08].groupby(['cid', 'name']).mean(numeric_only=True).reset_index()

# %%

obs_df_depth_mean = (obs_df_depth_mean
                  # .drop(columns=['date_ordinal_std'])
                  #.rename(columns={'date_ordinal_mean':'date_ordinal'})
                  #.reset_index() 
                  .dropna()
                  .assign(
                          #segment=(lambda x: key),
                          # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                          # month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                          # season=(lambda x: pd.cut(x['month'],
                          #                          bins=[0,3,6,9,12],
                          #                          labels=['winter', 'spring', 'summer', 'fall'])),
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

mod_df_0_depth_mean = (mod_df_0_depth_mean
                  # .drop(columns=['date_ordinal_std'])
                  #.rename(columns={'date_ordinal_mean':'date_ordinal'})
                  #.reset_index() 
                  .dropna()
                  .assign(
                          #segment=(lambda x: key),
                          # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                          # month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                          # season=(lambda x: pd.cut(x['month'],
                          #                          bins=[0,3,6,9,12],
                          #                          labels=['winter', 'spring', 'summer', 'fall'])),
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

mod_df_1_depth_mean = (mod_df_1_depth_mean
                  # .drop(columns=['date_ordinal_std'])
                  #.rename(columns={'date_ordinal_mean':'date_ordinal'})
                  #.reset_index() 
                  .dropna()
                  .assign(
                          #segment=(lambda x: key),
                          # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                          # month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                          # season=(lambda x: pd.cut(x['month'],
                          #                          bins=[0,3,6,9,12],
                          #                          labels=['winter', 'spring', 'summer', 'fall'])),
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )



# %%

mosaic = [['map', 'PENNCOVEWEST', 'PENNCOVEWEST'], ['map', 'PENNCOVEENT', 'PENNCOVEENT'], ['map', 'SARATOGARP', 'SARATOGARP']]

colors = {'PENNCOVEWEST': 'orange', 'PENNCOVEENT': 'fuchsia', 'SARATOGARP': 'purple'}



fig, ax = plt.subplot_mosaic(mosaic, figsize=(10,6), layout='constrained')

plot_df_map = obs_df[obs_df['name'].isin(['PENNCOVEWEST','PENNCOVEENT', 'SARATOGARP'])].groupby('name').first().reset_index()

plot_df_map['Site'] = plot_df_map['name']

sns.scatterplot(data=plot_df_map, x='lon', y='lat', hue='name', ax = ax['map'], s = 100, palette=colors)

ax['map'].autoscale(enable=False)

ax['map'].legend(loc='upper left')

pfun.add_coast(ax['map'])

pfun.dar(ax['map'])

ax['map'].set_xlim(-122.9, -122.3)

ax['map'].set_ylim(47.9,48.5)

ax['map'].set_xlabel('Longitude')

ax['map'].set_ylabel('Latitude')



for site in ['PENNCOVEWEST','PENNCOVEENT', 'SARATOGARP']:
    
    plot_df_obs = obs_df_depth_mean[obs_df_depth_mean['name'] == site]
    
    plot_df_mod_0 = mod_df_0_depth_mean[mod_df_0_depth_mean['name'] == site]
    
    plot_df_mod_1 = mod_df_1_depth_mean[mod_df_1_depth_mean['name'] == site]

    ax[site].scatter(plot_df_obs['datetime'], plot_df_obs['DO_mg_L'], color ='black', label= site + ' observations')
    
    ax[site].scatter(plot_df_mod_0['datetime'], plot_df_mod_0['DO_mg_L'], color =colors[site], label=site + ' model output', alpha=0.7)
    
    ax[site].scatter(plot_df_mod_1['datetime'], plot_df_mod_1['DO_mg_L'], color =colors[site], alpha=0.7)
    
    ax[site].legend()
    
    ax[site].grid(color = 'lightgray', linestyle = '--', alpha=0.5)

    ax[site].set_ylabel('Deep DO [mg/L]')

    ax[site].set_xlabel('Date')

    ax[site].axhspan(0,2, color = 'lightgray', alpha = 0.2)

    ax[site].set_ylim(0,18)

    
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_obs_mod_DO.png', bbox_inches='tight', dpi=500, transparent=True)





