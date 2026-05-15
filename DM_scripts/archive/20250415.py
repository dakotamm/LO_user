#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 13:11:49 2025

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




#poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson'] # 5 sites + 4 basins

poly_list = ['ps']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['kc_whidbey'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

# map

sns.reset_defaults()

fig, ax = plt.subplots()

plot_df_map = odf.groupby('name').first().reset_index().sort_values(by=['name'])

plot_df_map['Site'] = plot_df_map['name'] + ' (' + plot_df_map['h'].apply("{:.00f}".format)+ 'm)'


#ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

ax.pcolormesh(plon, plat, zm, linewidth=0.1, vmin=-1000, vmax=0, cmap = 'gray', zorder=-6, edgecolor='gray') #more white

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-10, vmax=0, cmap = 'gray', zorder=-5) #more gray

 
sns.scatterplot(data=plot_df_map, x='lon', y='lat', hue='Site', ax = ax, s = 100, alpha=0.8)

ax.autoscale(enable=False) 
 


ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
 
pfun.add_coast(ax) 

pfun.dar(ax)

ax.set_xlim(-122.9, -122.2)

ax.set_ylim(47.9,48.5)

ax.set_xlabel('')

ax.set_ylabel('')

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/kc_whidbey_sites.png', bbox_inches='tight', dpi=500, transparent=True)

# %%

#max depths + vars

fig, ax = plt.subplots()

plot_df = odf.copy().sort_values(by=['name'])

plot_df['Site'] = plot_df['name']

max_depths = plot_df.groupby('name').first().reset_index().sort_values(by=['name'])['h'].to_numpy()

sites = plot_df.groupby('name').first().reset_index().sort_values(by=['name'])['Site'].to_list()

#sns.set(font_scale=2) # Scales all font elements


for var in var_list:
    
    if var == 'CT':
        
        palette = 'coolwarm'
    
    elif var == 'SA':
    
        palette = 'crest'
        
    elif var == 'DO_mg_L':
        
        palette = 'flare'

    g = sns.relplot(data=plot_df[plot_df['var'] == var], x='datetime', y='z', hue='val', ax=ax, row='Site', kind='scatter', alpha=0.8, edgecolor='none', palette=palette)
    
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
    
    
    c=0
    for ax in g.axes.flat:
        depth = max_depths[c]
        ax.axhline(y=depth, color='gray', linestyle='--')
        
        ax.set_ylabel('Depth')
        
        ax.set_xlabel('Datetime')
        
        if var == 'DO_mg_L':
            site = sites[c]
            temp = plot_df[(plot_df['Site'] == site) & (plot_df['var'] == var) & (plot_df['val'] <= 2)]
            ax.scatter(temp['datetime'], temp['z'], color = 'red')
            
        c+=1

    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/kc_whidbey_' + var+ '_timeseries.png', bbox_inches='tight', dpi=500, transparent=True)

# %%

#^make the above a lot easier to read

# for site in odf['name'].unique():
    
#     fig, axd = plt.subplot_mosaic(mosaic = [['CT','SA','DO_mg_L']])
    
#     plot_df = odf[odf['name'] == site].copy().sort_values(by=['name'])

#     #plot_df['Site'] = plot_df['name']
    
#     max_depth = plot_df['h'].unique()[0]
    
    
#     ax = axd[var]

#     for var in var_list:
        
#         if var == 'CT':
            
#             palette = 'coolwarm'
        
#         elif var == 'SA':
        
#             palette = 'crest'
            
#         elif var == 'DO_mg_L':
            
#             palette = 'flare'

#         sns.scatterplot(data=plot_df[plot_df['var'] == var], x='datetime', y='z', hue='val', ax=ax, alpha=0.8, edgecolor='none', palette=palette)
        
#         #ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')

# %%

# just depths vs. max LO depth

plot_df = odf.copy().sort_values(by=['name'])

plot_df['Site'] = plot_df['name']

max_depths = plot_df.groupby('name').first().reset_index().sort_values(by=['name'])['h'].to_numpy()

sites = plot_df.groupby('name').first().reset_index().sort_values(by=['name'])['Site'].to_list()

g = sns.relplot(data=plot_df[plot_df['var'] == var], x='datetime', y='z', hue='Site', ax=ax, row='Site', kind='scatter', alpha=0.8, palette='husl', edgecolor='none', aspect=2, legend=False)

c=0
for ax in g.axes.flat:
    depth = max_depths[c]
    ax.axhline(y=depth, color='gray', linestyle='-')
    
    ax.axhline(y=depth*0.8, color='gray', linestyle='--')
    
    ax.set_ylabel('Depth')
    
    ax.set_xlabel('Datetime')
    
    c+=1
    
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/kc_whidbey_depths.png', bbox_inches='tight', dpi=500, transparent=True)

# %%

df_2022 = pd.read_pickle('/Users/dakotamascarenas/LPM_output/obsmod/multi_ctd_2022.p')

df_2023 = pd.read_pickle('/Users/dakotamascarenas/LPM_output/obsmod/multi_ctd_2023.p')

df_2024 = pd.read_pickle('/Users/dakotamascarenas/LPM_output/obsmod/multi_ctd_2024.p')



obs_df = df_2022['obs'].copy()

temp = df_2023['obs'].copy()

temp['cid'] = temp['cid'] + obs_df['cid'].max()

obs_df = pd.concat([obs_df, temp])

temp = df_2024['obs'].copy()

temp['cid'] = temp['cid'] + obs_df['cid'].max()

obs_df = pd.concat([obs_df, temp])



mod_df_0 = df_2022['cas6_v0_live'].copy()

temp = df_2023['cas6_v0_live'].copy()

temp['cid'] = temp['cid'] + mod_df_0['cid'].max()

mod_df_0 = pd.concat([mod_df_0, temp])

temp = df_2024['cas6_v0_live'].copy()

temp['cid'] = temp['cid'] + mod_df_0['cid'].max()

mod_df_0 = pd.concat([mod_df_0, temp])




mod_df_1 = df_2022['cas6_v1_live'].copy()

temp = df_2023['cas6_v1_live'].copy()

temp['cid'] = temp['cid'] + mod_df_1['cid'].max()

mod_df_1 = pd.concat([mod_df_1, temp])

temp = df_2024['cas6_v1_live'].copy()

temp['cid'] = temp['cid'] + mod_df_1['cid'].max()

mod_df_1 = pd.concat([mod_df_1, temp])


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




obs_df['h'] = obs_df.apply(lambda x: -depths[x['iy'], x['ix']], axis=1)

mod_df_0['h'] = mod_df_0.apply(lambda x: -depths[x['iy'], x['ix']], axis=1)

mod_df_1['h'] = mod_df_1.apply(lambda x: -depths[x['iy'], x['ix']], axis=1)


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



obs_df_depth_mean = obs_df[obs_df['z'] < obs_df['h']*0.8].groupby(['cid', 'name']).mean(numeric_only=True).reset_index()

mod_df_0_depth_mean = mod_df_0[mod_df_0['z'] < mod_df_0['h']*0.8].groupby(['cid', 'name']).mean(numeric_only=True).reset_index()

mod_df_1_depth_mean = mod_df_1[mod_df_1['z'] < mod_df_1['h']*0.8].groupby(['cid', 'name']).mean(numeric_only=True).reset_index()

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

plot_df = obs_df_depth_mean.copy().sort_values(by=['name'])

plot_df['Site'] = plot_df['name']


plot_df_ = mod_df_0_depth_mean.copy().sort_values(by=['name'])

plot_df_['Site'] = plot_df_['name']


plot_df__ = mod_df_1_depth_mean.copy().sort_values(by=['name'])

plot_df__['Site'] = plot_df__['name']



adds = plot_df.iloc[[0]]

adds['DO_mg_L'] = np.nan

adds['CT'] = np.nan

adds['SA'] = np.nan

adds_ = adds.copy()

adds['Site'] = 'PENNCOVECW'

adds_['Site'] = 'PSUSANBUOY'

plot_df = pd.concat([plot_df, adds, adds_]).sort_values(by=['Site'])

plot_df_ = pd.concat([plot_df_, adds, adds_]).sort_values(by=['Site'])

plot_df__ = pd.concat([plot_df__, adds, adds_]).sort_values(by=['Site'])

sites = plot_df.groupby('Site').first().reset_index().sort_values(by=['Site'])['Site'].to_list()



for var in var_list:
    
    if var == 'CT':
        
        marker = 'D'
        
    elif var == 'SA':
        
        marker = 's'

    elif var == 'DO_mg_L':
        
        marker ='o'

    g = sns.relplot(data=plot_df, x='datetime', y=var, hue='Site', ax=ax, row='Site', kind='scatter', aspect=2, marker=marker, legend=False, palette='husl')

    for ax in g.axes.flat:
        
        ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3)

    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/kc_whidbey_' + var+ '_bottom_timeseries.png', bbox_inches='tight', dpi=500, transparent=True)
