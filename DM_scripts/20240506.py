#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:32:22 2024

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

poly_list = ['ps', 'mb', 'wb', 'ss', 'hc'] #,'admiralty_sill', 'budd_inlet', 'carr_inlet_mid', 'dana_passage', 'hat_island', 'hazel_point', 'hood_canal_mouth', 'lynch_cove_mid', 'near_seattle_offshore', 'near_edmonds', 'port_susan_mid', 'saratoga_passage_north', 'saratoga_passage_mid']

odf_dict = dfun.getPolyData(Ldir, poly_list, source_list=['ecology'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))

# %%

basin_list = list(odf_dict.keys())

var_list = ['SA', 'CT', 'DO_mg_L'] #, 'NO3 (uM)', 'Chl (mg m-3)', 'NO2 (uM)', 'SiO4 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']

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
            
    odf_dict[key] = pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'source', 'otype', 'decade', 'name'],
                                          value_vars=var_list, var_name='var', value_name = 'val')
    
    
# %%

odf = pd.concat(odf_dict.values(), ignore_index=True)

# %%

odf['source_type'] = odf['source'] + '_' + odf['otype']

# %%

station_list = odf['name'].unique()


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

# max_depths_dict = dict()

# ox = lon
# oy = lat
# oxoy = np.concatenate((ox.reshape(-1,1),oy.reshape(-1,1)), axis=1)


# for poly in poly_list:

#     fnp = Ldir['LOo'] / 'section_lines' / (poly+'.p')
#     p = pd.read_pickle(fnp)
#     xx = p.x.to_numpy()
#     yy = p.y.to_numpy()
#     xxyy = np.concatenate((xx.reshape(-1,1),yy.reshape(-1,1)), axis=1)
#     path = mpth.Path(xxyy)
    
#     oisin = path.contains_points(oxoy)
    
#     this_depths = depths.flatten()[oisin]
    
#     max_depth = np.nanmax(this_depths)
    
#     max_depths_dict[poly] = max_depth.copy()
    
# # %%


# for basin in basin_list:
    
#     odf.loc[odf['segment'] == basin, 'min_segment_h'] = -max_depths_dict[basin]

    
# %%

odf_grow = odf[(odf['yearday'] > 200) & (odf['yearday'] <= 300)]

# %%

odf_grow_bottom = odf_grow[(odf_grow['z'] < 0.8*odf_grow['h'])] #& (~odf_grow['segment'].isin(['admiralty_sill', 'lynch_cove_mid', 'budd_inlet', 'dana_passage', 'ps', 'wb', 'mb', 'ss', 'hc']))]

# %%

odf_grow_bottom_mean = odf_grow_bottom.groupby(['segment','name','year','var','cid']).mean(numeric_only=True).reset_index().dropna()

# %%

odf_grow_bottom_mean = (odf_grow_bottom_mean
                  # .drop(columns=['date_ordinal_std'])
                  # .rename(columns={'date_ordinal_mean':'date_ordinal'})
                  # .reset_index() 
                  # .dropna()
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

# annual_counts = (temp0
#                      .dropna()
#                      #.set_index('datetime')
#                      .groupby(['segment','name','year','var']).agg({'cid' :lambda x: x.nunique()})
#                      .reset_index()
#                      .rename(columns={'cid':'cid_count'})
#                      )

# # %%

# odf_grow_bottom_mean = temp0.groupby(['segment', 'name', 'year','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})

# # HOOD CANAL WHOLE CAST VALUE?!?!?! what it is right now

# # %%

# odf_grow_bottom_mean.columns = odf_grow_bottom_mean.columns.to_flat_index().map('_'.join)

# odf_grow_bottom_mean = odf_grow_bottom_mean.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!

# # %%

# odf_grow_bottom_mean = (odf_grow_bottom_mean
#                   # .drop(columns=['date_ordinal_std'])
#                   .rename(columns={'date_ordinal_mean':'date_ordinal'})
#                   .reset_index() 
#                   .dropna()
#                   .assign(
#                           #segment=(lambda x: key),
#                           # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
#                           # month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
#                           # season=(lambda x: pd.cut(x['month'],
#                           #                          bins=[0,3,6,9,12],
#                           #                          labels=['winter', 'spring', 'summer', 'fall'])),
#                           datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
#                           )
#                   )

# # %%

# odf_grow_bottom_mean = pd.merge(odf_grow_bottom_mean, annual_counts, how='left', on=['segment','year','var'])

# # %%

# odf_grow_bottom_mean = odf_grow_bottom_mean[odf_grow_bottom_lc['cid_count'] >1] #redundant but fine (see note line 234)

# odf_grow_bottom_mean['val_ci95hi'] = odf_grow_bottom_mean['val_mean'] + 1.96*odf_grow_bottom_mean['val_std']/np.sqrt(odf_grow_bottom_mean['cid_count'])

# odf_grow_bottom_mean['val_ci95lo'] = odf_grow_bottom_mean['val_mean'] - 1.96*odf_grow_bottom_mean['val_std']/np.sqrt(odf_grow_bottom_mean['cid_count'])

# %%
c=0
for station in station_list:
    
    fig, ax = plt.subplot_mosaic([['map', 'SA'],['map', 'CT'], ['map', 'DO_mg_L']], layout='constrained', figsize=(15,10))
    
    plot_df_map = odf.groupby('name').first().reset_index()

    sns.scatterplot(data=plot_df_map, x='lon', y='lat', color = 'gray', ax = ax['map'], s = 100)
    
    sns.scatterplot(data=plot_df_map[plot_df_map['name'] == station], x='lon', y='lat', color = 'orange', ax = ax['map'], s = 200)

    ax['map'].autoscale(enable=False)

    pfun.add_coast(ax['map'])

    pfun.dar(ax['map'])

    ax['map'].set_xlim(-123.2, -122.1)

    ax['map'].set_ylim(47,48.5)
    
    
    var = 'SA'
        
    marker = 's'
    
    ymin = 25
    
    ymax = 35
    
    label = 'Salinity [PSU]'
    
    color = 'blue'
    
    plot_df = odf_grow_bottom_mean[(odf_grow_bottom_mean['var'] == var) & (odf_grow_bottom_mean['segment'] == 'ps') & (odf_grow_bottom_mean['name'] == station)]
    
    #plot_df_mean = plot_df.set_index('datetime').sort_index()
    
    #rolling_mean = plot_df_mean['val_mean'].rolling(window='3650D', min_periods=1).mean()
    
    sns.scatterplot(data=plot_df, x='datetime', y ='val', ax=ax[var], color=color)
    
    #rolling_mean.plot(label='Decadal Rolling Mean', ax=ax[c], color = color)
    
    # for idx in plot_df.index:
        
    #     ax['SA'].plot([plot_df.loc[idx,'datetime'], plot_df.loc[idx,'datetime']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color='lightgray', alpha =0.7)
                        
    ax[var].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax[var].set_ylabel(label)
    
    ax[var].set_ylim(ymin,ymax)
    
    ax[var].set_xlim([datetime.date(1998,1,1), datetime.date(2020,12,31)])
    
    
    
    var = 'CT'
    
    marker = '^'
    
    ymin = 8
    
    ymax = 20
    
    label = 'Temperature [deg C]'
    
    color = 'red'

    plot_df = odf_grow_bottom_mean[(odf_grow_bottom_mean['var'] == var) & (odf_grow_bottom_mean['segment'] == 'ps') & (odf_grow_bottom_mean['name'] == station)]
    
    #plot_df_mean = plot_df.set_index('datetime').sort_index()
    
    #rolling_mean = plot_df_mean['val_mean'].rolling(window='3650D', min_periods=1).mean()
    
    sns.scatterplot(data=plot_df, x='datetime', y ='val', ax=ax[var], color=color)
    
    #rolling_mean.plot(label='Decadal Rolling Mean', ax=ax[c], color = color)
    
    # for idx in plot_df.index:
        
    #     ax['SA'].plot([plot_df.loc[idx,'datetime'], plot_df.loc[idx,'datetime']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color='lightgray', alpha =0.7)
                        
    ax[var].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax[var].set_ylabel(label)
    
    ax[var].set_ylim(ymin,ymax)
    
    ax[var].set_xlim([datetime.date(1998,1,1), datetime.date(2020,12,31)])
    
    
    
    var = 'DO_mg_L'
    
    marker = 'o'
    
    ymin = 0
    
    ymax = 12
    
    color = 'black'
    
    label = 'DO [mg/L]'
    
    plot_df = odf_grow_bottom_mean[(odf_grow_bottom_mean['var'] == var) & (odf_grow_bottom_mean['segment'] == 'ps') & (odf_grow_bottom_mean['name'] == station)]
    
    #plot_df_mean = plot_df.set_index('datetime').sort_index()
    
    #rolling_mean = plot_df_mean['val_mean'].rolling(window='3650D', min_periods=1).mean()
    
    sns.scatterplot(data=plot_df, x='datetime', y ='val', ax=ax[var], color=color)
    
    #rolling_mean.plot(label='Decadal Rolling Mean', ax=ax[c], color = color)
    
    # for idx in plot_df.index:
        
    #     ax['SA'].plot([plot_df.loc[idx,'datetime'], plot_df.loc[idx,'datetime']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color='lightgray', alpha =0.7)
                        
    ax[var].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax[var].set_ylabel(label)
    
    ax[var].set_ylim(ymin,ymax)
    
    ax[var].axhspan(0,2, color = 'lightgray', alpha = 0.2)
    
    ax[var].set_xlim([datetime.date(1998,1,1), datetime.date(2020,12,31)])
    
    
    
    ax['map'].set_title(station + ' bottom grow season')
    
    #ax[].set_xlabel('Date')
                    
    #fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/grow_bottom_per_cast_' + "{:04d}".format(c) +'.png', bbox_inches='tight', dpi=500)
                                      
    c+=1
                                 
# %%

# 2010-2018 O-35 m

plot_df = odf[(odf['segment'] == 'ps') & (odf['var'] == 'DO_mg_L') & (odf['z'] >=-35) & (odf['z'] < 0) & (odf['year'] >= 2010) & (odf['year'] <= 2018)].groupby(['cid']).mean(numeric_only=True).reset_index().dropna()

fig, ax = plt.subplots(figsize=(10,5))

sns.scatterplot(data=plot_df, x='date_ordinal', y='val', color='b')

slope, intercept, rvalue, pvalue, stderr = stats.linregress(plot_df['date_ordinal'], plot_df['val'])

# # Compute the SST for x
# sst_x = np.sum( (x - np.mean(x))**2 )

# # Compute the standard error
# sigma = stderr * np.sqrt(sst_x)
# print('sigma : {}'.format(np.round(sigma,3)))

x = np.linspace(plot_df['date_ordinal'].min(), plot_df['date_ordinal'], 2) # make two x coordinates from min and max values of SLI_max
y = slope * x + intercept

ax.plot(x, y, '-r')

ax.text(0.8,0.9,'r^2 = {}'.format(np.round(rvalue**2,3)), transform = ax.transAxes)

ax.axhspan(0,2, color = 'lightgray', alpha = 0.2)

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)

ax.set_title('ecology 0-35m 2010-2018, mean per cast')

ax.set(xlabel='Date', ylabel='DO [mg/L]')

new_labels = [datetime.date.fromordinal(int(item)) for item in ax.get_xticks()]

ax.set_xticklabels(new_labels, rotation=45, horizontalalignment='right')

ax.set_ylim([0,20])


fig.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/ecology_0-35m_2010-2018.png', bbox_inches='tight', dpi=500)

# %%

# 2010-2018 35-105 m

plot_df = odf[(odf['segment'] == 'ps') & (odf['var'] == 'DO_mg_L') & (odf['z'] >=-105) & (odf['z'] < -35) & (odf['year'] >= 2010) & (odf['year'] <= 2018)].groupby(['cid']).mean(numeric_only=True).reset_index().dropna()

fig, ax = plt.subplots(figsize=(10,5))

sns.scatterplot(data=plot_df, x='date_ordinal', y='val', color='b')

slope, intercept, rvalue, pvalue, stderr = stats.linregress(plot_df['date_ordinal'], plot_df['val'])

# # Compute the SST for x
# sst_x = np.sum( (x - np.mean(x))**2 )

# # Compute the standard error
# sigma = stderr * np.sqrt(sst_x)
# print('sigma : {}'.format(np.round(sigma,3)))

x = np.linspace(plot_df['date_ordinal'].min(), plot_df['date_ordinal'], 2) # make two x coordinates from min and max values of SLI_max
y = slope * x + intercept

ax.plot(x, y, '-r')

ax.text(0.8,0.9,'r^2 = {}'.format(np.round(rvalue**2,3)), transform = ax.transAxes)

ax.axhspan(0,2, color = 'lightgray', alpha = 0.2)

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)

ax.set_title('ecology 0-35m 2010-2018, mean per cast')

ax.set(xlabel='Date', ylabel='DO [mg/L]')

new_labels = [datetime.date.fromordinal(int(item)) for item in ax.get_xticks()]

ax.set_xticklabels(new_labels, rotation=45, horizontalalignment='right')

ax.set_ylim([0,20])


fig.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/ecology_35-105m_2010-2018.png', bbox_inches='tight', dpi=500)

# %%

# all years 0-35m

plot_df = odf[(odf['segment'] == 'ps') & (odf['var'] == 'DO_mg_L') & (odf['z'] >=-35) & (odf['z'] < 0) ].groupby(['cid']).mean(numeric_only=True).reset_index().dropna()

fig, ax = plt.subplots(figsize=(10,5))

sns.scatterplot(data=plot_df, x='date_ordinal', y='val', color='b')

slope, intercept, rvalue, pvalue, stderr = stats.linregress(plot_df['date_ordinal'], plot_df['val'])

# # Compute the SST for x
# sst_x = np.sum( (x - np.mean(x))**2 )

# # Compute the standard error
# sigma = stderr * np.sqrt(sst_x)
# print('sigma : {}'.format(np.round(sigma,3)))

x = np.linspace(plot_df['date_ordinal'].min(), plot_df['date_ordinal'], 2) # make two x coordinates from min and max values of SLI_max
y = slope * x + intercept

ax.plot(x, y, '-r')

ax.text(0.8,0.9,'r^2 = {}'.format(np.round(rvalue**2,3)), transform = ax.transAxes)

ax.axhspan(0,2, color = 'lightgray', alpha = 0.2)

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)

ax.set_title('ecology 0-35m 2010-2018, mean per cast')

ax.set(xlabel='Date', ylabel='DO [mg/L]')

new_labels = [datetime.date.fromordinal(int(item)) for item in ax.get_xticks()]

ax.set_xticklabels(new_labels, rotation=45, horizontalalignment='right')

ax.set_ylim([0,20])


fig.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/ecology_0-35m_all_years.png', bbox_inches='tight', dpi=500)

# %%

# all years 35-105m

plot_df = odf[(odf['segment'] == 'ps') & (odf['var'] == 'DO_mg_L') & (odf['z'] >=-105) & (odf['z'] < -35) ].groupby(['cid']).mean(numeric_only=True).reset_index().dropna()

fig, ax = plt.subplots(figsize=(10,5))

sns.scatterplot(data=plot_df, x='date_ordinal', y='val', color='b')

slope, intercept, rvalue, pvalue, stderr = stats.linregress(plot_df['date_ordinal'], plot_df['val'])

# # Compute the SST for x
# sst_x = np.sum( (x - np.mean(x))**2 )

# # Compute the standard error
# sigma = stderr * np.sqrt(sst_x)
# print('sigma : {}'.format(np.round(sigma,3)))

x = np.linspace(plot_df['date_ordinal'].min(), plot_df['date_ordinal'], 2) # make two x coordinates from min and max values of SLI_max
y = slope * x + intercept

ax.plot(x, y, '-r')

ax.text(0.8,0.9,'r^2 = {}'.format(np.round(rvalue**2,3)), transform = ax.transAxes)

ax.axhspan(0,2, color = 'lightgray', alpha = 0.2)

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)

ax.set_title('ecology 0-35m 2010-2018, mean per cast')

ax.set(xlabel='Date', ylabel='DO [mg/L]')

new_labels = [datetime.date.fromordinal(int(item)) for item in ax.get_xticks()]

ax.set_xticklabels(new_labels, rotation=45, horizontalalignment='right')

ax.set_ylim([0,20])


fig.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/ecology_35-105m_all_years.png', bbox_inches='tight', dpi=500)

# %%




    