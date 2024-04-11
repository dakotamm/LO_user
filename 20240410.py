#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:55:01 2024

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

year_list = np.arange(2022, 2025)

source_list = ['kc_whidbey']

otype_list = ['ctd']

ii=0

for year in year_list:
    for source in source_list:
        for otype in otype_list:
            odir = Ldir['LOo'] / 'obs' / source / otype
            
            try:
                if ii == 0:
                    odf_raw = pd.read_pickle( odir / (str(year) + '.p'))
                    # if 'ecology' in source_list:
                    #     if source == 'ecology' and otype == 'bottle': #keep an eye on this for calculating confidence intervals!!!
                    #         odf['DO (uM)'] == np.nan
                    odf_raw['source'] = source
                    odf_raw['otype'] = otype
                    # print(odf.columns)
                else:
                    this_odf = pd.read_pickle( odir / (str(year) + '.p'))
                    # if 'ecology' in source_list:
                    #     if source == 'ecology' and otype == 'bottle':
                    #         this_odf['DO (uM)'] == np.nan
                    this_odf['cid'] = this_odf['cid'] + odf_raw['cid'].max() + 1
                    this_odf['source'] = source
                    this_odf['otype'] = otype
                    # print(this_odf.columns)
                    odf_raw = pd.concat((odf_raw,this_odf),ignore_index=True)
                ii += 1
            except FileNotFoundError:
                pass


var_list = ['SA', 'CT', 'NO3 (uM)', 'Chl (mg m-3)', 'NO2 (uM)', 'SiO4 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO_mg_L', 'DO (uM)']

# %%
    
odf = (odf_raw
            .assign(
                datetime=(lambda x: pd.to_datetime(x['time'])),
                year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                season=(lambda x: pd.cut(x['month'],
                                         bins=[0,3,6,9,12],
                                         labels=['winter', 'spring', 'summer', 'fall'])),
                DO_mg_L=(lambda x: x['DO (uM)']*32/1000),
                date_ordinal=(lambda x: x['datetime'].apply(lambda x: x.toordinal())),
                #segment=(lambda x: key),
                decade=(lambda x: pd.cut(x['year'],
                                         bins=[1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2019, 2029],
                                         labels=['1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'], right=True))
                    )
            )
    
for var in var_list:
    
    if var not in odf.columns:
        
        odf[var] = np.nan
        
odf = pd.melt(odf, id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'z', 'year', 'month', 'season', 'date_ordinal', 'source', 'otype', 'decade', 'name'], #segment
                                     value_vars=var_list, var_name='var', value_name = 'val')
    
# %%

odf['source_type'] = odf['source'] + '_' + odf['otype']

odf['year_month'] = odf['year'].astype(str) + odf['month'].apply('{:0>2}'.format)


# %%

plot_df_map = odf.groupby('name').first().reset_index()

fig, ax = plt.subplot_mosaic([['map']], figsize=(10,10), layout='constrained')

sns.scatterplot(data=plot_df_map, x='lon', y='lat', hue='name', ax = ax['map'], s = 100)

ax['map'].autoscale(enable=False)

pfun.add_coast(ax['map'])

pfun.dar(ax['map'])

ax['map'].set_xlim(-123.2, -122.1)

ax['map'].set_ylim(47,48.5)

ax['map'].set_title('pen cove sampling locations')

ax['map'].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 2)

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/kc_whidbey_ctd_station_names.png', dpi=500, transparent=False)

# %%

pc_stations = ['PENNCOVEWEST', 'PENNCOVECW', 'PENNCOVEENT', 'SARATOGARP']

odf_pc = odf[odf['name'].isin(pc_stations)]

# %%

plot_df_map = odf_pc.groupby('name').first().reset_index()

fig, ax = plt.subplot_mosaic([['map']], figsize=(10,10), layout='constrained')

sns.scatterplot(data=plot_df_map, x='lon', y='lat', hue='name', ax = ax['map'], s = 100)

ax['map'].autoscale(enable=False)

pfun.add_coast(ax['map'])

pfun.dar(ax['map'])

ax['map'].set_xlim(-123.2, -122.1)

ax['map'].set_ylim(47,48.5)

ax['map'].set_title('pen cove sampling locations')

ax['map'].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 2)

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/kc_pc_ctd_station_names.png', dpi=500, transparent=False)

# %%

plot_df = odf_pc[odf_pc['var'] == 'Chl (mg m-3)']

plt.rc('font', size=14)


p = sns.relplot(kind = 'scatter', data = plot_df, x = 'val', y = 'z', col = 'year_month', row='name', row_order=pc_stations, hue='season', height=6, hue_order=['winter','spring','summer','fall'], legend=False)

#p.add(sns.objects.Area(color='lightgray', alpha = 0.5, edgewidth=0), x= 2, y = 'z', orient='y')

for ax in p.axes.flat:
    #ax.axvspan(0,2, color = 'lightgray', alpha = 0.2)

    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    #ax.set_xlim(22,18)
    
    ax.set_xlabel('Chl [mg/m^3]')



plt.savefig('/Users/dakotamascarenas/Desktop/pltz/kc_pc_monthly_Chl.png', dpi=200, transparent=False)

# %%

mosaic = [['map', 'DO_min'], ['map', 'Chl_max'],['map', 'NO3_max'],['map', 'SA_min'], ['map','CT_max']]

plot_df_map = odf_pc.groupby('name').first().reset_index()


for station in pc_stations:
    
    plot_df = odf_pc[(odf_pc['name'] == station) & (odf_pc['val'] >=0)]
    
    fig, ax = plt.subplot_mosaic(mosaic, layout='constrained', figsize=(20,10))
    
    plt.rc('font', size=14)
    
    sns.scatterplot(data=plot_df_map[plot_df_map['name'] == station], x='lon', y='lat', hue='name', ax = ax['map'], s = 100, legend=False)
    
    ax['map'].autoscale(enable=False)
    
    pfun.add_coast(ax['map'])
    
    pfun.dar(ax['map'])
    
    ax['map'].set_xlim(-123.2, -122.1)
    
    ax['map'].set_ylim(47,48.5)
    
    ax['map'].set_title(station)
    
    ax['map'].set(xlabel=None)
    
    ax['map'].set(ylabel=None) 
    
    
    
    plot_df_DO = plot_df[(plot_df['var'] == 'DO_mg_L')].groupby('cid').min().reset_index()

    sns.lineplot(data=plot_df_DO, x='datetime', y = 'val', ax=ax['DO_min'], marker='o', color='k')
    
    ax['DO_min'].set_ylabel('DO MIN [mg/L]')
    
    ax['DO_min'].axhspan(0,2, color = 'lightgray', alpha = 0.2) 
    
    ax['DO_min'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax['DO_min'].set(xlabel=None)
    
    ax['DO_min'].set_ylim(0,15)
    
    
    
    plot_df_Chl = plot_df[(plot_df['var'] == 'Chl (mg m-3)')].groupby('cid').max().reset_index()

    sns.lineplot(data=plot_df_Chl, x='datetime', y = 'val', ax=ax['Chl_max'], marker='o', color='green')
    
    ax['Chl_max'].set_ylabel('Chl MAX [mg/m^3]')
        
    ax['Chl_max'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax['Chl_max'].set(xlabel=None)
    
    ax['Chl_max'].set_ylim(0,25)

    
    
    plot_df_NO3 = plot_df[(plot_df['var'] == 'NO3 (uM)')].groupby('cid').max().reset_index()

    sns.lineplot(data=plot_df_NO3, x='datetime', y = 'val', ax=ax['NO3_max'], marker='o', color='blue')
    
    ax['NO3_max'].set_ylabel('NO3 MAX [uM]')
        
    ax['NO3_max'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax['NO3_max'].set(xlabel=None)
    
    ax['NO3_max'].set_ylim(0,10)

    
    
    plot_df_SA = plot_df[(plot_df['var'] == 'SA')].groupby('cid').min().reset_index()
    
    sns.lineplot(data=plot_df_SA, x='datetime', y = 'val', ax=ax['SA_min'], marker='o', color='cyan')
    
    ax['SA_min'].set_ylabel('SA MIN [psu]')
        
    ax['SA_min'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax['SA_min'].set(xlabel=None)
    
    ax['SA_min'].set_ylim(15,30)

    
    
    plot_df_CT = plot_df[(plot_df['var'] == 'CT')].groupby('cid').max().reset_index()
    
    sns.lineplot(data=plot_df_CT, x='datetime', y = 'val', ax=ax['CT_max'], marker='o', color='purple')
    
    ax['CT_max'].set_ylabel('CT MAX [deg C]')
        
    ax['CT_max'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax['CT_max'].set(xlabel=None)
    
    ax['CT_max'].set_ylim(5,25)

    
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/kc_' + station +'.png', dpi=200, transparent=False)


