#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 13:38:28 2025

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




#poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson', 'mb', 'hc', 'ss', 'wb'] # 5 sites + 4 basins

poly_list = ['ps', 'carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_his', 'kc_whidbeyBasin', 'nceiSalish', 'kc_pointJefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf.loc[odf['source'].isin(['kc_his', 'kc_whidbeyBasin', 'kc_pointJefferson', 'kc']), 'Data Source'] = 'King County'

odf.loc[odf['source'].isin(['ecology_nc', 'ecology_his']), 'Data Source'] = 'WA Dept. of Ecology'

odf.loc[odf['source'].isin(['collias']), 'Data Source'] = 'Collias'

odf.loc[odf['source'].isin(['nceiSalish']), 'Data Source'] = 'NCEI Salish Sea'


odf['site'] = odf['segment']


# %%

color =     "#EF5E3C"   # warm orange-red ##ff4040




for site in ['point_jefferson']:

    mosaic = [['map_source', 'depth_time_series', 'depth_time_series'], ['map_source', 'count_time_series', 'count_time_series']] #, ['map_source', '.', '.'],]
    
    fig, ax = plt.subplot_mosaic(mosaic, figsize=(9,5), layout='constrained', gridspec_kw=dict(wspace=0.1))
    
    plot_df = odf[odf['site'].isin(['ps', site])].groupby(['site','cid']).first().reset_index()
    
    ax['map_source'].pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray')
    
    #sns.scatterplot(data=plot_df[plot_df['site'] == 'ps'], x='lon', y='lat', ax = ax['map_source'], color = 'gray', alpha=0.01, legend=False)
            
    # path = path_dict[site]
        
    # patch = patches.PathPatch(path, facecolor=color, edgecolor='white', zorder=1, alpha=0.5)
         
    # ax['map_source'].add_patch(patch)
    
    sns.scatterplot(data=plot_df[plot_df['site'] == site], x='lon', y='lat', ax = ax['map_source'], color = color, alpha=0.3, legend=False)
    
    pfun.add_coast(ax['map_source'])
    
    pfun.dar(ax['map_source'])
    
    ax['map_source'].set_xlim(-123.2, -122.1)
    
    ax['map_source'].set_ylim(47,48.5)
    
    #ax['map_source'].legend(loc='upper center', title ='Data Source') #, bbox_to_anchor=(0.5, -0.1), title='Data Source')
    
    ax['map_source'].set_xlabel('')
    
    ax['map_source'].set_ylabel('')
    
    #ax['map_source'].tick_params(axis='x', labelrotation=45)
    
    ax['map_source'].set_xticks([-123.0, -122.6, -122.2], ['-123.0','-122.6', '-122.2']) #['','-123.0', '', '-122.6', '', '-122.2'])

    ax['map_source'].text(0.05,0.025, 'a', transform=ax['map_source'].transAxes, fontsize=14, fontweight='bold', color = 'k')
    
    
    
    
    plot_df = (odf[odf['site'].isin(['ps', site])]
                          .groupby(['site','year']).agg({'cid' :lambda x: x.nunique()})
                          .reset_index()
                          .rename(columns={'cid':'cid_count'})
                          )
    
   # sns.scatterplot(data=plot_df[plot_df['site'] == 'ps'], x='year', y='cid', ax=ax['count_time_series'], color = 'gray', alpha=0.9, legend=False)
    
    sns.scatterplot(data=plot_df[plot_df['site'] == site], x='year', y='cid_count', ax=ax['count_time_series'], color = color, alpha=0.9, legend = False)

    ax['count_time_series'].set_xlabel('')
    
    ax['count_time_series'].set_ylabel('Annual Cast Count')
    
    #ax['count_time_series'].set_ylim(0,1300)
    
    ax['count_time_series'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    #ax['count_time_series'].legend(loc='upper left', title ='Data Source') #, bbox_to_anchor=(0.5, -0.1), title='Data Source')
    
    
    ax['count_time_series'].text(0.025,0.075, 'c', transform=ax['count_time_series'].transAxes, fontsize=14, fontweight='bold', color = 'k')

    
    
    
    plot_df_ = odf[odf['site'].isin(['ps', site])].groupby(['site','year', 'cid']).min().reset_index()
    
    plot_df = plot_df_.groupby(['site', 'year']).mean(numeric_only=True).reset_index()
    
    #plot_df.loc[plot_df['site'] == 'ps', 'label']
    
   # sns.scatterplot(data=plot_df[plot_df['site'] == 'ps'], x='year', y='z', ax=ax['depth_time_series'],  color='gray', legend=False) #, label='Puget Sound Annual Average')
    
   # sns.scatterplot(data=plot_df[plot_df['site'] == site], x='year', y='z', ax=ax['depth_time_series'],  color='#ff4040', legend=False)
    
    sns.scatterplot(data=plot_df_[plot_df_['site'] == site], x='year', y='z', ax=ax['depth_time_series'],  color=color, legend=False, alpha = 0.1) #, label='Point Jefferson (Per Cast)')


    
    ax['depth_time_series'].set_xlabel('')
    
    ax['depth_time_series'].set_ylabel('Cast Depth [m]')
    
    ax['depth_time_series'].set_ylim(-300,0)
    
    ax['depth_time_series'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    #ax['depth_time_series'].legend(loc='upper left') #, title ='Data Source') #, bbox_to_anchor=(0.5, -0.1), title='Data Source')

    ax['depth_time_series'].text(0.025,0.075, 'b', transform=ax['depth_time_series'].transAxes, fontsize=14, fontweight='bold', color = 'k')

    
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/paper_1_fig_2.png', bbox_inches='tight', dpi=500, transparent=True)
    