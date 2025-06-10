#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:51:27 2025

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
lon_rho = dsg.lon_rho.values
lat_rho = dsg.lat_rho.values #these are cell centers anyway...

lon_psi = dsg.lon_psi.values
lat_psi = dsg.lat_psi.values #these are cell edges...

m = dsg.mask_rho.values
xp, yp = pfun.get_plon_plat(lon_rho,lat_rho)
depths = dsg.h.values
depths[m==0] = 0 #np.nan #set to 0 on landmask

lon_rho_1D = lon_rho[0,:]

lat_rho_1D = lat_rho[:,0]

lon_psi_1D = lon_psi[0,:]

lat_psi_1D = lat_psi[:,0]

# weird, to fix

mask_rho = np.transpose(dsg.mask_rho.values)
zm = -depths.copy()
zm[np.transpose(mask_rho) == 0] = np.nan
zm[np.transpose(mask_rho) != 0] = -1

zm_inverse = zm.copy()

zm_inverse[np.isnan(zm)] = -1

zm_inverse[zm==-1] = np.nan


X = lon_rho[0,:] # grid cell X values
Y = lat_rho[:,0] # grid cell Y values

plon, plat = pfun.get_plon_plat(lon_rho,lat_rho)


j1 = 570
j2 = 1170
i1 = 220
i2 = 652




#poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson'] # 5 sites + 4 basins

poly_list = ['ps']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['kc_whidbey'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_rho_1D, lat_rho_1D, depths, lon_rho, lat_rho, poly_list, path_dict, basin_list)

# %%

pc_sites = ['PENNCOVEWEST', 'PENNCOVEENT', 'SARATOGARP']

cast_location_lat = odf[odf['name'].isin(pc_sites)].groupby('name').first().reset_index().dropna()['lat'].to_numpy()

cast_location_lon = odf[odf['name'].isin(pc_sites)].groupby('name').first().reset_index().dropna()['lon'].to_numpy()





# %%


fig, axd= plt.subplot_mosaic(mosaic = [['map','map', 'map'],['PENNCOVEWEST',  'PENNCOVEENT', 'SARATOGARP']], figsize=(14,9),layout='constrained')

axd['map'].scatter(cast_location_lon, cast_location_lat, s=50, color='k')

 
axd['map'].set_xlim(lon_rho_1D[553], lon_rho_1D[586])
 
axd['map'].set_ylim(lat_rho_1D[878], lat_rho_1D[888]) 


pfun.dar(axd['map'])  
  
pfun.add_coast(axd['map'])



for site in pc_sites:
    ax = axd[site]
    
    for season in odf['season'].unique():
        if season == 'loDO':
            color = 'red'
            label = 'Aug-Nov (Low DO)'
        elif season == 'winter':
            color = 'blue'
            label = 'Dec-Mar (Winter)'
        elif season == 'grow':
            color = 'gold'
            label = 'Apr-Jul (Spring Bloom)'

        season_df = odf[
            (odf['name'] == site) & 
            (odf['var'] == 'DO_mg_L') & 
            (odf['season'] == season) 
        ]
        
        for i, (cid, group) in enumerate(season_df.groupby('cid')):
            group = group.sort_values('z') 
            current_label = label if i == 0 else None 

            # Outline (drawn first)
            ax.plot(group['val'], group['z'], color='black', linewidth=3.5, alpha=0.5)

            # Colored line (drawn over the outline)
            ax.plot(group['val'], group['z'], color=color, linewidth=2, label=current_label)

    ax.grid(color='lightgray', linestyle='--', alpha=0.3)
    ax.axvspan(0, 2, color='lightgray', alpha=0.3, zorder=0)
    ax.set_ylabel('Depth [m]', wrap=True)
    ax.set_xlabel('DO [mg/L]')
    ax.set_xlim(0, 19)
    ax.set_ylim(top=0)

    

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_DO_profiles.png', bbox_inches='tight', dpi=500, transparent=True)


# %%

fig, axd= plt.subplot_mosaic(mosaic = [['map','map', 'map'],['PENNCOVEWEST',  'PENNCOVEENT', 'SARATOGARP']], figsize=(16,9))

axd['map'].scatter(cast_location_lon, cast_location_lat, s=50, color='k')

 
axd['map'].set_xlim(lon_rho_1D[553], lon_rho_1D[586])
 
axd['map'].set_ylim(lat_rho_1D[878], lat_rho_1D[888]) 


pfun.dar(axd['map'])  
  
pfun.add_coast(axd['map'])



for site in pc_sites:
    ax = axd[site]
    
    for season in odf['season'].unique():
        if season == 'loDO':
            color = 'red'
            label = 'Aug-Nov (Low DO)'
        elif season == 'winter':
            color = 'blue'
            label = 'Dec-Mar (Winter)'
        elif season == 'grow':
            color = 'gold'
            label = 'Apr-Jul (Spring Bloom)'

        season_df = odf[
            (odf['name'] == site) & 
            (odf['var'] == 'DO_mg_L') & 
            (odf['season'] == season) &
            (odf['month'] == 11) &
            (odf['year'] == 2023)
        ]
        
        for i, (cid, group) in enumerate(season_df.groupby('cid')):
            group = group.sort_values('z') 
            current_label = label if i == 0 else None 

            # Outline (drawn first)
            ax.plot(group['val'], group['z'], color='black', linewidth=3.5, alpha=0.5)

            # Colored line (drawn over the outline)
            ax.plot(group['val'], group['z'], color=color, linewidth=2, label=current_label)

    ax.grid(color='lightgray', linestyle='--', alpha=0.3)
    ax.axvspan(0, 2, color='lightgray', alpha=0.3, zorder=0)
    ax.set_ylabel('Depth [m]', wrap=True)
    ax.set_xlabel('DO [mg/L]')
    ax.set_xlim(0, 19)
    ax.set_ylim(top=0)

    

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_nov_2023_DO_profiles.png', bbox_inches='tight', dpi=500, transparent=False)

    
# %%

fig, axd= plt.subplot_mosaic(mosaic = [['map','map', 'map'],['PENNCOVEWEST',  'PENNCOVEENT', 'SARATOGARP']], figsize=(16,9))

axd['map'].scatter(cast_location_lon, cast_location_lat, s=50, color='k')

 
axd['map'].set_xlim(lon_rho_1D[553], lon_rho_1D[586])
 
axd['map'].set_ylim(lat_rho_1D[878], lat_rho_1D[888]) 


pfun.dar(axd['map'])  
  
pfun.add_coast(axd['map'])



for site in pc_sites:
    ax = axd[site]
    
    for season in odf['season'].unique():
        if season == 'loDO':
            color = 'red'
            label = 'Aug-Nov (Low DO)'
        elif season == 'winter':
            color = 'blue'
            label = 'Dec-Mar (Winter)'
        elif season == 'grow':
            color = 'gold'
            label = 'Apr-Jul (Spring Bloom)'

        season_df = odf[
            (odf['name'] == site) & 
            (odf['var'] == 'DO_mg_L') & 
            (odf['season'] == season) &
            (odf['month'] == 10) &
            (odf['year'] == 2023)
        ]
        
        for i, (cid, group) in enumerate(season_df.groupby('cid')):
            group = group.sort_values('z') 
            current_label = label if i == 0 else None 

            # Outline (drawn first)
            ax.plot(group['val'], group['z'], color='black', linewidth=3.5, alpha=0.5)

            # Colored line (drawn over the outline)
            ax.plot(group['val'], group['z'], color=color, linewidth=2, label=current_label)

    ax.grid(color='lightgray', linestyle='--', alpha=0.3)
    ax.axvspan(0, 2, color='lightgray', alpha=0.3, zorder=0)
    ax.set_ylabel('Depth [m]', wrap=True)
    ax.set_xlabel('DO [mg/L]')
    ax.set_xlim(0, 19)
    ax.set_ylim(top=0)

    

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_oct_2023_DO_profiles.png', bbox_inches='tight', dpi=500, transparent=False)    
    
# %%

fig, axd= plt.subplot_mosaic(mosaic = [['map','map', 'map'],['PENNCOVEWEST',  'PENNCOVEENT', 'SARATOGARP']], figsize=(16,9))

axd['map'].scatter(cast_location_lon, cast_location_lat, s=50, color='k')

 
axd['map'].set_xlim(lon_rho_1D[553], lon_rho_1D[586])
 
axd['map'].set_ylim(lat_rho_1D[878], lat_rho_1D[888]) 


pfun.dar(axd['map'])  
  
pfun.add_coast(axd['map'])



for site in pc_sites:
    ax = axd[site]
    
    for season in odf['season'].unique():
        if season == 'loDO':
            color = 'red'
            label = 'Aug-Nov (Low DO)'
        elif season == 'winter':
            color = 'blue'
            label = 'Dec-Mar (Winter)'
        elif season == 'grow':
            color = 'gold'
            label = 'Apr-Jul (Spring Bloom)'

        season_df = odf[
            (odf['name'] == site) & 
            (odf['var'] == 'DO_mg_L') & 
            (odf['season'] == season) &
            (odf['month'] == 9) &
            (odf['year'] == 2023)
        ]
        
        for i, (cid, group) in enumerate(season_df.groupby('cid')):
            group = group.sort_values('z') 
            current_label = label if i == 0 else None 

            # Outline (drawn first)
            ax.plot(group['val'], group['z'], color='black', linewidth=3.5, alpha=0.5)

            # Colored line (drawn over the outline)
            ax.plot(group['val'], group['z'], color=color, linewidth=2, label=current_label)

    ax.grid(color='lightgray', linestyle='--', alpha=0.3)
    ax.axvspan(0, 2, color='lightgray', alpha=0.3, zorder=0)
    ax.set_ylabel('Depth [m]', wrap=True)
    ax.set_xlabel('DO [mg/L]')
    ax.set_xlim(0, 19)
    ax.set_ylim(top=0)

    

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_sep_2023_DO_profiles.png', bbox_inches='tight', dpi=500, transparent=False)    
