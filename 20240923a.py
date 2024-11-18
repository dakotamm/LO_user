#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:26:50 2024

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

poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))

path_dict = dfun.getPathDict(Ldir, poly_list)

# %%

long_site_list = poly_list

# %%




mosaic = [['map_source', ' ', ' ', ' '], ['map_source', '', '', '']] #, ['map_source', '.', '.'],]

fig, axd = plt.subplot_mosaic(mosaic, figsize=(11,5), layout='constrained')


ax = axd['map_source']

ax.set_xlim(X[i1],-121.4)#X[i2]) # Salish Sea
ax.set_ylim(Y[j1],Y[j2]) # Salish Sea
        
ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-100, vmax=0, cmap = 'gray', zorder=-5)

pfun.add_coast(ax)

pfun.dar(ax)

for site in ['point_jefferson']:
    
    path = path_dict[site]
    
        
    if site in ['point_jefferson', 'near_seattle_offshore']:
    

        patch = patches.PathPatch(path, facecolor='#ff4040', edgecolor='white', zorder=1)#, label='>60-year history')
        
    else:
        
        patch = patches.PathPatch(path, facecolor='#4565e8', edgecolor='white', zorder=1)
        
    ax.add_patch(patch)

axd[' '].set_ylabel('Annual Cast Count')

axd[''].set_ylabel('Annual Cast Count')

axd[''].set_ylim(0,1300)

axd[' '].set_ylim(-300,0)





ax.set_xlim(-123.2, -122.1) 

ax.set_ylim(47,48.5)


ax.set_xlabel('')

ax.set_ylabel('')

ax.tick_params(axis='x', labelrotation=45)

    
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pj_PECS_PRESENT.png', bbox_inches='tight', dpi=500,transparent=True)
