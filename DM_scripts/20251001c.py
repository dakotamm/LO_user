#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 17:53:47 2025

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



fig, axd = plt.subplot_mosaic([['map']], figsize=(8,4), layout='constrained', gridspec_kw=dict(wspace=0.1))

ax = axd['map']

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

sta_dict = {
    'M1': (-122.710941, 48.225977),
    'M2': (-122.656317, 48.241464),
    'M3': (-122.652904, 48.234749),
    'M4': (-122.649333, 48.229201),
    'M5': (-122.582462, 48.245981)}

df = pd.DataFrame.from_dict(sta_dict, orient='index', columns=['lon', 'lat']).reset_index()
df = df.rename(columns={'index': 'station'})


sns.scatterplot(data =df, x = 'lon', y ='lat', hue = 'station', ax =ax, palette='Set2')


pfun.add_coast(ax)

pfun.dar(ax)

ax.set_xlim(-122.8,-122.5)#X[i2]) # Salish Sea
ax.set_ylim(48.2,48.3) # Salish Sea

#ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-100, vmax=0, cmap = 'gray')

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/M1-5.png', dpi=500,transparent=True, bbox_inches='tight')
