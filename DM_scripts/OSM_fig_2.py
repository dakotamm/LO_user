#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 14:29:22 2026

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

#import seaborn as sns

import scipy.stats as stats

import D_functions as dfun

import pickle

import math

from scipy.interpolate import interp1d

import gsw

import matplotlib.path as mpth

import matplotlib.patches as patches

import cmocean

from cmcrameri import cm



Ldir = Lfun.Lstart(gridname='wb1', tag='r0', ex_name='xn11b')

# %%

Ldir['ds0'] = '2017.01.01'

Ldir['ds1'] = '2017.12.31'

Ldir['list_type'] = 'daily'

Ldir['his_num'] = 2

fn_list = Lfun.get_fn_list(Ldir['list_type'], Ldir,
    Ldir['ds0'], Ldir['ds1'], his_num=Ldir['his_num'])

# %%

fng = Ldir['grid'] / 'grid.nc'
dsg = xr.open_dataset(fng)
lon_rho = dsg.lon_rho.values
lat_rho = dsg.lat_rho.values
lon_u = dsg.lon_u.values
lat_u = dsg.lat_u.values
lon_v = dsg.lon_v.values
lat_v = dsg.lat_v.values
m = dsg.mask_rho.values
depths = dsg.h.values
depths[m==0] = np.nan


lon_1D = lon_rho[0,:]

lat_1D = lat_rho[:,0]

# weird, to fix

mask_rho = np.transpose(dsg.mask_rho.values)
zm = -depths.copy()
zm[np.transpose(mask_rho) == 0] = np.nan
zm[np.transpose(mask_rho) != 0] = -1

zm_inverse = zm.copy()

zm_inverse[np.isnan(zm)] = -1

zm_inverse[zm==-1] = np.nan


plon, plat = pfun.get_plon_plat(lon_rho,lat_rho)

plon_u, plat_u = pfun.get_plon_plat(lon_u,lat_u)

plon_v, plat_v = pfun.get_plon_plat(lon_v,lat_v)

# %%

# outdir0 = Ldir['LOo'] / 'DM_fields'
# Lfun.make_dir(outdir0)

if '_mac' in Ldir['lo_env']:
    
    test = '_TEST'

else:
    
    test = ''
    
    

accum = None

threshold=2

M = 368

L = 272

for fn in fn_list:
    
    ds = xr.open_dataset(fn)
    
    oxygen_mg_L = ds.oxygen*32/1000 #molar mass of O2
    
    oxygen_mg_L_np = oxygen_mg_L.isel(ocean_time = 0).to_numpy().reshape(30, M, L)

    # build 2D hypoxia mask
    mask2d = oxygen_mg_L_np.min(axis=0) <= threshold

    if accum is None:
        accum = np.zeros_like(mask2d, dtype=int)

    accum += mask2d.astype(int)
    
accum = accum.astype(float)
    
accum[accum == 0] = np.nan
    
    
outdir = Ldir['LOo'] / 'DM_outs'

Lfun.make_dir(outdir)

fig, ax = plt.subplots(figsize=(6,6))

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

cs = ax.pcolormesh(plon,plat,accum,vmin = 0,vmax=30, cmap=cm.lajolla)



pfun.add_coast(ax)

pfun.dar(ax)

#pfun.add_bathy_contours(ax, ds, depth_levs = [20], txt=True)
# ax.set_ylim(48.2, 48.3)
# ax.set_xlim(-122.740, -122.510)

ax.set_ylim(48.21, 48.25)
ax.set_xlim(-122.740, -122.64)

ax.set_xticks([-122.73, -122.69, -122.65], ['-122.72','-122.68', '-122.64'])

ax.set_yticks([48.21, 48.23, 48.25], ['48.21','48.23', '48.25'])



fig.colorbar(cs, ax=ax, shrink=0.6, label = 'Hypoxic Days')

#ax.set_title('2017 Hypoxic Days')


plt.savefig(outdir/ (Ldir['gtagex'] + '_' + Ldir['list_type'] + '_'
    + Ldir['ds0'] + '-' + Ldir['ds1'] + '_hypoxic_days' + test +'.png'), dpi=500, transparent = True, bbox_inches='tight')