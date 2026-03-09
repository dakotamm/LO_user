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

import cmocean as cmo 

from cmcrameri import cm as cmc



Ldir = Lfun.Lstart(gridname='wb1', tag='r0', ex_name='xn11b')

# %%

Ldir['ds0'] = '2017.09.05'

Ldir['ds1'] = '2017.09.18'

Ldir['list_type'] = 'lowpass'

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

sum_ubar = None

sum_vbar = None

threshold=2

M = 368

L = 272

count = 0

for fn in fn_list:
    
    ds = xr.open_dataset(fn)
    
    oxygen_mg_L = ds.oxygen*32/1000 #molar mass of O2
    
    oxygen_mg_L_np = oxygen_mg_L.isel(ocean_time = 0).to_numpy().reshape(30, M, L)

    # build 2D hypoxia mask
    mask2d = oxygen_mg_L_np.min(axis=0) <= threshold

    if accum is None:
        accum = np.zeros_like(mask2d, dtype=int)

    accum += mask2d.astype(int)
    
    ubar = ds.ubar[0,:,:].values
    
    vbar = ds.vbar[0,:,:].values
    
    # mask3 = np.broadcast_to(mask_u[None, :, :], u.shape)

    # u_poly = np.where(mask3, u, np.nan)
    
    ubar_poly = ubar.copy()
    
    vbar_poly = vbar.copy()
            
    
   # u_poly_depth_avg = np.mean(u_poly, axis=0)
        
    if sum_ubar is None:
        sum_ubar = ubar_poly.copy()
    else:
        sum_ubar += ubar_poly
        
    if sum_vbar is None:
        sum_vbar = vbar_poly.copy()
    else:
        sum_vbar += vbar_poly
    
    count += 1
    
mean_ubar_2d = sum_ubar / count

mean_vbar_2d = sum_vbar / count
        
accum = accum.astype(float)
    
accum[accum == 0] = np.nan
    
# %%
sta_dict = {
    'M1': (-122.710941, 48.225977),
    'M2': (-122.656317, 48.241464),
    'M3': (-122.652904, 48.234749),
    'M4': (-122.649333, 48.229201),
    'M5': (-122.582462, 48.245981)}

def generate_lajolla_colors(n_colors=3):
    """
    Generates a list of n colors from the 'batlow' colormap.

    Args:
        n_colors (int): The number of colors to generate.

    Returns:
        list: A list of RGBA color tuples.
    """
    # Access the batlow colormap from cmcrameri
    cmap = cmc.lajolla
    
    # Generate evenly spaced values from 0 to 1 to sample the colormap
    # We use np.linspace to get n_colors evenly spaced points
    sample_points = np.linspace(0, 1, n_colors)

    # Get the colors from the colormap
    # cmap returns RGBA values as a numpy array
    colors = cmap(sample_points)
    
    return colors.tolist()

# Generate the three batlow colors
three_colors = generate_lajolla_colors(3)

# %%

from matplotlib.path import Path

def points_in_polygon_mask(lon2d, lat2d, poly_lonlat):
    """
    lon2d, lat2d: 2D arrays (same shape), e.g. ds.lon_rho, ds.lat_rho
    poly_lonlat: sequence of (lon, lat) vertices, e.g. [(lon0,lat0), ...]
    Returns: mask2d (bool) True where grid point is inside polygon
    """
    poly = Path(np.asarray(poly_lonlat))

    pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    inside = poly.contains_points(pts)          # excludes boundary by default
    # inside = poly.contains_points(pts, radius=1e-12)  # include boundary-ish

    return inside.reshape(lon2d.shape)

sect_path = Ldir['LOo'] / 'section_lines'
sect = 'pc.p'

sect_fn = sect_path / sect

pdict = pickle.load(open(sect_fn, 'rb'))

# # --- example ---
# # lon2d = ds["lon_rho"].values
# # lat2d = ds["lat_rho"].values
# # poly_lonlat = [(-122.6, 47.4), (-122.3, 47.4), (-122.3, 47.7), (-122.6, 47.7)]
mask_rho = points_in_polygon_mask(lon_rho, lat_rho, pdict)

mask_u = points_in_polygon_mask(lon_u, lat_u, pdict)

mask_v = points_in_polygon_mask(lon_v, lat_v, pdict)



# # indices (iy, ix) of points inside:
iy_rho, ix_rho = np.where(mask_rho)
iy_u, ix_u = np.where(mask_u)
iy_v, ix_v = np.where(mask_v)


# %%

outdir = Ldir['LOo'] / 'DM_outs'

Lfun.make_dir(outdir)

fig, ax = plt.subplots(figsize=(10,3))

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

cs = ax.pcolormesh(plon,plat,accum,vmin = 1,vmax=30, cmap=cmc.lajolla)


# ADD VELOCITY VECTORS # FROM PFUN ADAPTED 20260210
nngrid = 200
v_scl = 1
v_leglen=0.1
center=(.8,.05)
# set masked values to 0
mean_ubar_2d[np.isnan(mean_ubar_2d)] = 0
mean_vbar_2d[np.isnan(mean_vbar_2d)] = 0
# mean_ubar_2d[~mask_u] = 0
# mean_vbar_2d[~mask_v] = 0
# create regular grid
aaa = ax.axis()
daax = aaa[1] - aaa[0]
daay = aaa[3] - aaa[2]
axrat = np.cos(np.deg2rad(aaa[2])) * daax / daay
x = np.linspace(aaa[0], aaa[1], int(round(nngrid * axrat)))
y = np.linspace(aaa[2], aaa[3], int(nngrid))
xx, yy = np.meshgrid(x, y)
# interpolate to regular grid
uu = zfun.interp2(xx, yy, lon_u, lat_u, mean_ubar_2d)
vv = zfun.interp2(xx, yy, lon_v, lat_v, mean_vbar_2d)

mask = (uu != 0) & (uu < 0.05)
# plot velocity vectors
Q = ax.quiver(xx[mask], yy[mask], uu[mask], vv[mask],   
              scale=v_scl, scale_units='width', color='red', units='width')
plt.quiverkey(Q, .15, .8, v_leglen, str(v_leglen)+' $ms^{-1}$', angle=20, color='black')

c = 0 
for m in ['M1', 'M3', 'M5']:
    ax.scatter(sta_dict[m][0], sta_dict[m][1], color = three_colors[c], edgecolors = 'k')
    c+=1
    
pfun.add_coast(ax)

pfun.dar(ax)


ax.set_ylim(48.21, 48.26)
ax.set_xlim(-122.74, -122.56)

ax.set_xticks([-122.7, -122.6], ['-122.7','-122.6'])

ax.set_yticks([48.22,48.25], ['48.22','48.25'])



fig.colorbar(cs, ax=ax, label = 'Hypoxic Days')

#ax.set_title('2017 Hypoxic Days')


plt.savefig(outdir/ (Ldir['gtagex'] + '_' + Ldir['list_type'] + '_'
    + Ldir['ds0'] + '-' + Ldir['ds1'] + '_hypoxic_days_depth_avg_u_moorings' + test +'.png'), dpi=500, transparent = True, bbox_inches='tight')