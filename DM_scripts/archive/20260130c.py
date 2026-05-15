#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 10:36:26 2026

@author: dakotamascarenas
"""

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from matplotlib.path import Path
import gsw
from cmcrameri import cm as cmc

from cmocean import cm as cmo# have to import after matplotlib to work on remote machine

# %%

gridname = 'wb1'

tag = 'r0'

ex_name = 'xn11b'

Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)

Ldir['list_type'] = 'hourly0'

# %%

sect_path = Ldir['LOo'] / 'section_lines'
sect = 'pc.p'

sect_fn = sect_path / sect

pdict = pickle.load(open(sect_fn, 'rb'))

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

# # --- example ---
# # lon2d = ds["lon_rho"].values
# # lat2d = ds["lat_rho"].values
# # poly_lonlat = [(-122.6, 47.4), (-122.3, 47.4), (-122.3, 47.7), (-122.6, 47.7)]
mask_rho = points_in_polygon_mask(lon_rho, lat_rho, pdict)

mask_u = points_in_polygon_mask(lon_u, lat_u, pdict)


# # indices (iy, ix) of points inside:
iy_rho, ix_rho = np.where(mask_rho)
iy_u, ix_u = np.where(mask_u)




# %% NEAP FLOOD DO

Ldir['ds0'] = '2017.09.10'

#Ldir['ds1'] = '2017.09.10'

his_num_list = [9, 10, 11, 12, 13, 14, 15]

fn_list = []

dir0 = Ldir['roms_out'] / Ldir['gtagex']

for num in his_num_list:
        
    f_string = 'f' + Ldir['ds0']
    
    num_str = str(num).zfill(2)
    
    his_str = 'ocean_his_00' + num_str + '.nc'
    
    fn = dir0 / f_string / his_str
    
    fn_list.append(fn)

sum_u_depth_avg = None
count=0

for fn in fn_list:
    ds = xr.open_dataset(fn)
    
    u = ds.u[0,:,:,:]
    
   # mask3 = np.broadcast_to(mask_u[None, :, :], u.shape)

   # u_poly = np.where(mask3, u, np.nan)
   
    u_poly = u.copy()
    
    u_poly_depth_avg = np.mean(u_poly, axis=0)
    
    if sum_u_depth_avg is None:
        sum_u_depth_avg = u_poly_depth_avg.copy()
    else:
        sum_u_depth_avg += u_poly_depth_avg
    
    count +=1
    
mean_u_depth_avg_2d = sum_u_depth_avg / count

fig, ax = plt.subplots()

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

cs = ax.pcolormesh(plon_u, plat_u, mean_u_depth_avg_2d, cmap=cmo.balance, vmin=-0.1,vmax=0.1)

# lons = [-122.63, -122.7, -122.7, -122.67, -122.650196, -122.63, -122.65, -122.67, -122.688425] #from pc field plan as of 2025/12/01
# lats = [48.247988, 48.226449, 48.232127, 48.23432, 48.240642, 48.228787, 48.230413, 48.228036, 48.222825]
# ax.scatter(lons,lats,color='k', edgecolor='white')

pfun.add_coast(ax)

pfun.dar(ax)

pfun.add_bathy_contours(ax, ds, depth_levs = [20], txt=True)
ax.set_ylim(48.21, 48.25)
ax.set_xlim(-122.740, -122.64)

fig.colorbar(cs, ax=ax)

ax.set_title('Neap Flood u [m/s] ' + 'Depth-Averaged\n' +Ldir['ds0']+ ' ' + str(his_num_list[0]).zfill(2) + '00-' + str(his_num_list[-1]).zfill(2) + '00')

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_depth_avg_u_neap_flood_' + Ldir['ds0']+ '_' + str(his_num_list[0]).zfill(2) + '00-' + str(his_num_list[-1]).zfill(2) + '00.png', dpi=500, transparent=False, bbox_inches='tight')

# %% NEAP EBB DO

Ldir['ds0'] = '2017.09.11'

#Ldir['ds1'] = '2017.09.10'

his_num_list = [4, 5, 6, 7, 8, 9, 10]

fn_list = []

dir0 = Ldir['roms_out'] / Ldir['gtagex']

for num in his_num_list:
        
    f_string = 'f' + Ldir['ds0']
    
    num_str = str(num).zfill(2)
    
    his_str = 'ocean_his_00' + num_str + '.nc'
    
    fn = dir0 / f_string / his_str
    
    fn_list.append(fn)

sum_u_depth_avg = None
count=0

for fn in fn_list:
    ds = xr.open_dataset(fn)
    
    u = ds.u[0,:,:,:]
    
    # mask3 = np.broadcast_to(mask_u[None, :, :], u.shape)

    # u_poly = np.where(mask3, u, np.nan)
    
    u_poly = u.copy()
    
    u_poly_depth_avg = np.mean(u_poly, axis=0)
    
    if sum_u_depth_avg is None:
        sum_u_depth_avg = u_poly_depth_avg.copy()
    else:
        sum_u_depth_avg += u_poly_depth_avg
    
    count +=1
    
mean_u_depth_avg_2d = sum_u_depth_avg / count

fig, ax = plt.subplots()

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

cs = ax.pcolormesh(plon_u, plat_u, mean_u_depth_avg_2d, cmap=cmo.balance, vmin=-0.1,vmax=0.1)

# lons = [-122.63, -122.7, -122.7, -122.67, -122.650196, -122.63, -122.65, -122.67, -122.688425] #from pc field plan as of 2025/12/01
# lats = [48.247988, 48.226449, 48.232127, 48.23432, 48.240642, 48.228787, 48.230413, 48.228036, 48.222825]
# ax.scatter(lons,lats,color='k', edgecolor='white')

pfun.add_coast(ax)

pfun.dar(ax)

pfun.add_bathy_contours(ax, ds, depth_levs = [20], txt=True)
ax.set_ylim(48.21, 48.25)
ax.set_xlim(-122.740, -122.64)

fig.colorbar(cs, ax=ax)

ax.set_title('Neap Ebb u [m/s] ' + 'Depth-Averaged\n' +Ldir['ds0']+ ' ' + str(his_num_list[0]).zfill(2) + '00-' + str(his_num_list[-1]).zfill(2) + '00')

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_depth_avg_u_neap_ebb_' + Ldir['ds0']+ '_' + str(his_num_list[0]).zfill(2) + '00-' + str(his_num_list[-1]).zfill(2) + '00.png', dpi=500, transparent=False, bbox_inches='tight')


# %% SPRING FLOOD DO

Ldir['ds0'] = '2017.09.17'

#Ldir['ds1'] = '2017.09.10'

his_num_list = [17, 18, 19, 20, 21, 22, 23]

fn_list = []

dir0 = Ldir['roms_out'] / Ldir['gtagex']

for num in his_num_list:
        
    f_string = 'f' + Ldir['ds0']
    
    num_str = str(num).zfill(2)
    
    his_str = 'ocean_his_00' + num_str + '.nc'
    
    fn = dir0 / f_string / his_str
    
    fn_list.append(fn)

sum_u_depth_avg = None
count=0

for fn in fn_list:
    ds = xr.open_dataset(fn)
    
    u = ds.u[0,:,:,:]
    
    # mask3 = np.broadcast_to(mask_u[None, :, :], u.shape)

    # u_poly = np.where(mask3, u, np.nan)
    
    u_poly = u.copy()
    
    u_poly_depth_avg = np.mean(u_poly, axis=0)
    
    if sum_u_depth_avg is None:
        sum_u_depth_avg = u_poly_depth_avg.copy()
    else:
        sum_u_depth_avg += u_poly_depth_avg
    
    count +=1
    
mean_u_depth_avg_2d = sum_u_depth_avg / count

fig, ax = plt.subplots()

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

cs = ax.pcolormesh(plon_u, plat_u, mean_u_depth_avg_2d, cmap=cmo.balance, vmin=-0.1,vmax=0.1)

# lons = [-122.63, -122.7, -122.7, -122.67, -122.650196, -122.63, -122.65, -122.67, -122.688425] #from pc field plan as of 2025/12/01
# lats = [48.247988, 48.226449, 48.232127, 48.23432, 48.240642, 48.228787, 48.230413, 48.228036, 48.222825]
# ax.scatter(lons,lats,color='k', edgecolor='white')

pfun.add_coast(ax)

pfun.dar(ax)

pfun.add_bathy_contours(ax, ds, depth_levs = [20], txt=True)
ax.set_ylim(48.21, 48.25)
ax.set_xlim(-122.740, -122.64)

fig.colorbar(cs, ax=ax)

ax.set_title('Spring Flood u [m/s] ' + 'Depth-Averaged\n' +Ldir['ds0']+ ' ' + str(his_num_list[0]).zfill(2) + '00-' + str(his_num_list[-1]).zfill(2) + '00')

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_depth_avg_u_spring_flood_' + Ldir['ds0']+ '_' + str(his_num_list[0]).zfill(2) + '00-' + str(his_num_list[-1]).zfill(2) + '00.png', dpi=500, transparent=False, bbox_inches='tight')

# %% SPRING EBB DO

Ldir['ds0'] = '2017.09.18'

#Ldir['ds1'] = '2017.09.10'

his_num_list = [11, 12, 13, 14, 15, 16, 17]

fn_list = []

dir0 = Ldir['roms_out'] / Ldir['gtagex']

for num in his_num_list:
        
    f_string = 'f' + Ldir['ds0']
    
    num_str = str(num).zfill(2)
    
    his_str = 'ocean_his_00' + num_str + '.nc'
    
    fn = dir0 / f_string / his_str
    
    fn_list.append(fn)

sum_u_depth_avg = None
count=0

for fn in fn_list:
    ds = xr.open_dataset(fn)
    
    u = ds.u[0,:,:,:]
    
    # mask3 = np.broadcast_to(mask_u[None, :, :], u.shape)

    # u_poly = np.where(mask3, u, np.nan)
    
    u_poly = u.copy()
    
    u_poly_depth_avg = np.mean(u_poly, axis=0)
    
    if sum_u_depth_avg is None:
        sum_u_depth_avg = u_poly_depth_avg.copy()
    else:
        sum_u_depth_avg += u_poly_depth_avg
    
    count +=1
    
mean_u_depth_avg_2d = sum_u_depth_avg / count

fig, ax = plt.subplots()

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

cs = ax.pcolormesh(plon_u, plat_u, mean_u_depth_avg_2d, cmap=cmo.balance, vmin=-0.1,vmax=0.1)

# lons = [-122.63, -122.7, -122.7, -122.67, -122.650196, -122.63, -122.65, -122.67, -122.688425] #from pc field plan as of 2025/12/01
# lats = [48.247988, 48.226449, 48.232127, 48.23432, 48.240642, 48.228787, 48.230413, 48.228036, 48.222825]
# ax.scatter(lons,lats,color='k', edgecolor='white')

pfun.add_coast(ax)

pfun.dar(ax)

pfun.add_bathy_contours(ax, ds, depth_levs = [20], txt=True)
ax.set_ylim(48.21, 48.25)
ax.set_xlim(-122.740, -122.64)

fig.colorbar(cs, ax=ax)

ax.set_title('Spring Ebb u [m/s] ' + 'Depth-Averaged\n' +Ldir['ds0']+ ' ' + str(his_num_list[0]).zfill(2) + '00-' + str(his_num_list[-1]).zfill(2) + '00')

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_depth_avg_u_spring_ebb_' + Ldir['ds0']+ '_' + str(his_num_list[0]).zfill(2) + '00-' + str(his_num_list[-1]).zfill(2) + '00.png', dpi=500, transparent=False, bbox_inches='tight')


