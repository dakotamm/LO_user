#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 14:32:08 2026

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
from cmcrameri import cm

# %%

gridname = 'wb1'

tag = 'r0'

ex_name = 'xn11b'

Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)

Ldir['list_type'] = 'hourly0'

Ldir['ds0'] = '2017.12.3'

Ldir['ds1'] = '2017.12.5'

Ldir['his_num'] = 2
    
fn_list = Lfun.get_fn_list(Ldir['list_type'], Ldir,
    Ldir['ds0'], Ldir['ds1'], his_num=Ldir['his_num'])

sect_path = Ldir['LOo'] / 'section_lines'
sect = 'pc.p'

sect_fn = sect_path / sect

pdict = pickle.load(open(sect_fn, 'rb'))

fng = Ldir['grid'] / 'grid.nc'
dsg = xr.open_dataset(fng)
lon_rho = dsg.lon_rho.values
lat_rho = dsg.lat_rho.values
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
mask = points_in_polygon_mask(lon_rho, lat_rho, pdict)

# # indices (iy, ix) of points inside:
iy, ix = np.where(mask)


sum_dens = None
count=0

for fn in fn_list:
    ds = xr.open_dataset(fn)
    
    G, S, T = zrfun.get_basic_info(fn)
    
    zeta = ds.zeta.values
    NT = len(zeta)
    hh = ds.h.values * np.ones(NT)
    z_rho, z_w = zrfun.get_z(hh, zeta, S)

    
    salt = ds.salt[0,:,:,:].values.squeeze()
    
    temp = ds.temp[0,:,:,:].values.squeeze()
    
    lat3 = np.broadcast_to(lat_rho[None, :, :], z_rho.shape)
    
    lon3 = np.broadcast_to(lon_rho[None, :, :], z_rho.shape)


    pres = gsw.p_from_z(z_rho, lat3) # pressure [dbar]
    SA = gsw.SA_from_SP(salt, pres, lon3, lat3)
    CT = gsw.CT_from_pt(SA, temp)
    
    rho = gsw.rho(SA, CT, pres)
    
    mask3 = np.broadcast_to(mask[None, :, :], rho.shape)

    rho_poly = np.where(mask3, rho, np.nan)
    
    if sum_dens is None:
        sum_dens = rho_poly.copy()
    else:
        sum_dens += rho_poly
    
    count +=1
    
mean_dens_3d = sum_dens / count

mean_dens_total = np.nanmean(mean_dens_3d)

mean_dens_per_layer = np.nanmean(mean_dens_3d, axis=(1, 2))


# calc variance from total dens mean

sum_dens_var_mean_total = None

count=0

for fn in fn_list:
    ds = xr.open_dataset(fn)
    
    G, S, T = zrfun.get_basic_info(fn)
    
    zeta = ds.zeta.values
    NT = len(zeta)
    hh = ds.h.values * np.ones(NT)
    z_rho, z_w = zrfun.get_z(hh, zeta, S)

    
    salt = ds.salt[0,:,:,:].values.squeeze()
    
    temp = ds.temp[0,:,:,:].values.squeeze()
    
    lat3 = np.broadcast_to(lat_rho[None, :, :], z_rho.shape)
    
    lon3 = np.broadcast_to(lon_rho[None, :, :], z_rho.shape)


    pres = gsw.p_from_z(z_rho, lat3) # pressure [dbar]
    SA = gsw.SA_from_SP(salt, pres, lon3, lat3)
    CT = gsw.CT_from_pt(SA, temp)
    
    rho = gsw.rho(SA, CT, pres)
    
    mask3 = np.broadcast_to(mask[None, :, :], rho.shape)
    
    rho_poly = np.where(mask3, rho, np.nan)
        
    d = (rho_poly - mean_dens_total)**2
    
    if sum_dens_var_mean_total is None:
        
        sum_dens_var_mean_total = d.copy()
        
    else:
        
        sum_dens_var_mean_total += d

        
    count+=1        
    
var_dens_total_mean = sum_dens_var_mean_total / count


# %%


fig, ax = plt.subplots()

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

cs = ax.pcolormesh(plon, plat, var_dens_total_mean[0], cmap=cm.batlow)

lons = [-122.63, -122.7, -122.7, -122.67, -122.650196, -122.63, -122.65, -122.67, -122.688425] #from pc field plan as of 2025/12/01
lats = [48.247988, 48.226449, 48.232127, 48.23432, 48.240642, 48.228787, 48.230413, 48.228036, 48.222825]
ax.scatter(lons,lats,color='k', edgecolor='white')

pfun.add_coast(ax)

pfun.dar(ax)

pfun.add_bathy_contours(ax, ds, depth_levs = [20], txt=True)
ax.set_ylim(48.21, 48.25)
ax.set_xlim(-122.740, -122.64)

fig.colorbar(cs, ax=ax)

ax.set_title('Dens. Var. (Total Mean) [kg/m^3]^2\n' + 'Bottom ' + Ldir['ds0']+ '-' + Ldir['ds1'])

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_total_mean_dens_var_bottom_' + Ldir['ds0']+ '-' + Ldir['ds1']+'.png', dpi=500, transparent=False, bbox_inches='tight')



fig, ax = plt.subplots()

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

cs = ax.pcolormesh(plon, plat, var_dens_total_mean[-1],  cmap=cm.batlow)

lons = [-122.63, -122.7, -122.7, -122.67, -122.650196, -122.63, -122.65, -122.67, -122.688425] #from pc field plan as of 2025/12/01
lats = [48.247988, 48.226449, 48.232127, 48.23432, 48.240642, 48.228787, 48.230413, 48.228036, 48.222825]
ax.scatter(lons,lats,color='k', edgecolor='white')

pfun.add_coast(ax)

pfun.dar(ax)

pfun.add_bathy_contours(ax, ds, depth_levs = [20], txt=True)
ax.set_ylim(48.21, 48.25)
ax.set_xlim(-122.740, -122.64)

fig.colorbar(cs, ax=ax)

ax.set_title('Dens. Var. (Total Mean) [kg/m^3]^2\n' + 'Top ' + Ldir['ds0']+ '-' + Ldir['ds0'])

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_total_mean_dens_var_top_' + Ldir['ds0']+ '-' + Ldir['ds1']+'.png', dpi=500, transparent=False, bbox_inches='tight')



fig, ax = plt.subplots()

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

cs = ax.pcolormesh(plon, plat, var_dens_total_mean[15], cmap=cm.batlow)

lons = [-122.63, -122.7, -122.7, -122.67, -122.650196, -122.63, -122.65, -122.67, -122.688425] #from pc field plan as of 2025/12/01
lats = [48.247988, 48.226449, 48.232127, 48.23432, 48.240642, 48.228787, 48.230413, 48.228036, 48.222825]
ax.scatter(lons,lats,color='k', edgecolor='white')

pfun.add_coast(ax)

pfun.dar(ax)

pfun.add_bathy_contours(ax, ds, depth_levs = [20], txt=True)
ax.set_ylim(48.21, 48.25)
ax.set_xlim(-122.740, -122.64)

fig.colorbar(cs, ax=ax)

ax.set_title('Dens. Var. (Total Mean) [kg/m^3]^2\n' + 'Mid ' + Ldir['ds0']+ '-' + Ldir['ds1'])

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_total_mean_dens_var_mid_' + Ldir['ds0']+ '-' + Ldir['ds1']+'.png', dpi=500, transparent=False, bbox_inches='tight')

# %%

sum_speed = None
count=0

for fn in fn_list:
    ds = xr.open_dataset(fn)

    u = ds.u[0,:,:,:].values.squeeze()
    
    v = ds.v[0,:,:,:].values.squeeze()
    
    Nz, eta_rho, xi_rho = 30, 368, 272

    # u: (Nz, eta_rho, xi_rho-1) = (30,368,271)
    u_rho = np.full((Nz, eta_rho, xi_rho), np.nan, dtype=float)
    u_rho[:, :, 1:-1] = 0.5 * (u[:, :, :-1] + u[:, :, 1:])  # (30,368,270)
    u_rho[:, :, 0]    = u[:, :, 0]      # left edge
    u_rho[:, :, -1]   = u[:, :, -1]     # right edge
    
    # v: (Nz, eta_rho-1, xi_rho) = (30,367,272)
    v_rho = np.full((Nz, eta_rho, xi_rho), np.nan, dtype=float)
    v_rho[:, 1:-1, :] = 0.5 * (v[:, :-1, :] + v[:, 1:, :])  # (30,366,272)
    v_rho[:, 0, :]    = v[:, 0, :]      # bottom edge
    v_rho[:, -1, :]   = v[:, -1, :]     # top edge
    
    speed_rho = np.hypot(u_rho, v_rho)  # (30,368,272)
    
    #speed = np.sqrt(u**2 + v**2)
    
    mask3 = np.broadcast_to(mask[None, :, :], speed_rho.shape)

    speed_poly = np.where(mask3, speed_rho, np.nan)
    
    if sum_speed is None:
        sum_speed = speed_poly.copy()
    else:
        sum_speed += speed_poly
    
    count +=1
    
mean_speed_3d = sum_speed / count

mean_speed_total = np.nanmean(mean_speed_3d)

# mean_speed_per_layer = np.nanmean(mean_speed_3d, axis=(1, 2))


# calc variance from total speed mean

sum_speed_var_mean_total = None

count=0

for fn in fn_list:
    ds = xr.open_dataset(fn)
    
    u = ds.u[0,:,:,:].values.squeeze()
    
    v = ds.v[0,:,:,:].values.squeeze()
    
    u = ds.u[0,:,:,:].values.squeeze()
    
    v = ds.v[0,:,:,:].values.squeeze()
    
    Nz, eta_rho, xi_rho = 30, 368, 272

    # u: (Nz, eta_rho, xi_rho-1) = (30,368,271)
    u_rho = np.full((Nz, eta_rho, xi_rho), np.nan, dtype=float)
    u_rho[:, :, 1:-1] = 0.5 * (u[:, :, :-1] + u[:, :, 1:])  # (30,368,270)
    u_rho[:, :, 0]    = u[:, :, 0]      # left edge
    u_rho[:, :, -1]   = u[:, :, -1]     # right edge
    
    # v: (Nz, eta_rho-1, xi_rho) = (30,367,272)
    v_rho = np.full((Nz, eta_rho, xi_rho), np.nan, dtype=float)
    v_rho[:, 1:-1, :] = 0.5 * (v[:, :-1, :] + v[:, 1:, :])  # (30,366,272)
    v_rho[:, 0, :]    = v[:, 0, :]      # bottom edge
    v_rho[:, -1, :]   = v[:, -1, :]     # top edge
    
    speed_rho = np.hypot(u_rho, v_rho)  # (30,368,272)
    
    #speed = np.sqrt(u**2 + v**2)
    
    mask3 = np.broadcast_to(mask[None, :, :], speed_rho.shape)

    speed_poly = np.where(mask3, speed_rho, np.nan)
        
    d = (speed_poly - mean_speed_total)**2
    
    if sum_speed_var_mean_total is None:
        
        sum_speed_var_mean_total = d.copy()
        
    else:
        
        sum_speed_var_mean_total += d

        
    count+=1        
    
var_speed_total_mean = sum_speed_var_mean_total / count

# %%


fig, ax = plt.subplots()

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

cs = ax.pcolormesh(plon, plat, var_speed_total_mean[0], cmap=cm.batlow)

lons = [-122.63, -122.7, -122.7, -122.67, -122.650196, -122.63, -122.65, -122.67, -122.688425] #from pc field plan as of 2025/12/01
lats = [48.247988, 48.226449, 48.232127, 48.23432, 48.240642, 48.228787, 48.230413, 48.228036, 48.222825]
ax.scatter(lons,lats,color='k', edgecolor='white')

pfun.add_coast(ax)

pfun.dar(ax)

pfun.add_bathy_contours(ax, ds, depth_levs = [20], txt=True)
ax.set_ylim(48.21, 48.25)
ax.set_xlim(-122.740, -122.64)

fig.colorbar(cs, ax=ax)

ax.set_title('Speed Var. (Total Mean) [m/s]^2\n' + 'Bottom ' + Ldir['ds0']+ '-' + Ldir['ds1'])

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_total_mean_speed_var_bottom_' + Ldir['ds0']+ '-' + Ldir['ds1']+'.png', dpi=500, transparent=False, bbox_inches='tight')



fig, ax = plt.subplots()

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

cs = ax.pcolormesh(plon, plat, var_speed_total_mean[-1],  cmap=cm.batlow)

lons = [-122.63, -122.7, -122.7, -122.67, -122.650196, -122.63, -122.65, -122.67, -122.688425] #from pc field plan as of 2025/12/01
lats = [48.247988, 48.226449, 48.232127, 48.23432, 48.240642, 48.228787, 48.230413, 48.228036, 48.222825]
ax.scatter(lons,lats,color='k', edgecolor='white')

pfun.add_coast(ax)

pfun.dar(ax)

pfun.add_bathy_contours(ax, ds, depth_levs = [20], txt=True)
ax.set_ylim(48.21, 48.25)
ax.set_xlim(-122.740, -122.64)

fig.colorbar(cs, ax=ax)

ax.set_title('Speed Var. (Total Mean) [m/s]^2\n' + 'Top ' + Ldir['ds0']+ '-' + Ldir['ds0'])

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_total_mean_speed_var_top_' + Ldir['ds0']+ '-' + Ldir['ds1']+'.png', dpi=500, transparent=False, bbox_inches='tight')



fig, ax = plt.subplots()

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

cs = ax.pcolormesh(plon, plat, var_speed_total_mean[15], cmap=cm.batlow)

lons = [-122.63, -122.7, -122.7, -122.67, -122.650196, -122.63, -122.65, -122.67, -122.688425] #from pc field plan as of 2025/12/01
lats = [48.247988, 48.226449, 48.232127, 48.23432, 48.240642, 48.228787, 48.230413, 48.228036, 48.222825]
ax.scatter(lons,lats,color='k', edgecolor='white')

pfun.add_coast(ax)

pfun.dar(ax)

pfun.add_bathy_contours(ax, ds, depth_levs = [20], txt=True)
ax.set_ylim(48.21, 48.25)
ax.set_xlim(-122.740, -122.64)

fig.colorbar(cs, ax=ax)

ax.set_title('Speed Var. (Total Mean) [m/s]^2\n' + 'Mid ' + Ldir['ds0']+ '-' + Ldir['ds1'])

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_total_mean_speed_var_mid_' + Ldir['ds0']+ '-' + Ldir['ds1']+'.png', dpi=500, transparent=False, bbox_inches='tight')
