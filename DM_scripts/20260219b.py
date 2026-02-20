#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 16:09:44 2026

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

# %%

for i in [0,1,2,3]:
    
    if i == 0:

        Ldir['ds0'] = '2017.08.01'
        
        Ldir['ds1'] = '2017.08.06'
        
    elif i == 1:
        
        Ldir['ds0'] = '2017.09.05'
        
        Ldir['ds1'] = '2017.09.11'
        
    elif i == 2:
        
        Ldir['ds0'] = '2017.09.12'
        
        Ldir['ds1'] = '2017.09.18'
        
    elif i == 3:
        
        Ldir['ds0'] = '2017.11.21'
        
        Ldir['ds1'] = '2017.11.27'
        


    fn_list = Lfun.get_fn_list(Ldir['list_type'], Ldir, Ldir['ds0'], Ldir['ds1'])
        
    sum_bottom_oxygen = None
    sum_u = None
    sum_v = None
    count=0
    
    for fn in fn_list:
        ds = xr.open_dataset(fn)
        
        bottom_oxygen = ds.oxygen[0,0,:,:].values*32/1000
        
        u = ds.u[0,:,:,:].values
        
        v = ds.v[0,:,:,:].values
        
        # mask3 = np.broadcast_to(mask_u[None, :, :], u.shape)
    
        # u_poly = np.where(mask3, u, np.nan)
        
        u_poly = u.copy()
        
        v_poly = v.copy()
        
        bottom_oxygen_poly = bottom_oxygen.copy()
                
        
       # u_poly_depth_avg = np.mean(u_poly, axis=0)
        
        if sum_bottom_oxygen is None:
            sum_bottom_oxygen = bottom_oxygen_poly.copy()
        else:
            sum_bottom_oxygen += bottom_oxygen_poly
            
        if sum_u is None:
            sum_u = u_poly.copy()
        else:
            sum_u += u_poly
            
        if sum_v is None:
            sum_v = v_poly.copy()
        else:
            sum_v += v_poly
        
        count +=1
        
    mean_u_3d = sum_u / count
    
    mean_v_3d = sum_v / count
    
    mean_u_depth_avg_2d = np.mean(mean_u_3d, axis=0)
    
    mean_v_depth_avg_2d = np.mean(mean_v_3d, axis=0)
        
    mean_bottom_oxygen_2d = sum_bottom_oxygen / count
    
    
    fig, ax = plt.subplots()
    
        
    ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)
    
    cs = ax.pcolormesh(plon, plat, mean_bottom_oxygen_2d, cmap=cmc.batlow_r, vmin=0,vmax=10)
    
    cont = ax.contour(lon_rho, lat_rho, mean_bottom_oxygen_2d, levels=[2], colors='magenta', linewidths=1) 
    
    
    # ADD VELOCITY VECTORS # FROM PFUN ADAPTED 20260210
    nngrid = 200
    v_scl = 1
    v_leglen=0.1
    center=(.8,.05)
    # set masked values to 0
    mean_u_depth_avg_2d[np.isnan(mean_u_depth_avg_2d)] = 0
    mean_v_depth_avg_2d[np.isnan(mean_v_depth_avg_2d)] = 0
    # create regular grid
    aaa = ax.axis()
    daax = aaa[1] - aaa[0]
    daay = aaa[3] - aaa[2]
    axrat = np.cos(np.deg2rad(aaa[2])) * daax / daay
    x = np.linspace(aaa[0], aaa[1], int(round(nngrid * axrat)))
    y = np.linspace(aaa[2], aaa[3], int(nngrid))
    xx, yy = np.meshgrid(x, y)
    # interpolate to regular grid
    uu = zfun.interp2(xx, yy, lon_u, lat_u, mean_u_depth_avg_2d)
    vv = zfun.interp2(xx, yy, lon_v, lat_v, mean_v_depth_avg_2d)
    
    mask = uu != 0
    # plot velocity vectors
    Q = ax.quiver(xx[mask], yy[mask], uu[mask], vv[mask],
        scale=v_scl, scale_units='width', color='white', units='width')
    plt.quiverkey(Q, .9, .1, v_leglen, str(v_leglen)+' $ms^{-1}$', angle=20, color='black')
    
    
    pfun.add_coast(ax)
    
    pfun.dar(ax)
    
    #pfun.add_bathy_contours(ax, ds, depth_levs = [20], txt=True)
    ax.set_ylim(48.21, 48.25)
    ax.set_xlim(-122.740, -122.64)
    
    fig.colorbar(cs, ax=ax, shrink=0.5)
    
    ax.set_title('DO [mg/L] ' + 'Bottom\n' +Ldir['ds0']+ '-' + Ldir['ds1'])
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_bottom_DO_' + Ldir['ds0']+ '-' + Ldir['ds1'] + '_wquiver.png', dpi=500, transparent=True, bbox_inches='tight')
        
    
