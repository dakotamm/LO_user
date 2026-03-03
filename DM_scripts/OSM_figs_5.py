#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 15:24:57 2026

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


gridname = 'wb1'

tag = 'r0'

ex_name = 'xn11b'

Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)

Ldir['list_type'] = 'lowpass'

Ldir['ds0'] = '2017.09.05'

Ldir['ds1'] = '2017.09.18'

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

fn_list = Lfun.get_fn_list(Ldir['list_type'], Ldir,
    Ldir['ds0'], Ldir['ds1'])


fn = fn_list[0]

ds = xr.open_dataset(fn)

G, S, T = zrfun.get_basic_info(fn)

zeta = ds.zeta.values
NT = len(zeta)
hh = ds.h.values * np.ones(NT)
z_rho, z_w = zrfun.get_z(hh, zeta, S)

# %%

ds = xr.open_dataset('/Users/dakotamascarenas/LO_output/extract/wb1_r0_xn11b/tef2/extractions_2017.01.01_2017.12.31/pc0.nc')

df = pd.read_pickle('/Users/dakotamascarenas/LO_output/extract/tef2/sect_df_wb1_pc0.p')

# %%

ij_pc0 = df[df['sn'] == 'pc0'][['i','j']]

lons = lon_1D[ij_pc0['i']]

lats = lat_1D[ij_pc0['j']]

z_rho_section = z_rho[:, ij_pc0['j'], ij_pc0['i']]

lat2d = np.tile(lats, (30, 1))      # (Nz, Nsection)


# %%


for i in [0,1,2,3]:
    
    if i == 0:

        Ldir['ds0'] = '2017.08.01'
        
        Ldir['ds1'] = '2017.08.07'
        
    elif i == 1:
        
        Ldir['ds0'] = '2017.09.05'
        
        Ldir['ds1'] = '2017.09.11'
        
    elif i == 2:
        
        Ldir['ds0'] = '2017.09.12'
        
        Ldir['ds1'] = '2017.09.18'
        
    elif i == 3:
        
        Ldir['ds0'] = '2017.11.21'
        
        Ldir['ds1'] = '2017.11.27'
        
    ds_sub = ds.sel(time=slice(Ldir['ds0'], Ldir['ds1']))
    
    oxygen = ds_sub.oxygen.values*32/1000
    
    oxygen_mean = oxygen.mean(axis=0)
        
    vel = ds_sub.vel.values
    
    vel_mean = vel.mean(axis=0)
    
    transport = oxygen*vel
    
    transport_mean = transport.mean(axis=0)
    
    fig, ax = plt.subplots(figsize=(4,2))
    
    ax.axhspan(-27,0, color = '#f4f6f6')
    
    
    pcm = ax.pcolormesh(lat2d, z_rho_section, transport_mean*-1, vmin = -0.8, vmax = 0.8, cmap=cmc.vik)
    
    ax.set_xticks([48.23, 48.24], ['48.23', '48.24'])
    
    ax.set_ylim(-27,0)
    
    ax.set_xlim(lats.min(), lats.max())

    
    ax.set_xlabel('Latitude [°N]')
    
    ax.set_ylabel('z [m]')
            
    ax.plot(lat_rho[:,66], -hh[:,66], color='k', lw=0.5, zorder=5)




    
    cont = ax.contour(lat2d, z_rho_section, oxygen_mean, levels=[2], colors='red', linewidths=1)
        
    fig.colorbar(pcm, ax=ax, label="DO Transport [mg/L/s]")
    
    #ax.set_title(Ldir['ds0']+ '-' + Ldir['ds1'])
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc0_section_DO_transport_' + Ldir['ds0']+ '-' + Ldir['ds1'] + '.png', dpi=500, transparent=True, bbox_inches='tight')
        

# %%

# for i in range(len(vel)):

#     fig, ax = plt.subplots()
    
#     pcm = ax.pcolormesh(lat2d, z_rho_section, vel[i]*-1, vmin = -0.2, vmax = 0.2, cmap="RdBu")
    
#     fig.colorbar(pcm, ax=ax, label="Velocity (m/s)")
    
#     ax.set_title(str(ds.time.values[i])[0:16])
    
#     plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc0_vel_movie/pc0_vel_' + str(i).zfill(4) + '.png', bbox_inches='tight', dpi=500, transparent=False)


















