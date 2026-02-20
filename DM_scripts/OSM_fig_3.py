#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 09:26:00 2026

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

def generate_buda_colors(n_colors=3):
    """
    Generates a list of n colors from the 'batlow' colormap.

    Args:
        n_colors (int): The number of colors to generate.

    Returns:
        list: A list of RGBA color tuples.
    """
    # Access the batlow colormap from cmcrameri
    cmap = cmc.buda
    
    # Generate evenly spaced values from 0 to 1 to sample the colormap
    # We use np.linspace to get n_colors evenly spaced points
    sample_points = np.linspace(0, 1, n_colors)

    # Get the colors from the colormap
    # cmap returns RGBA values as a numpy array
    colors = cmap(sample_points)
    
    return colors.tolist()

# Generate the three batlow colors
three_colors = generate_buda_colors(3)

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

mosaic = [['M3_bottom_less_top_salinity'], ['M3_wind'], ['M3_bottom_temp'], ['qprism'], ['M1-M3-M5_bottom_DO']]

fig, axd = plt.subplot_mosaic(mosaic, figsize=(6,8), layout='constrained', gridspec_kw=dict(wspace=0.1), sharex=True)



ax = axd['M3_bottom_less_top_salinity']

moor_dir = Ldir['LOo'] / 'extract'

m = 'M3'

fn = m + '_2017.01.02_2017.12.30.nc'
moor_fn = moor_dir / 'wb1_r0_xn11b' / 'moor' / 'pc0'/ fn
ds = xr.open_dataset(moor_fn)

times = ds.ocean_time

temp = ds.temp.values

salt = ds.salt.values

h = ds.h.values

z_rho = ds.z_rho.values

lats = ds.lat_rho.values

lons = ds.lon_rho.values

temp_bottom = temp[:,0]

temp_top = temp[:,-1]

salt_bottom = salt[:,0]

salt_top = salt[:,-1]

pres_top = gsw.p_from_z(z_rho[:,-1], lats) # pressure [dbar]
pres_bottom = gsw.p_from_z(z_rho[:,0], lats) # pressure [dbar]

SA_top = gsw.SA_from_SP(salt_top, pres_top, lons, lats)
SA_bottom = gsw.SA_from_SP(salt_bottom, pres_bottom, lons, lats)

CT_top = gsw.CT_from_pt(SA_top, temp_top)
CT_bottom = gsw.CT_from_pt(SA_bottom, temp_bottom)


rho_top = gsw.rho(SA_top, CT_top, pres_top)
rho_bottom = gsw.rho(SA_bottom, CT_bottom, pres_bottom)

d_rho = rho_bottom - rho_top

#ax.axvspan()

ax.plot(times, d_rho, color = 'k')

ax.text(0.025,0.85, '\u0394\u03C1 (bottom - top) at entrance', transform=ax.transAxes, fontweight='bold', color = 'k')

ax.set_ylabel(r'$[kg/m^3]$')

ax.set_ylim(top =15)

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)





ax = axd['M3_wind']

uwind = ds.Uwind.values

vwind = ds.Vwind.values

wind_mag = np.hypot(uwind, vwind)

ax.plot(times, wind_mag, color = 'k')

ax.text(0.025,0.85, '10m wind speed at entrance', transform=ax.transAxes, fontweight='bold', color = 'k')

ax.set_ylabel(r'$[m/s]$')

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)



ax = axd['M3_bottom_temp']

temp = ds.temp.values

temp_bottom = temp[:,0]

wind_mag = np.hypot(uwind, vwind)

ax.plot(times, temp_bottom, color = 'k')

ax.text(0.025,0.85, 'bottom temperature at entrance', transform=ax.transAxes, fontweight='bold', color = 'k')

ax.set_ylabel(r'[°C]')

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)



ax = axd['qprism']

df = pd.read_pickle('/Users/dakotamascarenas/LO_output/extract/tef2/seg_info_dict_wb1_pc0_riv00.p')

ds = xr.open_dataset('/Users/dakotamascarenas/LO_output/extract/wb1_r0_xn11b/tef2/bulk_2017.01.01_2017.12.31/pc0.nc')

qprism = ds.qprism.values/1000

qprism_times = ds.qprism.time

ax.plot(qprism_times[1:-2], qprism[1:-2], color = 'k')

ax.text(0.025,0.85, 'tidal prism flow rate ' + r'$Q_{prism}$', transform=ax.transAxes, fontweight='bold', color = 'k')


ax.set_ylabel(r'$[km^3/s]$')

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)




ax = axd['M1-M3-M5_bottom_DO']


moor_dir = Ldir['LOo'] / 'extract'

m_list = ['M1', 'M3', 'M5']

c=0

for m in m_list:

    fn = m + '_2017.01.02_2017.12.30.nc'
    moor_fn = moor_dir / 'wb1_r0_xn11b' / 'moor' / 'pc0'/ fn
    ds = xr.open_dataset(moor_fn)
    
    times = ds.ocean_time

    oxygen = ds.oxygen.values*32/1000

    oxygen_bottom = oxygen[:,0]

    ax.plot(times, oxygen_bottom, color = three_colors[c]) #, label = m)
    
    #ax.legend()
    
    c+=1
    
ax.axhspan(0,2, color = 'lightgray', alpha = 0.5, zorder=-5, label='Hypoxia')

    

ax.set_ylim(0,10)

ax.set_ylabel('Bottom [DO] [mg/L]')

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)

import matplotlib.dates as mdates


# --- format shared x axis (apply to bottom axis only) ---
bottom_ax = axd['M1-M3-M5_bottom_DO']

# hide x labels on all but bottom
for key, ax in axd.items():
    if ax is not bottom_ax:
        ax.tick_params(labelbottom=False)

# major ticks: every month (or every 2 months)
bottom_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # e.g., Jan\n2017

# minor ticks: monthly (for faint grid / tick marks)
bottom_ax.xaxis.set_minor_locator(mdates.MonthLocator())

# optional: keep labels from overlapping
bottom_ax.tick_params(axis='x', rotation=0)

# optional: keep x-limits tight (shared across all axes)
bottom_ax.set_xlim(np.datetime64('2017-01-01'), np.datetime64('2017-12-31'))
bottom_ax.set_xlabel('2017')



plt.savefig('/Users/dakotamascarenas/Desktop/pltz/OSM26_fig_3.png', bbox_inches='tight', dpi=500, transparent=True)







 
    