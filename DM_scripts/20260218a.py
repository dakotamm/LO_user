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

mosaic = [['qprism'], ['d_rho_d_z'], ['bot_DO_section'], ['bot_DO_in'], ['bot_DO_out']]

fig, axd = plt.subplot_mosaic(mosaic, figsize=(9,9), layout='constrained', gridspec_kw=dict(wspace=0.1), sharex=True)

ax = axd['qprism']

df = pd.read_pickle('/Users/dakotamascarenas/LO_output/extract/tef2/seg_info_dict_wb1_pc0_riv00.p')

ds = xr.open_dataset('/Users/dakotamascarenas/LO_output/extract/wb1_r0_xn11b/tef2/bulk_2017.01.01_2017.12.31/pc0.nc')

qprism = ds.qprism.values/1000

qprism_times = ds.qprism.time

ax.plot(qprism_times[1:-2], qprism[1:-2])

ax.set_ylabel('Qprism [m^3/s]')


ax = axd['d_rho_d_z']

ds = xr.open_dataset('/Users/dakotamascarenas/LO_output/extract/wb1_r0_xn11b/tef2/extractions_2017.01.01_2017.12.31/pc0.nc')

df = pd.read_pickle('/Users/dakotamascarenas/LO_output/extract/tef2/sect_df_wb1_pc0.p')

ij_pc0 = df[df['sn'] == 'pc0'][['i','j']]

lons = lon_1D[ij_pc0['i']]

lats = lat_1D[ij_pc0['j']]


temp = ds.temp.values

salt = ds.salt.values

zeta = ds.zeta.values

h = ds.h.values

section_times = ds.time



temp_bottom = temp[:,0,:]

temp_top = temp[:,-1,:]

salt_bottom = salt[:,0,:]

salt_top = salt[:,-1,:]


pres_top = gsw.p_from_z(z_rho[-1, ij_pc0['j'], ij_pc0['i']], lats) # pressure [dbar]
pres_bottom = gsw.p_from_z(z_rho[0, ij_pc0['j'], ij_pc0['i']], lats) # pressure [dbar]

SA_top = gsw.SA_from_SP(salt_top, pres_top, lons, lats)
SA_bottom = gsw.SA_from_SP(salt_bottom, pres_bottom, lons, lats)

CT_top = gsw.CT_from_pt(SA_top, temp_top)
CT_bottom = gsw.CT_from_pt(SA_bottom, temp_bottom)


rho_top = gsw.rho(SA_top, CT_top, pres_top)
rho_bottom = gsw.rho(SA_bottom, CT_bottom, pres_bottom)


d_rho = rho_bottom - rho_top

d_rho_d_z = d_rho/h

d_rho_d_z_avg = d_rho_d_z.mean(axis = 1)


ax.plot(section_times, d_rho_d_z_avg)

ax.set_ylabel('drho/dz [g/kg/m]')



ax = axd['bot_DO_section']

oxygen = ds.oxygen.values*32/1000

oxygen_bottom = oxygen[:,0,:]

oxygen_bottom_avg = oxygen_bottom.mean(axis=1)


ax.plot(section_times, oxygen_bottom_avg)

ax.set_ylim(0,10)


ax.set_ylabel('bot DO, section [mgL]')

ax = axd['bot_DO_in']


moor_dir = Ldir['LOo'] / 'extract'

m = 'M1'

fn = m + '_2017.01.02_2017.12.30.nc'
moor_fn = moor_dir / 'wb1_r0_xn11b' / 'moor' / 'pc0'/ fn
ds = xr.open_dataset(moor_fn)



oxygen = ds.oxygen.values*32/1000

oxygen_times = ds.ocean_time

oxygen_bottom = oxygen[:,0]

ax.plot(oxygen_times, oxygen_bottom)

ax.set_ylim(0,10)


ax.set_ylabel('bot DO, in [mgL]')


ax = axd['bot_DO_out']


moor_dir = Ldir['LOo'] / 'extract'

m = 'M5'

fn = m + '_2017.01.02_2017.12.30.nc'
moor_fn = moor_dir / 'wb1_r0_xn11b' / 'moor' / 'pc0'/ fn
ds = xr.open_dataset(moor_fn)



oxygen = ds.oxygen.values*32/1000

oxygen_times = ds.ocean_time

oxygen_bottom = oxygen[:,0]

ax.plot(oxygen_times, oxygen_bottom)

ax.set_ylim(0,10)

ax.set_ylabel('bot DO, out [mgL]')


plt.savefig('/Users/dakotamascarenas/Desktop/pltz/splat.png', bbox_inches='tight', dpi=500, transparent=False)







 
    