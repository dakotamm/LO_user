#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 13:11:22 2026

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


fig, axd = plt.subplot_mosaic([['puget_sound', 'penn_cove', 'penn_cove', 'penn_cove']], figsize=(9,6), layout='constrained', gridspec_kw=dict(wspace=0.1))

ax = axd['puget_sound']
 
ax.set_xlim(X[i1],-121.4)#X[i2]) # Salish Sea
ax.set_ylim(Y[j1],Y[j2]) # Salish Sea
        
ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

#sns.scatterplot(data=odf_ixiy_unique, x='lon', y='lat', ax = ax, color = 'gray', alpha=0.3, label= 'Cast Location')


pfun.add_coast(ax)

pfun.dar(ax)


#ax.text(0.15,0.81, 'Strait of\nJuan de Fuca', transform=ax.transAxes, fontsize = 8, color = 'black', ha='center', va='center', rotation = -30)

#ax.text(0.3,0.85, '^ to Strait\nof Georgia', transform=ax.transAxes, fontsize = 7, color = 'black', ha='center', va='center')

#ax.text(0.36,0.785, 'Admiralty\nInlet', transform=ax.transAxes, fontsize = 8, color = 'gray', ha='center', va='center')

#ax.text(0.65,0.16, 'Tacoma\nNarrows', transform=ax.transAxes, fontsize = 8, color = 'gray', ha='center', va='center')


#ax.text(0.36,0.785, 'Deception Pass', transform=ax.transAxes, fontsize = 8, color = 'gray', ha='center', va='center')

ax.text(0.6,0.03, 'Puget\nSound', multialignment='center', transform=ax.transAxes, fontsize = 12, color = 'black')

#ax.text(0.025,0.36, 'Hood Canal', transform=ax.transAxes, fontsize = 12, color = 'black', rotation = 55)

#ax.text(0.57,0.1, 'South Sound', transform=ax.transAxes, fontsize = 12, color = 'black')

#ax.text(0.77,0.5, 'Main Basin', transform=ax.transAxes, fontsize = 12, color = 'black', rotation = 50)

#ax.text(0.45,0.73, 'Whidbey Basin', transform=ax.transAxes, fontsize = 10, color = 'black')
 
ax.text(0.88,0.9, 'Skagit\nRiver', transform=ax.transAxes, fontsize = 8, color = 'gray', ha='center', va='center')

ax.plot([-122.740, -122.64], [48.21, 48.21], color='k')

ax.plot([-122.740, -122.64], [48.25, 48.25], color='k')

ax.plot([-122.740, -122.740], [48.21, 48.25], color='k')

ax.plot([-122.64, -122.64], [48.21, 48.25], color='k')

ax.set_ylim(48.21, 48.25)
ax.set_xlim(-122.740, -122.64)


 

ax.text(0.05,0.025, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')


# ax.plot([-122.65,-122.65],[48.35, 48.45], color = 'black', linestyle='--', linewidth=3)

# ax.plot([-122.8,-122.7],[48.1, 48.2], color = 'black', linestyle='--', linewidth=3)



# ax.plot([-122.75,-122.55],[47.95, 47.9], color = 'gray', linestyle='--', linewidth=2)

# ax.plot([-122.61,-122.49],[47.37, 47.27], color = 'gray', linestyle='--', linewidth=2)

# ax.plot([-122.61,-122.49],[47.37, 47.27], color = 'gray', linestyle='--', linewidth=2)

# ax.plot([-122.40,-122.27],[47.95, 47.87], color = 'gray', linestyle='--', linewidth=2)



 
#ax.legend(loc = 'upper left')

ax.set_xlim(-123.2, -122.1) 
 
ax.set_ylim(47,48.5)


ax.set_xlabel('')

ax.set_ylabel('')
 
#xlbl = ax.get_xticklabels()

ax.set_xticks([-123.0, -122.6, -122.2], ['-123.0','-122.6', '-122.2']) #['','-123.0', '', '-122.6', '', '-122.2'])



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



fn = '/Users/dakotamascarenas/LO_roms/wb1_r0_xn11b/averages/monthly_mean_2017_09.nc'

ds = xr.open_dataset(fn)

ax = axd['penn_cove']

bottom_oxygen = ds.oxygen[0,0,:,:]*32/1000

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

cs = ax.pcolormesh(plon, plat, bottom_oxygen, cmap=cmo.oxy, vmin=0,vmax=10)

pfun.add_coast(ax)

pfun.dar(ax)

pfun.add_bathy_contours(ax, ds, depth_levs = [20], txt=True)
ax.set_ylim(48.21, 48.25)
ax.set_xlim(-122.740, -122.64)

ax.set_xticks([-122.7, -122.65], ['-122.70','-122.65']) #['','-123.0', '', '-122.6', '', '-122.2'])

ax.set_yticks([48.22, 48.24], ['48.22','48.24']) #['','-123.0', '', '-122.6', '', '-122.2'])

ax.text(0.05,0.9, 'Penn Cove', multialignment='center', transform=ax.transAxes, fontsize = 14, color = 'black')




fig.colorbar(cs, ax=ax, fraction = 0.03, label = 'DO [mg/L]')

ax.text(0.05,0.075, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')




plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pecs_2026_abstract_fig_1.png', bbox_inches='tight', dpi=500, transparent=True)





