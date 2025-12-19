#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:29:56 2024

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

import matplotlib.patheffects as pe




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

# %%

poly_list = ['ps', 'carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson']


odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_his', 'kc_whidbeyBasin', 'nceiSalish', 'kc_pointJefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf['ix_iy'] = odf['ix'].astype(str).apply(lambda x: x.zfill(4)) + '_' + odf['iy'].astype(str).apply(lambda x: x.zfill(4))


# %%

odf_ixiy_unique = odf.groupby(['ix_iy']).first().reset_index()


# %%

red =     "#EF5E3C"   # warm orange-red ##ff4040 #e04256

blue =     "#3A59B3"  # deep blue #4565e8

mosaic = [['map_big', 'map_source']]

fig, axd = plt.subplot_mosaic(mosaic, figsize=(9,9), layout='constrained', gridspec_kw=dict(wspace=0.1))

ax = axd['map_big']

ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

ax.set_xlim(-126,-122)#X[i2]) # Salish Sea
ax.set_ylim(47,50) # Salish Sea

ax.plot([-123.2, -122.1], [47, 47], color='k')

ax.plot([-123.2, -122.1], [48.5, 48.5], color='k')

ax.plot([-123.2, -123.2], [47, 48.5], color='k')

ax.plot([-122.1, -122.1], [47, 48.5], color='k')

ax.text(0.1,0.1, 'Pacific\nOcean', transform=ax.transAxes, fontsize = 14, color = 'black')

ax.text(0.85,0.65, 'Salish\nSea', transform=ax.transAxes, multialignment= 'center', ha='center', fontsize = 14, color = 'black')


ax.text(0.6,0.8, 'British Columbia,\nCanada', transform=ax.transAxes, multialignment= 'center', fontsize = 12, color = 'gray')

ax.text(0.41,0.35, 'Washington,\nUSA', transform=ax.transAxes, multialignment= 'center', fontsize = 12, color = 'gray')


ax.text(0.48,0.435, 'Strait of Juan de Fuca', rotation = -21, transform=ax.transAxes, ha= 'center', fontsize = 10, color = 'black')

ax.text(0.59,0.58, 'Strait of Georgia', rotation = -38, transform=ax.transAxes, ha= 'center', fontsize = 10, color = 'black')

ax.text(0.41,0.54, 'Vancouver Island\n(BC)', transform=ax.transAxes, multialignment= 'center', ha= 'center', fontsize = 8, color = 'gray')

#ax.text(0.8,0.09, 'Columbia River', transform=ax.transAxes, multialignment= 'center', ha= 'center', fontsize = 8, color = 'gray')



pfun.add_coast(ax)

pfun.dar(ax)


ax = axd['map_source']
 
ax.set_xlim(X[i1],-121.4)#X[i2]) # Salish Sea
ax.set_ylim(Y[j1],Y[j2]) # Salish Sea
        
ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

#sns.scatterplot(data=odf_ixiy_unique, x='lon', y='lat', ax = ax, color = 'gray', alpha=0.3, label= 'Cast Location')


pfun.add_coast(ax)

pfun.dar(ax)

for site in ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'point_jefferson', 'saratoga_passage_mid']:
    
    path = path_dict[site]
        
    if site in ['near_seattle_offshore']:
        
        patch = patches.PathPatch(path, facecolor=red, edgecolor='white', zorder=1, label='Main Basin')
    
    elif site in ['point_jefferson']:
            

        patch = patches.PathPatch(path, facecolor=red, edgecolor='white', zorder=1)
                
    elif site in ['saratoga_passage_mid']:
        
        patch = patches.PathPatch(path, facecolor=blue, edgecolor='white', zorder=1, label = 'Sub-Basins')
        
    else:
        
        patch = patches.PathPatch(path, facecolor=blue, edgecolor='white', zorder=1)
         
    ax.add_patch(patch)
    
sns.scatterplot(data=odf_ixiy_unique, x='lon', y='lat', ax = ax, color = 'gray', alpha=0.3, label= 'Cast Location')

    
ax.text(0.58,0.51, 'PJ', transform=ax.transAxes, fontsize=18, color = red, path_effects=[pe.withStroke(linewidth=4, foreground="white")])

ax.text(0.55,0.33, 'NS', transform=ax.transAxes, fontsize=18, color = red, path_effects=[pe.withStroke(linewidth=4, foreground="white")])

    
ax.text(0.64,0.69, 'SP', transform=ax.transAxes, fontsize=18, color = blue, path_effects=[pe.withStroke(linewidth=4, foreground="white")])

ax.text(0.22,0.29, 'LC', transform=ax.transAxes, fontsize=18, color = blue, path_effects=[pe.withStroke(linewidth=4, foreground="white")])
 
ax.text(0.48,0.2, 'CI', transform=ax.transAxes, fontsize=18, color = blue, path_effects=[pe.withStroke(linewidth=4, foreground="white")])

#ax.text(0.15,0.81, 'Strait of\nJuan de Fuca', transform=ax.transAxes, fontsize = 8, color = 'black', ha='center', va='center', rotation = -30)

#ax.text(0.3,0.85, '^ to Strait\nof Georgia', transform=ax.transAxes, fontsize = 7, color = 'black', ha='center', va='center')

ax.text(0.36,0.785, 'Admiralty\nInlet', transform=ax.transAxes, fontsize = 8, color = 'gray', ha='center', va='center')

ax.text(0.65,0.16, 'Tacoma\nNarrows', transform=ax.transAxes, fontsize = 8, color = 'gray', ha='center', va='center')


#ax.text(0.36,0.785, 'Deception Pass', transform=ax.transAxes, fontsize = 8, color = 'gray', ha='center', va='center')

ax.text(0.1,0.6, 'Puget\nSound', multialignment='center', transform=ax.transAxes, fontsize = 14, color = 'black')

ax.text(0.025,0.36, 'Hood Canal', transform=ax.transAxes, fontsize = 12, color = 'black', rotation = 55)

ax.text(0.57,0.1, 'South Sound', transform=ax.transAxes, fontsize = 12, color = 'black')

ax.text(0.77,0.5, 'Main Basin', transform=ax.transAxes, fontsize = 12, color = 'black', rotation = 50)

ax.text(0.83,0.73, 'Whidbey Basin', transform=ax.transAxes, fontsize = 12, color = 'black', rotation = -70)
 
ax.text(0.86,0.95, 'Skagit\nRiver', transform=ax.transAxes, fontsize = 8, color = 'gray', ha='center', va='center')


 

#ax.text(0.05,0.025, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')


ax.plot([-122.65,-122.65],[48.35, 48.45], color = 'black', linestyle='--', linewidth=3)

ax.plot([-122.8,-122.7],[48.1, 48.2], color = 'black', linestyle='--', linewidth=3)



ax.plot([-122.75,-122.55],[47.95, 47.9], color = 'gray', linestyle='--', linewidth=2)

ax.plot([-122.61,-122.49],[47.37, 47.27], color = 'gray', linestyle='--', linewidth=2)

ax.plot([-122.61,-122.49],[47.37, 47.27], color = 'gray', linestyle='--', linewidth=2)

ax.plot([-122.40,-122.27],[47.95, 47.87], color = 'gray', linestyle='--', linewidth=2)



 
ax.legend(loc = 'upper left')

ax.set_xlim(-123.2, -122.1) 
 
ax.set_ylim(47,48.5)


ax.set_xlabel('')

ax.set_ylabel('')
 
#xlbl = ax.get_xticklabels()

ax.set_xticks([-123.0, -122.6, -122.2], ['-123.0','-122.6', '-122.2']) #['','-123.0', '', '-122.6', '', '-122.2'])

# 🔑 AFTER everything is set, make ax_big match the *box* proportions of ax_src
fig.canvas.draw()  # let constrained_layout finalize positions first

pos = ax.get_position()          # in figure coordinates
ratio = pos.height / pos.width       # height / width of the reference axis box

axd['map_big'].set_box_aspect(ratio)         # only change the NON-reference axis

axd['map_big'].set_xticks([-125, -124, -123], ['-125','-124', '-123']) #['','-123.0', '', '-122.6', '', '-122.2'])


        
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/paper_1_fig_3.png', bbox_inches='tight', dpi=500, transparent=True)
    
