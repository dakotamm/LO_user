#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finalized for publication: 2026/01/07

Author: Dakota Mascarenas

Plotting code for: "Century-Scale Changes in Dissolved Oxygen, Temperature, and Salinity in Puget Sound" (Mascarenas et al., in review; submitted 2026/01/09 to Estuaries & Coasts)

This script processes data for and plots Figure 1 in corresponding manuscript. Please reach out to the author at dakotamm@uw.edu for any questions.

"""

# import modules
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import figure_functions_x as ffun

### FIGURE 1

# load pickled data frame for all Puget Sound cast data from user-specified directory
df_directory = '/Users/dakotamascarenas/Desktop/Mascarenas_etal_2026/' #SPECIFY LOCAL DIRECTORY
ps_casts_DF = pd.read_pickle(df_directory + 'ps_casts_DF.p')

# load grid attribute arrays from LiveOcean (MacCready et al., 2021; see text for more details and citations)
with open(df_directory + 'zm_inverse.p', 'rb') as fp:
    zm_inverse = pickle.load(fp)
with open(df_directory + 'plon.p', 'rb') as fp:
    plon = pickle.load(fp)
with open(df_directory + 'plat.p', 'rb') as fp:
    plat = pickle.load(fp)
    
# # load site polygon bounding paths
# with open(df_directory + 'site_polygon_dict.p', 'rb') as fp:
#     site_polygon_dict = pickle.load(fp)
    
# # get unique cast locations for all casts in Puget Sound given locations on LiveOcean grid (MacCready et al., 2021; see text for more details and citations)
# ps_cast_locations = ffun.get_cast_locations(ps_casts_DF)

# %%

from lo_tools import Lfun, zfun, zrfun

import xarray as xr

import numpy as np

import cmocean as cmo

gridname = 'cas7'

Ldir = Lfun.Lstart(gridname=gridname)

fng = Ldir['grid'] / 'grid.nc'
dsg = xr.open_dataset(fng)

m = dsg.mask_rho.values
depths = dsg.h.values
depths[m==0] = np.nan


mask_rho = np.transpose(dsg.mask_rho.values)
zm = -depths.copy()

zm[np.transpose(mask_rho) == 0] = np.nan
zm[np.transpose(mask_rho) != 0] = -1


# %%
# plot and save to user-specified directory
plot_directory = '/Users/dakotamascarenas/Desktop/pltz/' #SPECIFY SAVE LOCATION
site_list = ['point_jefferson', 'near_seattle_offshore', 'saratoga_passage_mid', 'carr_inlet_mid', 'lynch_cove_mid']
red =     "#4565e8"
blue =     "#e04256"
mosaic = [['map_big', 'map_ps']]
fig, axd = plt.subplot_mosaic(mosaic, figsize=(5,5), layout='constrained', gridspec_kw=dict(wspace=0.1))
ax = axd['map_big']
ax.pcolormesh(plon, plat, zm, linewidth=0.5, vmin=-1.25, vmax=0, cmap = 'Blues', zorder=-5)
ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)
ax.set_xlim(-127,-122)
ax.set_ylim(47,50)
# ax.scatter(-126.67, 48.65, color='k')
# ax.text(-126.55, 48.6, 'P4', fontsize = 10, color = 'black')
ax.plot([-123.2, -122.1], [47, 47], color='k', linewidth=0.5)
ax.plot([-123.2, -122.1], [48.5, 48.5], color='k', linewidth=0.5)
ax.plot([-123.2, -123.2], [47, 48.5], color='k', linewidth=0.5)
ax.plot([-122.1, -122.1], [47, 48.5], color='k', linewidth=0.5)
# ax.text(0.1,0.2, 'Pacific\nOcean', transform=ax.transAxes, fontsize = 14, color = 'black')
# ax.text(0.8,0.7, 'Salish\nSea', transform=ax.transAxes, multialignment= 'center', ha='center', fontsize = 14, color = 'black')
# ax.text(0.5,0.85, 'British Columbia,\nCanada', transform=ax.transAxes, multialignment= 'center', fontsize = 12, color = 'gray')
# ax.text(0.7,0.2, 'Washington,\nUSA', transform=ax.transAxes, multialignment= 'center', fontsize = 12, color = 'gray')
# ax.text(0.5,0.455, 'Strait of Juan de Fuca', rotation = -21, transform=ax.transAxes, ha= 'center', fontsize = 10, color = 'black')
# ax.text(0.69,0.55, 'Strait of Georgia', rotation = -36, transform=ax.transAxes, ha= 'center', fontsize = 10, color = 'black')
# ax.text(0.29,0.64, 'Vancouver Island\n(BC)', transform=ax.transAxes, multialignment= 'center', ha= 'center', fontsize = 8, color = 'gray')
# ax.text(0.05,0.025, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')
ffun.add_coast(ax, df_directory, linewidth=0.25)
ffun.dar(ax)
ax = axd['map_ps']
ax.pcolormesh(plon, plat, zm, linewidth=0.5, vmin=-1.25, vmax=0, cmap = 'Blues', zorder=-5)
ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)
ax.plot([-122.836, -122.1], [47.836, 47.836], color='k', linewidth=0.5)
ax.plot([-122.836, -122.1], [48.5, 48.5], color='k', linewidth=0.5)
ax.plot([-122.836, -122.836], [47.836, 48.5], color='k', linewidth=0.5)
ax.plot([-122.1, -122.1], [47.836, 48.5], color='k', linewidth=0.5)
# for site in site_list:
#     path = site_polygon_dict[site]      
#     if site in ['near_seattle_offshore']:
#         patch = patches.PathPatch(path, facecolor=red, edgecolor='white', zorder=1, label='Main Basin Sites')
#     elif site in ['point_jefferson']:
#         patch = patches.PathPatch(path, facecolor=red, edgecolor='white', zorder=1)
#     elif site in ['saratoga_passage_mid']:
#         patch = patches.PathPatch(path, facecolor=blue, edgecolor='white', zorder=1, label = 'Sub-Basin Sites')
#     else:
#         patch = patches.PathPatch(path, facecolor=blue, edgecolor='white', zorder=1)
#     ax.add_patch(patch)
# sns.scatterplot(data=ps_cast_locations, x='lon', y='lat', ax = ax, color = 'gray', alpha=0.3, label= 'Cast Locations')
# ax.text(0.58,0.51, 'PJ', transform=ax.transAxes, fontsize=18, color = red, path_effects=[pe.withStroke(linewidth=4, foreground="white")])
# ax.text(0.55,0.33, 'NS', transform=ax.transAxes, fontsize=18, color = red, path_effects=[pe.withStroke(linewidth=4, foreground="white")])
# ax.text(0.64,0.69, 'SP', transform=ax.transAxes, fontsize=18, color = blue, path_effects=[pe.withStroke(linewidth=4, foreground="white")])
# ax.text(0.22,0.29, 'LC', transform=ax.transAxes, fontsize=18, color = blue, path_effects=[pe.withStroke(linewidth=4, foreground="white")])
# ax.text(0.48,0.2, 'CI', transform=ax.transAxes, fontsize=18, color = blue, path_effects=[pe.withStroke(linewidth=4, foreground="white")])
# ax.text(0.36,0.785, 'Admiralty\nInlet', transform=ax.transAxes, fontsize = 8, color = 'gray', ha='center', va='center')
# ax.text(0.65,0.16, 'Tacoma\nNarrows', transform=ax.transAxes, fontsize = 8, color = 'gray', ha='center', va='center')
# ax.text(0.1,0.6, 'Puget\nSound', multialignment='center', transform=ax.transAxes, fontsize = 14, color = 'black')
# ax.text(0.025,0.36, 'Hood Canal', transform=ax.transAxes, fontsize = 12, color = 'black', rotation = 55)
# ax.text(0.57,0.1, 'South Sound', transform=ax.transAxes, fontsize = 12, color = 'black')
# ax.text(0.77,0.5, 'Main Basin', transform=ax.transAxes, fontsize = 12, color = 'black', rotation = 50)
# ax.text(0.83,0.73, 'Whidbey Basin', transform=ax.transAxes, fontsize = 12, color = 'black', rotation = -70)
# ax.text(0.86,0.95, 'Skagit\nRiver', transform=ax.transAxes, fontsize = 8, color = 'gray', ha='center', va='center')
# ax.text(0.05,0.025, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')
# ax.plot([-122.65,-122.65],[48.35, 48.45], color = 'black', linestyle='--', linewidth=3)
# ax.plot([-122.8,-122.7],[48.1, 48.2], color = 'black', linestyle='--', linewidth=3)
# ax.plot([-122.75,-122.55],[47.95, 47.9], color = 'gray', linestyle='--', linewidth=2)
# ax.plot([-122.61,-122.49],[47.37, 47.27], color = 'gray', linestyle='--', linewidth=2)
# ax.plot([-122.61,-122.49],[47.37, 47.27], color = 'gray', linestyle='--', linewidth=2)
# ax.plot([-122.40,-122.27],[47.95, 47.87], color = 'gray', linestyle='--', linewidth=2)
#ax.legend(loc = 'upper left')
ax.set_xlim(-123.2, -122.1) 
ax.set_ylim(47,48.5)
ffun.add_coast(ax, df_directory, linewidth=0.25)
ffun.dar(ax)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([-123.0, -122.6, -122.2], ['-123.0','-122.6', '-122.2'])
fig.canvas.draw()
position = ax.get_position()
height_div_width = position.height/position.width
axd['map_big'].set_box_aspect(height_div_width)
axd['map_big'].set_xticks([-126, -125, -124, -123], ['-126', '-125', '-124', '-123'])
plt.savefig(plot_directory + 'ss_ps.png', bbox_inches='tight', dpi=500, transparent=True)
    
