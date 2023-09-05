#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:51:55 2023

@author: dakotamascarenas
"""

import VFC_functions as vfun

import pandas as pd

import numpy as np

from lo_tools import forcing_argfun2 as ffun

from lo_tools import zrfun

from lo_tools import plotting_functions as pfun

import matplotlib.pyplot as plt



Ldir = ffun.intro() # this handles all the argument passing

min_lat = Lat[min(jjj) - 10]
max_lat = Lat[max(jjj) + 10]

min_lon = Lon[min(iii) - 10]
max_lon = Lon[max(iii) + 10]


dt = pd.Timestamp('2017-01-01 01:30:00')
fn_his = vfun.get_his_fn_from_dt(Ldir, dt)

G, S, T = zrfun.get_basic_info(fn_his)
land_mask = G['mask_rho']
Lon = G['lon_rho'][0,:]
Lat = G['lat_rho'][:,0]
plon,plat = pfun.get_plon_plat(G['lon_rho'], G['lat_rho'])
z_rho_grid, z_w_grid = zrfun.get_z(G['h'], 0*G['h'], S)
dz = np.diff(z_w_grid,axis=0)
dv = dz*G['DX']*G['DY']
h = G['h']

var_list = ['salt','temp','NO3','NH4','chlorophyll','TIC','alkalinity','oxygen']


for var in V.keys():
    
    if var in var_list:

        for cell in range(30):
        
            pfun.start_plot(fs=14, figsize=(15,15))
            fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)
            
            plot_var = V[var][0,cell, :,:]
            
            c0 = ax[0,0].pcolormesh(plon,plat, plot_var)
            
            # for cid in info_df.index:
                        
            #     ax[0,0].plot(info_df.loc[cid, 'lon'], info_df.loc[cid, 'lat'],'o', c = 'white', markeredgecolor='black', markersize=10) #c = cmap0(n)
                
            
            ax[0,0].set_xlim([min(plon[0]),max(plon[0])])
            ax[0,0].set_ylim([min(plat[0]), max(plat[-1])])
            ax[0,0].tick_params(labelrotation=45)
            ax[0,0].set_title(var + 'Layer ' + str(cell))
            
            fig.colorbar(c0,ax=ax[0,0])
            
            
            
            pfun.add_coast(ax[0,0])
            pfun.dar(ax[0,0])
    
            fig.tight_layout()
            plt.savefig('/Users/dakotamascarenas/Desktop/pltz/'+var + '_layer_' + str(cell) + '_OG.png', bbox_inches='tight')

        
        
        
# %%


for var in V.keys():
    
    if var in var_list:

        for cell in range(30):
        
            pfun.start_plot(fs=14, figsize=(25,15))
            fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=False)
            
            plot_var = V[var][0,cell, :,:]
            
            plot_var_OG = V_OG[var][0,cell, :, :]
            
            c0 = ax[0,0].pcolormesh(plon,plat, plot_var_OG)
            
            c1 = ax[0,1].pcolormesh(plon,plat, plot_var)
            
            
            if var == 'temp' or var == 'salt':

                for cid in info_df_ctd.index:
                            
                    ax[0,1].plot(info_df_ctd.loc[cid, 'lon'], info_df_ctd.loc[cid, 'lat'],'o', c = 'white', markeredgecolor='black', markersize=10) #c = cmap0(n)
                
            else:
                
                for cid in info_df.index:
                            
                    ax[0,1].plot(info_df.loc[cid, 'lon'], info_df.loc[cid, 'lat'],'o', c = 'white', markeredgecolor='black', markersize=10) #c = cmap0(n)
                
            if var == 'chlorophyll':
                
                units = '[mg m-3]'
                
                clim_min = 0.022
                
                clim_max = 0.028
                
            elif var == 'salt':
                
                units = '[PSU]'
                
                clim_min = 20
                
                clim_max = 35
                
            elif var == 'temp':
                
                units = '[deg C]'
                
                clim_min = 4
                
                clim_max = 14
                
            elif var == 'NO3':
                
                units = '[uM]'
                
                clim_min = 0
                
                clim_max = 45
                
            elif var == 'NH4':
                
                units = '[uM]'
                
                clim_min = -0.1
                
                clim_max = 0.1

            elif var == 'TIC':

                units = '[uM]'

                clim_min = 1700

                clim_max = 2500
                
            elif var == 'alkalinity':
                
                units = '[uM]'
                
                clim_min = 1900
                
                clim_max = 2500
                
            elif var == 'oxygen':
                
                units = '[uM]'
                
                clim_min = 0
                
                clim_max = 350
            
            ax[0,0].set_xlim([min(plon[0]),max(plon[0])])
            ax[0,0].set_ylim([min(plat[0]), max(plat[-1])])
            ax[0,0].tick_params(labelrotation=45)
            ax[0,0].set_title('original ' + var + ' ' + units + ' layer ' + str(cell))
            
            ax[0,1].set_xlim([min(plon[0]),max(plon[0])])
            ax[0,1].set_ylim([min(plat[0]), max(plat[-1])])
            ax[0,1].tick_params(labelrotation=45)
            ax[0,1].set_title('new ' + var + ' ' + units + ' layer ' + str(cell))
            
            c0.set_clim(clim_min, clim_max)
            
            c1.set_clim(clim_min, clim_max)
            
            cb0 = fig.colorbar(c0, ax = ax[0,0])
            
            
            cb1 = fig.colorbar(c1, ax = ax[0,1])
            
                        
            
            pfun.add_coast(ax[0,0])
            pfun.dar(ax[0,0])
            
            pfun.add_coast(ax[0,1])
            pfun.dar(ax[0,1])
    
    
            fig.tight_layout()
            plt.savefig('/Users/dakotamascarenas/Desktop/pltz/'+var + '_comp_layer_{:04d}.png'.format(cell), bbox_inches='tight')

                    